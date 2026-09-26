#!/usr/bin/env python3
"""
openGP OpenAI-Compatible API Server.
Exposes standard OpenAI REST API endpoints (/v1/chat/completions, /v1/models, etc.)
backed by the openGP peer-to-peer distributed LLM engine.
Includes full support for:
  - ChatML & Qwen2.5 native stop tokens (<|im_end|>, <|endoftext|>)
  - Tool calling / Function calling integration (passes tools to chat template & parses <tool_call> tags)
  - Repetition penalty to prevent degenerate token loops
  - Server-Sent Events (SSE) real-time streaming
"""

import json
import queue
import re
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


def get_local_ip() -> str:
    """Detect LAN IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas (OpenAI Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "default"
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = Field(default=512, alias="max_tokens")
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.1
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False

    class Config:
        extra = "allow"


class CompletionRequest(BaseModel):
    model: Optional[str] = "default"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 1.1
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False

    class Config:
        extra = "allow"


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_openai_app(dist_model) -> FastAPI:
    """Create a FastAPI app wired to a DistributedModel instance."""
    app = FastAPI(
        title="openGP Distributed LLM API",
        description="OpenAI-compatible inference server powered by openGP distributed clustering",
        version="1.1.0",
    )

    # Enable CORS for all origins (enables web UIs like Open-WebUI, Chatbox, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Thread lock to serialize token generation through the distributed cluster
    infer_lock = threading.Lock()

    def format_chat_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format messages using tokenizer chat template if available, else plain text."""
        # Sanitize messages (ensure content is string, handle tool call history)
        cleaned_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")
            if content is None:
                # If message contains tool_calls or function_call without content
                if "tool_calls" in m:
                    content = ""
                else:
                    content = ""
            msg_dict = {"role": role, "content": str(content)}
            if "tool_calls" in m:
                msg_dict["tool_calls"] = m["tool_calls"]
            if "name" in m:
                msg_dict["name"] = m["name"]
            if "tool_call_id" in m:
                msg_dict["tool_call_id"] = m["tool_call_id"]
            cleaned_msgs.append(msg_dict)

        # Check if caller already supplied a raw pre-formatted ChatML prompt
        if len(cleaned_msgs) == 1 and "<|im_start|>" in cleaned_msgs[0]["content"]:
            prompt = cleaned_msgs[0]["content"]
            if not prompt.rstrip().endswith("<|im_start|>assistant"):
                prompt = prompt.rstrip() + "\n<|im_start|>assistant\n"
            return prompt

        tokenizer = getattr(dist_model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            # Attempt to apply chat template with tools (Qwen2.5 native tool format)
            if tools:
                try:
                    return tokenizer.apply_chat_template(
                        cleaned_msgs,
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except TypeError:
                    pass
                except Exception:
                    pass

            try:
                return tokenizer.apply_chat_template(
                    cleaned_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Robust ChatML Fallback (keeps Qwen2.5 in native ChatML mode instead of raw text)
        lines = []
        if tools:
            tool_spec = (
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                + "\n".join(json.dumps(t) for t in tools)
                + "\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n</tool_call>"
            )
            lines.append(f"<|im_start|>system\n{tool_spec}<|im_end|>")
        for m in cleaned_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)

    def extract_tool_calls(text: str) -> tuple[Optional[List[Dict[str, Any]]], str]:
        """Extract Qwen2.5 / ChatML <tool_call>...</tool_call> blocks."""
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if not matches:
            return None, text

        tool_calls = []
        for match in matches:
            try:
                call_data = json.loads(match.strip())
                name = call_data.get("name", "")
                args = call_data.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                })
            except Exception:
                continue

        clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return (tool_calls if tool_calls else None), clean_text

    @app.get("/health")
    @app.get("/")
    def health():
        return {
            "status": "ok",
            "cluster_ready": dist_model.is_distributed,
            "model": dist_model.model_id or "unknown",
            "nodes_connected": len(dist_model.assignments),
        }

    @app.get("/v1/models")
    def list_models():
        model_name = dist_model.model_id or "openGP-cluster"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openGP",
                    "permission": [],
                    "root": model_name,
                    "parent": None,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if not dist_model.is_distributed:
            raise HTTPException(status_code=503, detail="Cluster distribution has not been applied yet.")

        prompt = format_chat_prompt(req.messages, tools=req.tools)
        max_tokens = req.max_completion_tokens or req.max_tokens or 512
        temperature = float(req.temperature if req.temperature is not None else 0.7)
        rep_penalty = float(req.repetition_penalty if req.repetition_penalty is not None else 1.1)
        model_id = req.model or dist_model.model_id or "openGP-cluster"
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        print(f"\n  [API REQ] Msgs: {len(req.messages)}, Tools: {len(req.tools) if req.tools else 0}, MaxTok: {max_tokens}, Temp: {temperature}, Stream: {req.stream}")
        if req.messages:
            last_role = req.messages[-1].get("role", "unknown")
            last_body = str(req.messages[-1].get("content", ""))[:140].replace("\n", " ")
            print(f"  [API REQ LAST MSG] {last_role}: {last_body}...")

        # Collect stop words
        stops = ["<|im_end|>", "<|endoftext|>", "\nHuman:", "\nUser:", "\nuser:", "\nSystem:", "\nsystem:", "\nassistant:", "Human:"]
        if req.stop:
            if isinstance(req.stop, str):
                stops.append(req.stop)
            elif isinstance(req.stop, list):
                stops.extend(req.stop)

        if not req.stream:
            with infer_lock:
                stats = dist_model.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stop=stops,
                    repetition_penalty=rep_penalty,
                )
            raw_text = stats.get("text", "")
            tool_calls, clean_text = extract_tool_calls(raw_text)

            choice_msg = {"role": "assistant"}
            if tool_calls:
                choice_msg["content"] = clean_text if clean_text else None
                choice_msg["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
                print(f"  [API RESP TOOL CALLS] {tool_calls}")
            else:
                choice_msg["content"] = raw_text
                finish_reason = "stop"
            print(f"  [API RESP] Tokens: {stats.get('num_output', 0)} | Finish: {finish_reason}")

            return {
                "id": req_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": choice_msg,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": stats.get("num_input", 0),
                    "completion_tokens": stats.get("num_output", 0),
                    "total_tokens": stats.get("total_seq", 0),
                },
            }

        # Streaming mode
        def stream_generator():
            token_q = queue.Queue()

            def _on_token(tok):
                token_q.put(tok)

            def _worker():
                try:
                    with infer_lock:
                        dist_model.generate(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            callback=_on_token,
                            stop=stops,
                            repetition_penalty=rep_penalty,
                        )
                except Exception as e:
                    token_q.put(e)
                finally:
                    token_q.put(None)  # Sentinel

            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()

            # First chunk: set role
            first_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            while True:
                item = token_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    err_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"\n[Error: {item}]"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    break

                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": item},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk
            final_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        if not dist_model.is_distributed:
            raise HTTPException(status_code=503, detail="Cluster distribution has not been applied yet.")

        prompt = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
        max_tokens = req.max_tokens or 512
        temperature = float(req.temperature if req.temperature is not None else 0.7)
        rep_penalty = float(req.repetition_penalty if req.repetition_penalty is not None else 1.1)
        model_id = req.model or dist_model.model_id or "openGP-cluster"
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        stops = ["<|im_end|>", "<|endoftext|>", "\nHuman:", "\nUser:", "\nuser:", "\nSystem:", "\nsystem:", "\nassistant:", "Human:"]
        if req.stop:
            if isinstance(req.stop, str):
                stops.append(req.stop)
            elif isinstance(req.stop, list):
                stops.extend(req.stop)

        if not req.stream:
            with infer_lock:
                stats = dist_model.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stop=stops,
                    repetition_penalty=rep_penalty,
                )
            response_text = stats.get("text", "")
            return {
                "id": req_id,
                "object": "text_completion",
                "created": created_ts,
                "model": model_id,
                "choices": [
                    {
                        "text": response_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": stats.get("num_input", 0),
                    "completion_tokens": stats.get("num_output", 0),
                    "total_tokens": stats.get("total_seq", 0),
                },
            }

        def stream_generator():
            token_q = queue.Queue()

            def _on_token(tok):
                token_q.put(tok)

            def _worker():
                try:
                    with infer_lock:
                        dist_model.generate(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            callback=_on_token,
                            stop=stops,
                            repetition_penalty=rep_penalty,
                        )
                except Exception as e:
                    token_q.put(e)
                finally:
                    token_q.put(None)

            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()

            while True:
                item = token_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    break
                chunk = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "text": item,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            yield f"data: {json.dumps({'id': req_id, 'object': 'text_completion', 'created': created_ts, 'model': model_id, 'choices': [{'text': '', 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    return app


def start_server(dist_model, host: str = "0.0.0.0", port: int = 8000):
    """Run the OpenAI API server."""
    app = create_openai_app(dist_model)
    local_ip = get_local_ip()

    print("\n" + "=" * 62)
    print("  🚀 openGP OpenAI-Compatible API Server Online!")
    print("=" * 62)
    print(f"  • Local Endpoint:    http://127.0.0.1:{port}/v1")
    print(f"  • Network Endpoint:  http://{local_ip}:{port}/v1")
    print(f"  • Active Model:      {dist_model.model_id or 'Cluster Default'}")
    print("=" * 62)
    print("  Features Active:")
    print("    ✓ Native ChatML stop tokens (<|im_end|>, <|endoftext|>)")
    print("    ✓ Tool / Function Calling support (passed to ChatML template)")
    print("    ✓ Repetition Penalty (1.1) to prevent loop degeneration")
    print("    ✓ Server-Sent Events (SSE) live streaming")
    print("=" * 62)
    print("  Press Ctrl+C anytime to stop server and return to menu.\n")

    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n  Shutting down OpenAI API server...")
