#!/usr/bin/env bash
# ==============================================================================
# openGP Standalone Worker Launcher (Zero Admin / Zero Docker / Fully Offline)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use bundled portable Python if present, otherwise fallback to system python3
if [ -f "$SCRIPT_DIR/python/bin/python3" ]; then
    export PYTHONHOME="$SCRIPT_DIR/python"
    export PATH="$SCRIPT_DIR/python/bin:$PATH"
    export LD_LIBRARY_PATH="$SCRIPT_DIR/python/lib:$SCRIPT_DIR/python/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
    PY_BIN="$SCRIPT_DIR/python/bin/python3"
elif [ -f "$SCRIPT_DIR/env/bin/python3" ]; then
    PY_BIN="$SCRIPT_DIR/env/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
else
    echo "✗ Error: No Python runtime found in bundle or system."
    exit 1
fi

echo "============================================================"
echo "  openGP Volunteer Worker Node (Offline Standalone Mode)"
echo "============================================================"
echo "  Runtime: $($PY_BIN --version 2>&1)"
echo "  Directory: $SCRIPT_DIR"
echo "============================================================"

exec "$PY_BIN" "$SCRIPT_DIR/worker_node.py" "$@"
