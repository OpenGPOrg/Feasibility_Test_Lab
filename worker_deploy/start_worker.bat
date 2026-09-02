@echo off
rem ==============================================================================
rem openGP Standalone Worker Launcher for Windows (Zero Admin / Offline)
rem ==============================================================================

set SCRIPT_DIR=%~dp0

if exist "%SCRIPT_DIR%python\python.exe" (
    set "PY_BIN=%SCRIPT_DIR%python\python.exe"
) else (
    set "PY_BIN=python"
)

echo ============================================================
echo   openGP Volunteer Worker Node (Windows Standalone Mode)
echo ============================================================
echo   Directory: %SCRIPT_DIR%
echo ============================================================

"%PY_BIN%" "%SCRIPT_DIR%worker_node.py" %*
pause
