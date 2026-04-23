@echo off
REM run_windows.bat — Launch micropp training on Windows
REM ======================================================
REM Run this from the MICROPP project root folder:
REM   .\run_windows.bat
REM
REM Or run from PowerShell:
REM   cmd /c run_windows.bat

echo.
echo  micropp — starting pipeline training on Windows
echo  ================================================
echo.

REM ── MASTER_ADDR / MASTER_PORT ─────────────────────────────────────────────
REM These tell the 4 workers where to meet and coordinate.
REM 127.0.0.1 = localhost (your own machine).
REM Port 29501 is arbitrary — just needs to be a free port.
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29501

REM ── DO NOT set GLOO_SOCKET_IFNAME ─────────────────────────────────────────
REM Gloo on Windows uses libuv which does NOT accept "loopback" or custom
REM names. Leaving this unset lets Gloo resolve the interface from
REM MASTER_ADDR automatically — which works correctly on Windows.

REM ── Silence the noisy c10d warning ────────────────────────────────────────
REM Workers print "[c10d] socket failed" while waiting for each other.
REM These are harmless — training still works. This suppresses them.
set TORCH_CPP_LOG_LEVEL=0

echo  MASTER_ADDR = %MASTER_ADDR%
echo  MASTER_PORT = %MASTER_PORT%
echo.
echo  Starting 4 workers...
echo  Open http://localhost:5005 in your browser for the dashboard.
echo.

torchrun --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29501 src/main.py

echo.
echo  Training finished.
pause
