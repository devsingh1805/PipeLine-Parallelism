@echo off
REM run_monolith_windows.bat — Run the single-process baseline on Windows
REM ======================================================================
REM Run this to train the monolith (ground-truth) version.
REM Compare its final loss to the pipeline version.

echo.
echo  micropp — starting monolith (single process) training
echo  =======================================================
echo.

python src/monolith.py

echo.
echo  Done.
pause
