@echo off
REM ============================================================================
REM Unfold Portable - USB Drive Launcher
REM ============================================================================
REM Double-click this file to start Unfold from a USB drive.
REM All data will be stored in the unfold_data folder alongside this file.
REM ============================================================================

setlocal

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Set title
title Unfold - Knowledge Graph Extraction

echo.
echo  ============================================================
echo   Unfold Portable - Starting...
echo  ============================================================
echo.

REM Check if executable exists
if exist "Unfold.exe" (
    echo Starting Unfold.exe...
    start "" "Unfold.exe"
    goto :end
)

REM Check if we're in development mode
if exist "unfold_portable.py" (
    echo Starting in development mode...
    python unfold_portable.py
    goto :end
)

REM Check dist folder
if exist "dist\Unfold_Portable\Unfold.exe" (
    echo Starting from dist folder...
    start "" "dist\Unfold_Portable\Unfold.exe"
    goto :end
)

echo ERROR: Unfold.exe not found!
echo.
echo Please build the executable first by running build.bat
echo Or run directly with: python unfold_portable.py
echo.
pause

:end
