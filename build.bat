@echo off
REM ============================================================================
REM Unfold Portable - Windows Build Script
REM ============================================================================
REM This script builds Unfold as a standalone Windows executable for USB drives.
REM
REM Prerequisites:
REM   - Python 3.11+ installed
REM   - pip install pyinstaller
REM
REM Usage:
REM   build.bat           - Build the executable
REM   build.bat clean     - Clean build artifacts
REM   build.bat full      - Clean and rebuild everything
REM ============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   Unfold Portable - Build Script
echo ============================================================
echo.

REM Check for command line arguments
if "%1"=="clean" goto :clean
if "%1"=="full" (
    call :clean
    goto :build
)

:build
echo [1/6] Checking Python installation...
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    goto :error
)

echo [2/6] Checking PyInstaller...
pip show pyinstaller >NUL 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        goto :error
    )
)

echo [3/6] Installing required dependencies...
pip install -r requirements_portable.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may not have installed correctly
    echo Continuing with build...
)

echo [4/6] Creating version info...
echo VSVersionInfo( > version_info.txt
echo   ffi=FixedFileInfo( >> version_info.txt
echo     filevers=(1, 0, 0, 0), >> version_info.txt
echo     prodvers=(1, 0, 0, 0), >> version_info.txt
echo     mask=0x3f, >> version_info.txt
echo     flags=0x0, >> version_info.txt
echo     OS=0x40004, >> version_info.txt
echo     fileType=0x1, >> version_info.txt
echo     subtype=0x0, >> version_info.txt
echo     date=(0, 0) >> version_info.txt
echo   ), >> version_info.txt
echo   kids=[ >> version_info.txt
echo     StringFileInfo( >> version_info.txt
echo       [ >> version_info.txt
echo         StringTable( >> version_info.txt
echo           u'040904B0', >> version_info.txt
echo           [StringStruct(u'CompanyName', u'Unfold'), >> version_info.txt
echo            StringStruct(u'FileDescription', u'Unfold - PDF Knowledge Graph Extraction'), >> version_info.txt
echo            StringStruct(u'FileVersion', u'1.0.0'), >> version_info.txt
echo            StringStruct(u'InternalName', u'Unfold'), >> version_info.txt
echo            StringStruct(u'OriginalFilename', u'Unfold.exe'), >> version_info.txt
echo            StringStruct(u'ProductName', u'Unfold Portable'), >> version_info.txt
echo            StringStruct(u'ProductVersion', u'1.0.0')]) >> version_info.txt
echo       ]), >> version_info.txt
echo     VarFileInfo([VarStruct(u'Translation', [1033, 1200])]) >> version_info.txt
echo   ] >> version_info.txt
echo ) >> version_info.txt

echo [5/6] Building executable with PyInstaller...
echo.
python -m PyInstaller unfold.spec --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    goto :error
)

echo [6/6] Creating USB package structure...
if not exist "dist\Unfold_Portable" mkdir "dist\Unfold_Portable"

REM Copy executable
if exist "dist\Unfold.exe" (
    copy "dist\Unfold.exe" "dist\Unfold_Portable\" >NUL
) else if exist "dist\Unfold\Unfold.exe" (
    xcopy "dist\Unfold\*" "dist\Unfold_Portable\" /E /Y /Q >NUL
)

REM Create launcher
echo @echo off > "dist\Unfold_Portable\Start_Unfold.bat"
echo cd /d "%%~dp0" >> "dist\Unfold_Portable\Start_Unfold.bat"
echo start "" Unfold.exe >> "dist\Unfold_Portable\Start_Unfold.bat"

REM Create portable marker file
echo Unfold Portable Edition > "dist\Unfold_Portable\portable.marker"
echo This folder contains user data when running from USB. > "dist\Unfold_Portable\portable.marker"

REM Create data directory structure
if not exist "dist\Unfold_Portable\unfold_data" mkdir "dist\Unfold_Portable\unfold_data"
if not exist "dist\Unfold_Portable\unfold_data\graphs" mkdir "dist\Unfold_Portable\unfold_data\graphs"
if not exist "dist\Unfold_Portable\unfold_data\cache" mkdir "dist\Unfold_Portable\unfold_data\cache"

REM Create default config
echo { > "dist\Unfold_Portable\unfold_data\config.json"
echo   "host": "127.0.0.1", >> "dist\Unfold_Portable\unfold_data\config.json"
echo   "port": 8080, >> "dist\Unfold_Portable\unfold_data\config.json"
echo   "open_browser": true, >> "dist\Unfold_Portable\unfold_data\config.json"
echo   "debug": false, >> "dist\Unfold_Portable\unfold_data\config.json"
echo   "ollama_host": "http://localhost:11434", >> "dist\Unfold_Portable\unfold_data\config.json"
echo   "llm_provider": "ollama", >> "dist\Unfold_Portable\unfold_data\config.json"
echo   "llm_model": "llama3.2" >> "dist\Unfold_Portable\unfold_data\config.json"
echo } >> "dist\Unfold_Portable\unfold_data\config.json"

REM Create README
echo Unfold Portable Edition > "dist\Unfold_Portable\README.txt"
echo ======================== >> "dist\Unfold_Portable\README.txt"
echo. >> "dist\Unfold_Portable\README.txt"
echo PDF Knowledge Graph Extraction Tool >> "dist\Unfold_Portable\README.txt"
echo. >> "dist\Unfold_Portable\README.txt"
echo QUICK START: >> "dist\Unfold_Portable\README.txt"
echo   1. Double-click Start_Unfold.bat or Unfold.exe >> "dist\Unfold_Portable\README.txt"
echo   2. Browser will open automatically to http://127.0.0.1:8080 >> "dist\Unfold_Portable\README.txt"
echo   3. Paste text to extract knowledge graph >> "dist\Unfold_Portable\README.txt"
echo. >> "dist\Unfold_Portable\README.txt"
echo FOR BEST LLM EXTRACTION: >> "dist\Unfold_Portable\README.txt"
echo   Install Ollama (https://ollama.ai) and run: >> "dist\Unfold_Portable\README.txt"
echo   ollama pull llama3.2 >> "dist\Unfold_Portable\README.txt"
echo. >> "dist\Unfold_Portable\README.txt"
echo CONFIGURATION: >> "dist\Unfold_Portable\README.txt"
echo   Edit unfold_data\config.json to change settings >> "dist\Unfold_Portable\README.txt"
echo. >> "dist\Unfold_Portable\README.txt"
echo DATA LOCATION: >> "dist\Unfold_Portable\README.txt"
echo   All data is stored in the unfold_data folder >> "dist\Unfold_Portable\README.txt"
echo   Copy this entire folder to move your data >> "dist\Unfold_Portable\README.txt"
echo. >> "dist\Unfold_Portable\README.txt"

echo.
echo ============================================================
echo   BUILD COMPLETE!
echo ============================================================
echo.
echo Output location: dist\Unfold_Portable\
echo.
echo To deploy to USB drive:
echo   1. Copy the entire "Unfold_Portable" folder to your USB drive
echo   2. Run "Start_Unfold.bat" to launch
echo.
echo For LLM-powered extraction, install Ollama on the host machine:
echo   https://ollama.ai
echo   ollama pull llama3.2
echo.
goto :end

:clean
echo Cleaning build artifacts...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "version_info.txt" del "version_info.txt"
if exist "__pycache__" rmdir /s /q "__pycache__"
echo Done.
if "%1"=="clean" goto :end
goto :eof

:error
echo.
echo Build failed. Please check the errors above.
pause
exit /b 1

:end
echo Press any key to exit...
pause >NUL
exit /b 0
