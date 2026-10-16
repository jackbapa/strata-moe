@echo off
rem Strata-Moe installer. Run again to resume an interrupted model download.
setlocal
title Strata-Moe Setup
cd /d "%~dp0"
echo.
echo  strata-moe setup
echo  First run: Python environment, A3B engine build, verified IQ3_S model download.
echo  Missing C++ build tools or CUDA toolkit are offered through winget.
echo  Full-GPU mode also downloads the matching MTP model.
echo  Later: double-click START-HERE.bat to reuse everything and open the local web app.
echo  Options: --no-start installs only; --mode cpu, 6g, hybrid, or full selects a profile.
echo  Default mode is auto: 20 GiB+ full; 4 to under 20 GiB hybrid; otherwise CPU.
echo  To switch an existing profile, stop the server and add --reconfigure.
echo  Default web: http://127.0.0.1:8096/  API: http://127.0.0.1:8096/v1
echo  Close this window or press Ctrl+C to stop the server. See README.md for details.
echo.
if exist ".venv\Scripts\python.exe" goto run
set "PY="
set "PY_ARGS="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 (set "PY=py" & set "PY_ARGS=-3")
if defined PY goto venv
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto venv
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
if defined PY goto venv
echo Installing Python 3.12 for the current user...
where winget >nul 2>nul
if not errorlevel 1 winget install -e --id Python.Python.3.12 --scope user --silent --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 (set "PY=py" & set "PY_ARGS=-3")
if defined PY goto venv
for %%V in (312 313) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
if defined PY goto venv
echo Install 64-bit Python 3.10+ from https://www.python.org/downloads/ and run SETUP.bat again.
pause
exit /b 1
:venv
"%PY%" %PY_ARGS% -m venv .venv
if errorlevel 1 goto failed
:run
".venv\Scripts\python.exe" tools\setup_moe.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Strata-Moe setup stopped. Read the error above and rerun SETUP.bat to resume.
pause
exit /b 1
