@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo === Rice Classifier: install deps, build frontend, run API + static UI ===
echo Options:  nobuild   skip npm install/build ^(requires frontend\dist^)
echo           skippip   skip pip install ^(faster if deps already installed^)
echo Example:  run-fullstack.bat nobuild skippip
echo.

set "SKIPBUILD=0"
set "SKIPPIP=0"
for %%a in (%*) do (
  if /i "%%a"=="nobuild" set "SKIPBUILD=1"
  if /i "%%a"=="skippip" set "SKIPPIP=1"
  if /i "%%a"=="fast" set "SKIPPIP=1"
)

REM Python: prefer repo venv, else python on PATH
set "PY=%~dp0myenv\Scripts\python.exe"
if exist "%PY%" goto :have_py
set "PY=python"
where python >nul 2>&1
if not errorlevel 1 goto :have_py
echo ERROR: Python not found. Install Python 3.10+ on PATH, or create venv:
echo   python -m venv myenv
echo   myenv\Scripts\python.exe -m pip install -r backend\requirements.txt
exit /b 1

:have_py
where npm >nul 2>&1
if errorlevel 1 (
  echo ERROR: npm is not on PATH. Install Node.js LTS, then run this script again.
  exit /b 1
)

if not defined PORT set "PORT=8000"

if /i "%SKIPPIP%"=="1" goto :after_pip
echo [1/4] pip install -r backend\requirements.txt
"%PY%" -m pip install -r backend\requirements.txt
if errorlevel 1 (
  echo ERROR: Backend dependency install failed.
  exit /b 1
)
goto :after_pip_label
:after_pip
echo [1/4] Skipping pip ^(skippip / fast^).
:after_pip_label

if /i "%SKIPBUILD%"=="1" goto :nobuild
echo.
echo [2/4] npm install ^(frontend^)
cd frontend
call npm install
if errorlevel 1 (
  echo ERROR: npm install failed.
  exit /b 1
)
echo.
echo [3/4] npm run build ^(frontend^)
call npm run build
if errorlevel 1 (
  echo ERROR: Frontend build failed.
  exit /b 1
)
cd ..
goto :after_build
:nobuild
echo [2-3/4] Skipping npm install/build ^(nobuild^).
if not exist "frontend\dist\index.html" (
  echo ERROR: frontend\dist\index.html missing. Run without nobuild once.
  exit /b 1
)
:after_build

echo.
echo [4/4] uvicorn main:app — API + website on port %PORT%
echo Open: http://127.0.0.1:%PORT%   API: http://127.0.0.1:%PORT%/predict
echo Set PORT=8001 before running if 8000 is already in use.
echo Press Ctrl+C to stop.
echo.
cd backend
"%PY%" -m uvicorn main:app --host 0.0.0.0 --port %PORT%

endlocal
