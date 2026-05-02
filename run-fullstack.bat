@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo === Rice disease app: build frontend + run API + static UI ===
echo Usage: run-fullstack.bat   ^|   run-fullstack.bat nobuild   ^(skip npm build^)
echo.

if not exist "myenv\Scripts\python.exe" (
  echo ERROR: Python venv not found at "%~dp0myenv\Scripts\python.exe"
  echo Create it and install backend\requirements.txt, then run this script again.
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo ERROR: npm is not on PATH. Install Node.js, then run this script again.
  exit /b 1
)

if /i "%~1"=="nobuild" (
  echo [1/2] Skipping frontend build ^(nobuild^).
  if not exist "frontend\dist\index.html" (
    echo ERROR: frontend\dist\index.html missing. Run without nobuild once.
    exit /b 1
  )
) else (
  echo [1/2] npm run build ^(frontend^)
  cd frontend
  call npm run build
  if errorlevel 1 (
    echo ERROR: Frontend build failed.
    exit /b 1
  )
  cd ..
)

echo.
echo [2/2] uvicorn main:app ^(backend — serves API + dist on port 8000^)
echo Open: http://127.0.0.1:8000
echo Press Ctrl+C to stop.
echo.
cd backend
call "..\myenv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

endlocal
