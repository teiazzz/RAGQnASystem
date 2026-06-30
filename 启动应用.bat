@echo off
setlocal
title RAGQnA Medical Assistant

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

echo ============================================
echo   RAGQnA Medical Assistant - React Startup
echo ============================================
echo.

echo [1/4] Starting Neo4j container...
docker start ragqna-neo4j >nul 2>&1
if errorlevel 1 (
    echo [WARN] Neo4j container ragqna-neo4j was not started.
    echo        If Neo4j is not running, KG retrieval will be degraded.
) else (
    echo       Neo4j started.
)
echo.

echo [2/4] Starting PostgreSQL and Redis...
docker compose up -d postgres redis
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start PostgreSQL/Redis. Please open Docker Desktop and retry.
    echo.
    pause
    exit /b 1
)
echo.

echo [3/4] Starting FastAPI backend on http://localhost:8000 ...
start "RAGQnA API :8000" /D "%APP_DIR%" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
echo.

echo [4/4] Starting React frontend on http://127.0.0.1:5173 ...
if not exist "%APP_DIR%frontend\node_modules" (
    echo [WARN] frontend\node_modules not found. Installing dependencies first...
    pushd "%APP_DIR%frontend"
    call npm.cmd install
    popd
)
start "RAGQnA Frontend :5173" /D "%APP_DIR%frontend" cmd /k "npm.cmd run dev -- --host 127.0.0.1 --port 5173"
echo.

echo Startup commands have been launched.
echo Frontend: http://127.0.0.1:5173/
echo API docs:  http://localhost:8000/docs
echo.
echo Streamlit is no longer started by this script.
echo Close the API/Frontend command windows to stop the app.
echo.
pause
