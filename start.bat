@echo off
title PoseRAC Exercise Counter - External Access Mode
echo ======================================
echo PoseRAC Exercise Counter (External Access)
echo ======================================
echo.

REM Get current IP address
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr "IPv4"') do (
    for /f "tokens=1" %%j in ("%%i") do (
        set "IP=%%j"
        goto :found
    )
)
:found

echo Starting services for external access...
echo Your IP address: %IP%
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed!
    echo Please install Docker Desktop and try again.
    pause
    exit /b 1
)

echo Building and starting Docker container...
docker-compose -f docker-compose-external.yml up --build

echo.
echo Services are now accessible from other PCs:
echo.
echo Streamlit Frontend: http://%IP%:8501
echo FastAPI Backend: http://%IP%:8000
echo API Documentation: http://%IP%:8000/docs
echo Real-time Interface: http://%IP%:8000/real_time
echo.
echo To access from another PC, use the IP address above.
echo Make sure Windows Firewall allows connections on ports 8000 and 8501.
echo.
pause