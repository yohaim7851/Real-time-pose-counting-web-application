@echo off
title Setup Windows Firewall for PoseRAC
echo ======================================
echo Setting up Windows Firewall for PoseRAC
echo ======================================
echo.

echo This script will add firewall rules to allow external access to PoseRAC.
echo You need to run this as Administrator.
echo.
pause

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    echo Error: This script must be run as Administrator.
    echo Right-click the script and select "Run as administrator"
    pause
    exit /b 1
)

echo Adding firewall rules...

REM Add inbound rules for TCP ports
netsh advfirewall firewall add rule name="PoseRAC FastAPI Backend" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="PoseRAC Streamlit Frontend" dir=in action=allow protocol=TCP localport=8501

echo.
echo Firewall rules added successfully!
echo.
echo The following ports are now open for external access:
echo - Port 8000 (FastAPI Backend)
echo - Port 8501 (Streamlit Frontend)
echo.
echo You can now start the service with start_external.bat
echo.
pause