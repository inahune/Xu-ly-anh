@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo ========================================================
echo   DIRE - KHOI DONG HE THONG PHAT HIEN ANH AI (WEB)
echo ========================================================
echo.

set "PYTHON_EXE=python"
if exist "C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe"
)

echo [1/2] Kiem tra va tai checkpoint neu thieu...
"%PYTHON_EXE%" download_checkpoints.py
if errorlevel 1 (
    echo [!] Khong the tai checkpoint. Vui long kiem tra ket noi mang.
    pause
    exit /b 1
)

echo.
echo [2/2] Dang khoi dong Web Server...
"%PYTHON_EXE%" app.py
pause
