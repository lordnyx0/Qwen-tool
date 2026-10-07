@echo off
setlocal
REM Build do Qwen Control Center (PyInstaller one-dir)
cd /d "%~dp0"
echo Fechando instancias abertas do QwenControlCenter para liberar DLLs...
taskkill /F /IM QwenControlCenter.exe >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1

where pyinstaller >nul 2>&1
if errorlevel 1 (
  echo Instalando PyInstaller...
  python -m pip install pyinstaller
)
pyinstaller --noconfirm --clean QwenControlCenter.spec
echo.
echo EXE em: dist\QwenControlCenter\QwenControlCenter.exe
pause
