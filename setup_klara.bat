@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"
set "KLARA_SYSTEM_PATH=%PATH%"

REM 1) Готовим виртуалку
if not exist ".venv" (
    echo Creating venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        goto :end
    )
)

echo Opening venv...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate venv
    echo [WARN] Falling back to the system Python.
    set "PATH=%KLARA_SYSTEM_PATH%"
)
python -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [WARN] The saved venv is incompatible with this PC. Using system Python.
    set "PATH=%KLARA_SYSTEM_PATH%"
)

REM 2) Зависимости для Python
echo Downloading dependencies...
python -m pip install --upgrade pip
if errorlevel 1 echo [WARN] Failed to upgrade pip
python -m pip install omegaconf sounddevice soundfile vosk PyQt5 torch requests psutil
if errorlevel 1 echo [WARN] Some deps failed to install

REM 3) Запускаем Ollama НЕ блокируя батник
echo Starting up Ollama ...
REM --- Вариант А: если ollama в PATH ---
REM start "Ollama" /MIN /B ollama serve
REM --- Вариант Б: если не в PATH — укажи полный путь ---
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    start "Ollama" /MIN "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
) else (
    start "Ollama" /MIN ollama serve
)

REM 4) Ждём, пока Ollama поднимется на 11434 (до ~30 сек)
echo Waiting for Ollama to be ready...
for /l %%i in (1,1,30) do (
    >nul 2>&1 curl http://127.0.0.1:11434/api/tags
    if not errorlevel 1 (
        echo Ollama is ready.
        goto :run_klara
    )
    timeout /t 1 >nul
)
echo [WARN] Ollama did not respond in time, continuing anyway...

:run_klara
echo Starting up Klara ^<3 ...
python klara.py
if errorlevel 1 (
    echo [ERROR] Klara crashed or failed to start
)

:end
echo.
echo Done.
pause
