@echo off
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv )
.venv\Scripts\pip install -r requirements.txt pyinstaller
.venv\Scripts\pyinstaller --noconfirm --windowed --name STALCRAFT_Trade_Assistant ^
  --add-data "assets;assets" main.py
echo.
echo Готово: dist\STALCRAFT_Trade_Assistant\STALCRAFT_Trade_Assistant.exe
echo Папка data\ (база, настройки) создаётся рядом с EXE.
