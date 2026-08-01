@echo off
REM ============================================================
REM   Pipeline PDF -> TXT  (DeepDoc + VietOCR)
REM   - Copy file PDF vao thu muc "input"
REM   - Double-click run.bat (hoac chay trong cmd)
REM   - Ket qua TXT nam trong thu muc "output"
REM ============================================================

REM Dat code page UTF-8 de hien thi dung tieng Viet
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Pipeline PDF -^> TXT  (DeepDoc + VietOCR)
echo ============================================================

REM Chon Python: dung venv neu co, nguoc lai dung python he thong
set "PY=python"
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
)
echo Dung Python: %PY%

REM Tao thu muc input neu chua co
if not exist "input" (
    mkdir "input"
    echo [i] Da tao thu muc "input".
)

REM Neu thu muc input rong -> huong dan nguoi dung
dir /b "input\*" >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [!] Thu muc "input" dang trong.
    echo     Vui long copy file PDF vao thu muc "input" roi chay lai run.bat.
    echo.
    pause
    exit /b 0
)

echo.
REM Chay pipeline voi output khong dem bo dem (unbuffered) de xem tien trinh truc tiep
"%PY%" -u pdf_to_txt.py --inputs "./input" --output_dir "./output"

echo.
pause
