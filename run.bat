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
REM Cau hinh (khuyen dung): zoomin 8 + max_long_edge 5200 (~445 DPI) + det_limit_side 2048
REM -> text nho rai rac (cau hoi, 4 dap an A-D) duoc tach rieng, khong dinh line.
REM render 445DPI + detector 2048: tach du 20/20 box tren trang scan de thi mau.
"%PY%" -u pdf_to_txt.py --inputs "./input" --output_dir "./output" --zoomin 8 --max_long_edge 5200 --det_limit_side 2048

echo.
pause
