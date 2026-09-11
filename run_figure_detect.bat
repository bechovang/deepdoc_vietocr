@echo off
REM ============================================================
REM   Pipeline PDF -> TXT + TACH HINH (figure/diagram)
REM   Giong run.bat nhung them --figures:
REM   - Vung diagram/bieu do/anh (OCR khong xu ly duoc) duoc cat
REM     luu PNG vao output\<ten_file>_figs\
REM   - Trong TXT co marker [HÌNH k: duong_dan] dung vi tri
REM   - Khong anh huong gi den run.bat thong thuong
REM ============================================================

REM Dat code page UTF-8 de hien thi dung tieng Viet
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Pipeline PDF -^> TXT  +  TACH HINH  (DeepDoc + VietOCR)
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
    echo     Vui long copy file PDF vao thu muc "input" roi chay lai run_figure_detect.bat.
    echo.
    pause
    exit /b 0
)

echo.
REM Cung cau hinh khuyen dung nhu run.bat + bat them --figures
"%PY%" -u pdf_to_txt.py --inputs "./input" --output_dir "./output" --zoomin 8 --max_long_edge 5200 --det_limit_side 2048 --figures

echo.
pause
