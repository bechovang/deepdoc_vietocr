"""
Build script: dong goi DeepDoc + VietOCR thanh EXE (One-Dir) bang PyInstaller.
Chay: python build_exe.py
Output: ./dist/DeepDoc_VietOCR/
"""

import os
import sys
import subprocess
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_FILE = os.path.join(ROOT_DIR, "deepdoc_vietocr.spec")
DIST_DIR = os.path.join(ROOT_DIR, "dist")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
VENV_PYTHON = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")


def find_python():
    """Tim Python: uu tien venv, sau do he thong."""
    if os.path.isfile(VENV_PYTHON):
        print(f"[*] Dung Python venv: {VENV_PYTHON}")
        return VENV_PYTHON
    return sys.executable


def ensure_pyinstaller(python_exe):
    """Dam bao PyInstaller da duoc cai dat."""
    result = subprocess.run(
        [python_exe, "-m", "pip", "list", "--format=columns"],
        capture_output=True, text=True
    )
    if "pyinstaller" not in result.stdout.lower():
        print("[*] Dang cai PyInstaller...")
        subprocess.run([python_exe, "-m", "pip", "install", "pyinstaller"], check=True)
        print("[*] Cai PyInstaller xong!")
    else:
        print("[*] Da co PyInstaller.")


def clean_build():
    """Xoa build/ va dist/ cu."""
    for d in [BUILD_DIR, DIST_DIR]:
        if os.path.isdir(d):
            print(f"[*] Xoa thu muc cu: {d}")
            shutil.rmtree(d)


def build():
    """Chay PyInstaller build."""
    python_exe = find_python()
    ensure_pyinstaller(python_exe)

    print()
    print("=" * 56)
    print("  Build EXE: DeepDoc + VietOCR")
    print("=" * 56)
    print(f"  Python   : {python_exe}")
    print(f"  Spec     : {SPEC_FILE}")
    print(f"  Output   : {DIST_DIR}")
    print("=" * 56)
    print()

    # Chay PyInstaller
    cmd = [python_exe, "-m", "PyInstaller", "--clean", SPEC_FILE]
    print(f"[*] Chay: {' '.join(cmd)}")
    print("[*] Qua trinh build co the mat 10-30 phut...\n")

    subprocess.run(cmd, check=True)

    # Kiem tra ket qua
    output_exe = os.path.join(DIST_DIR, "DeepDoc_VietOCR")
    if os.path.isdir(output_exe):
        print()
        print("=" * 56)
        print(f"  BUILD THANH CONG!")
        print(f"  Output: {output_exe}")
        print("=" * 56)
        print()
        print("  Cac file chinh:")
        for f in sorted(os.listdir(output_exe)):
            fpath = os.path.join(output_exe, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"    {f:<30s} {size_mb:.1f} MB")
        print()
        print("  Cach dung:")
        print("    1. Mo thu muc dist/DeepDoc_VietOCR/")
        print("    2. Tao thu muc input/ va output/ ben canh .exe")
        print("    3. Chay DeepDoc_VietOCR.exe")
        print()
        print("  Hoac dung CLI:")
        print("    pdf_to_txt.exe --inputs ./input --output_dir ./output")
    else:
        print(f"\n[X] BUILD THAT BAI! Khong tim thay: {output_exe}")
        sys.exit(1)


if __name__ == "__main__":
    build()
