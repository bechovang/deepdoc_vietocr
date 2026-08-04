# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec cho DeepDoc + VietOCR.
Dong goi kieu One-Dir: dist/DeepDoc_VietOCR/
Chay: pyinstaller deepdoc_vietocr.spec
"""

import os
import sys

from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import EXE, COLLECT, PYZ
from PyInstaller.building.datastruct import Tree

# ---------- Ten ung dung ----------
app_name = "DeepDoc_VietOCR"

# ---------- Danh sach data files ----------
# Thu muc model ONNX (~407 MB) - quan trong nhat
onnx_tree = Tree("onnx", prefix="onnx")

# File cau hinh va model bo tro
extra_datas = [
    ("onnx/ocr.res", "onnx"),
    ("onnx/updown_concat_xgb.model", "onnx"),
]

# ---------- Thu vien an (hidden imports) ----------
# PyInstaller khong tu dong phat hien nhung import dong
hidden_imports = [
    # Module cua du an
    "module.ocr",
    "module.ocr_onnx",
    "module.recognizer",
    "module.layout_recognizer",
    "module.table_structure_recognizer",
    "module.operators",
    "module.postprocess",
    "module.seeit",
    "utils.file_utils",
    "utils.constants",
    "utils.settings",
    "utils.db.db_models",
    "utils.db.db_utils",
    # VietOCR
    "vietocr.tool.predictor",
    "vietocr.tool.config",
    "vietocr.model.vocab",
    "vietocr.model.backbone.cnn",
    "vietocr.model.backbone.vgg",
    "vietocr.model.seqmodel.seq2seq",
    "vietocr.model.transformerocr",
    # Thu vien ben thu ba
    "pdfplumber",
    "cachetools",
    "shapely",
    "pyclipper",
    "ruamel.yaml",
]

# ---------- Exclude nhung thu khong can ----------
# Giam dung luong bang cach loai bo CUDA khoi torch + cac module test
excludes = [
    "torch.cuda",
    "torch.cuda.amp",
    "torch.backends.cuda",
    "torch.distributed",
    "torch.jit",
    "torch.nn.parallel",
    "torch.onnx",
    "torch.testing",
    "torch.utils.tensorboard",
    "torch.utils.benchmark",
    "matplotlib.tests",
    "numpy.testing",
    # Chi giu lai cac PIL codec can thiet
    "PIL.GifImagePlugin",
    "PIL.IcoImagePlugin",
    "PIL.MicImagePlugin",
    "PIL.MpoImagePlugin",
    "PIL.MspImagePlugin",
    "PIL.PalmImagePlugin",
    "PIL.PcdImagePlugin",
    "PIL.PcxImagePlugin",
    "PIL.PixarImagePlugin",
    "PIL.PsdImagePlugin",
    "PIL.SgiImagePlugin",
    "PIL.SunImagePlugin",
    "PIL.TgaImagePlugin",
    "PIL.WalImagePlugin",
    "PIL.XbmImagePlugin",
    "PIL.XpmImagePlugin",
]

# ---------- Phan tich code ----------
a = Analysis(
    scripts=["gui.py"],
    pathex=[],
    binaries=[],
    datas=extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

# Bo sung Tree cho onnx/ (model weights) vao datas
a.datas += onnx_tree

# Dong goi pure Python
pyz = PYZ(a.pure)

# EXE chinh (GUI, khong console)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # An console -> GUI mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Dong goi thanh 1 thu muc duy nhat
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)