"""
Equation OCR module using pix2tex (LaTeX-OCR).

Chuyển đổi ảnh công thức toán học thành mã LaTeX.
Sử dụng model ViT + Transformer của LaTeX-OCR.
"""

import os
import logging
import numpy as np
from PIL import Image

# Lazy-load model để tránh chậm khi import
_latex_model = None


def get_latex_model():
    """
    Khởi tạo model LaTeX-OCR (singleton, load 1 lần).
    Returns:
        model: LatexOCR instance
    """
    global _latex_model
    if _latex_model is None:
        logging.info("Loading LaTeX-OCR model (pix2tex)...")
        from pix2tex.cli import LatexOCR
        _latex_model = LatexOCR()
        logging.info("LaTeX-OCR model loaded successfully!")
    return _latex_model


def image_to_latex(img, temperature=0.0):
    """
    Nhận dạng công thức toán từ ảnh -> mã LaTeX.

    Args:
        img: PIL Image hoặc numpy array (RGB)
        temperature: float, nhiệt độ sampling (0 = deterministic)

    Returns:
        str: mã LaTeX của công thức, hoặc string rỗng nếu lỗi
    """
    try:
        model = get_latex_model()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.open(img) if isinstance(img, str) else img

        # LaTeX-OCR nhận PIL Image
        result = model(img)
        if result:
            return result.strip()
        return ""
    except Exception as e:
        logging.error(f"LaTeX-OCR error: {e}")
        return ""


def crop_equation_region(img, region):
    """
    Crop vùng công thức từ ảnh gốc dựa vào layout region.

    Args:
        img: PIL Image gốc
        region: dict chứa thông tin bounding box
            (hỗ trợ cả 'bbox' và 'x0','top','x1','bottom')

    Returns:
        PIL Image: ảnh đã crop
    """
    if "bbox" in region:
        x0, y0, x1, y1 = map(int, region["bbox"])
    else:
        x0, y0, x1, y1 = map(int, [
            region.get("x0", 0),
            region.get("top", 0),
            region.get("x1", 0),
            region.get("bottom", 0)
        ])
    # Pad nhẹ để tránh cắt mất ký tự
    pad = 4
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img.width, x1 + pad)
    y1 = min(img.height, y1 + pad)
    return img.crop((x0, y0, x1, y1))