"""
Figure export module (tách hình/diagram khỏi trang OCR).

Phát hiện vùng "Figure" (sơ đồ, biểu đồ, ảnh đồ vật - thứ OCR thuần không
xử lý được) bằng mô hình layout ONNX sẵn có (onnx/layout.onnx), cắt vùng
đó khỏi ảnh render độ phân giải cao, lưu thành PNG và sinh marker
[HÌNH k: duong_dan] để chèn vào file TXT đúng vị trí theo trục y.

Mirror pattern cua equation_mvp/equation_ocr.py (lazy-load model).
"""

import os
import logging

from PIL import Image

# Ngưỡng score để nhận 1 vùng là figure (khớp default threshold của full_pipeline)
FIGURE_SCORE_THR = 0.5
# Pad quanh bbox khi crop (tỉ lệ theo kích thước vùng figure) để không cắt mất
# đường nối/nhãn ở mép diagram. Test tren de thi FuOverflow: bbox cua model
# thuong thua ~5-10% moi canh so voi exhibit thuc te.
FIGURE_CROP_PAD_RATIO = 0.10
FIGURE_CROP_PAD_MIN = 8
# Bỏ qua vùng figure quá nhỏ (vụn vặt / nhiễu)
FIGURE_MIN_SIZE = 40

# Lazy-load model để tránh chậm khi import (model layout ~75MB)
_layout_recognizer = None


def get_layout_recognizer():
    """
    Khởi tạo LayoutRecognizer (singleton, load 1 lần).
    Returns:
        model: LayoutRecognizer4YOLOv10 instance (dùng onnx/layout.onnx)
    """
    global _layout_recognizer
    if _layout_recognizer is None:
        logging.info("Loading layout recognizer (figure detection)...")
        from module import LayoutRecognizer
        _layout_recognizer = LayoutRecognizer("layout")
        logging.info("Layout recognizer loaded successfully!")
    return _layout_recognizer


def detect_figures(img, thr=FIGURE_SCORE_THR):
    """
    Phát hiện các vùng figure trong 1 ảnh trang.

    Args:
        img: PIL Image hoặc numpy array (RGB) - render đầy đủ của trang
        thr: ngưỡng score (postprocess YOLOv10 lỏng bên trong nên phải tự lọc)

    Returns:
        list dict {"type": "figure", "bbox": [x0,y0,x1,y1], "score": float}
        (tọa độ pixel theo ảnh gốc; "figure caption" bị loại - để OCR text)
    """
    recognizer = get_layout_recognizer()
    layouts = recognizer.forward([img], thr=thr)[0] or []
    return [r for r in layouts
            if r.get("type", "").lower() in ("figure", "image")
            and r.get("score", 0.0) >= thr]


def _extract_native_image(stream):
    """
    Giải mã ảnh nhúng trong PDF (pdfplumber page.images[i]['stream']) về PIL.

    Hỗ trợ FlateDecode (RGB/Gray raw) và DCTDecode/JPXDecode (JPEG/JPEG2000).
    Trả None nếu không giải mã được (caller bỏ qua, không raise).
    """
    try:
        import io
        attrs = stream.attrs
        w = int(attrs.get("Width") or attrs.get("W") or 0)
        h = int(attrs.get("Height") or attrs.get("H") or 0)
        if w <= 0 or h <= 0:
            return None
        filt = str(attrs.get("Filter", ""))
        if "DCT" in filt or "JPX" in filt:
            return Image.open(io.BytesIO(stream.get_data())).convert("RGB")
        # FlateDecode / khong filter: raw samples
        data = stream.get_data()
        cs = str(attrs.get("ColorSpace", "DeviceRGB"))
        bpc = int(attrs.get("BitsPerComponent", 8) or 8)
        if bpc != 8:
            return None
        if "Gray" in cs:
            mode, need = "L", w * h
        elif "CMYK" in cs:
            mode, need = "CMYK", w * h * 4
        else:
            mode, need = "RGB", w * h * 3
        if len(data) < need:
            return None
        return Image.frombytes(mode, (w, h), data[:need]).convert("RGB")
    except Exception:
        return None


def _iou(a, b):
    """IoU cua 2 bbox [x0,y0,x1,y1]."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def detect_figures_page(page, page_img, thr=FIGURE_SCORE_THR):
    """
    Phát hiện vùng figure cho 1 trang PDF: gộp 2 nguồn -

    1. Detect tren render trang (bắt diagram vector / ảnh lớn chiếm trang).
    2. Detect tren TỪNG ẢNH NHÚNG giải mã ở độ phân giải gốc (vd đề thi:
       mỗi câu hỏi là 1 screenshot 1920x780 nhúng trong trang) rồi map bbox
       về tọa độ trang. Ở độ phân giải gốc, exhibit (topology) cho score
       figure cao hơn hẳn so với detect trên render trang đã downscale.

    Chỉ nhận nhãn 'figure'/'image'. Đặc biệt KHÔNG nhận 'reference' dù
    score cao: khảo sát 10 đề thi cho thấy vùng reference >= 0.5 trong
    screenshot câu hỏi LUÔN là logo/watermark template của nền tảng
    (vd logo FUO góc dưới-trái, cùng vị trí mọi trang) chứ không phải
    exhibit - nhận nhãn này sẽ crop ra logo rác trên mọi trang.

    Args:
        page: trang pdfplumber (dùng page.images để lấy ảnh nhúng);
              None -> chỉ detect mức trang
        page_img: PIL Image render đầy đủ của trang
        thr: ngưỡng score

    Returns:
        list dict {"type","bbox":[x0,y0,x1,y1],"score"} - tọa độ pixel
        theo page_img, đã dedupe (IoU > 0.5).
    """
    regions = [dict(r) for r in detect_figures(page_img, thr=thr)]

    if page is None:
        return regions

    try:
        embedded = list(page.images or [])
    except Exception:
        embedded = []

    sx = page_img.width / float(page.width or 1)     # px / pt
    sy = page_img.height / float(page.height or 1)
    for im in embedded:
        # Bo qua anh nhung chiem vai tri be (logo, chu ky...)
        try:
            area = (im["x1"] - im["x0"]) * (im["bottom"] - im["top"])
            if area < 0.05 * (page.width * page.height):
                continue
            native = _extract_native_image(im["stream"])
            if native is None:
                continue
            sub = get_layout_recognizer().forward([native], thr=thr)[0] or []
        except Exception:
            continue
        w, h = native.size
        for r in sub:
            t = r.get("type", "").lower()
            if t not in ("figure", "image"):
                continue
            if r.get("score", 0.0) < thr:
                continue
            bx0, by0, bx1, by1 = r["bbox"]
            # map toa do anh native -> pt tren trang -> pixel render trang
            px0 = (im["x0"] + bx0 / w * (im["x1"] - im["x0"])) * sx
            px1 = (im["x0"] + bx1 / w * (im["x1"] - im["x0"])) * sx
            py0 = (im["top"] + by0 / h * (im["bottom"] - im["top"])) * sy
            py1 = (im["top"] + by1 / h * (im["bottom"] - im["top"])) * sy
            mapped = {"type": "figure", "bbox": [px0, py0, px1, py1],
                      "score": r.get("score", 0.0)}
            # dedupe voi region da co (IoU > 0.5 -> giu cai score cao hon)
            dup = next((k for k in regions if _iou(k["bbox"], mapped["bbox"]) > 0.5),
                       None)
            if dup is None:
                regions.append(mapped)
            elif mapped["score"] > dup.get("score", 0.0):
                regions[regions.index(dup)] = mapped
    return regions


def crop_figure_region(img, region, pad_ratio=FIGURE_CROP_PAD_RATIO,
                       pad_min=FIGURE_CROP_PAD_MIN):
    """
    Crop vùng figure từ ảnh gốc dựa vào layout region.

    Pad tỷ lệ theo kích thước vùng (pad = max(pad_min, pad_ratio * cạnh))
    thay vì pad cố định: bbox model thường thua vài % so với exhibit thật
    (nhãn thiết bị, dòng lệnh terminal nằm ngoài bbox) nên crop thiếu
    là mất thông tin, còn crop rộng hơn chút vô hại.

    Args:
        img: PIL Image gốc
        region: dict chứa bounding box
            (hỗ trợ cả 'bbox' và 'x0','top','x1','bottom')
        pad_ratio: tỉ lệ pad theo kích thước bbox (mỗi cạnh)
        pad_min: pad tối thiểu (pixel) cho vùng nhỏ

    Returns:
        PIL Image: ảnh đã crop (clamp trong biên ảnh)
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
    pad_x = max(pad_min, int(pad_ratio * (x1 - x0)))
    pad_y = max(pad_min, int(pad_ratio * (y1 - y0)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img.width, x1 + pad_x)
    y1 = min(img.height, y1 + pad_y)
    return img.crop((x0, y0, x1, y1))


def extract_figures(img, fig_dir, page_label, rel_prefix=None, thr=FIGURE_SCORE_THR,
                    regions=None, page=None):
    """
    Phát hiện + cắt + lưu các vùng figure trong 1 ảnh trang.

    Args:
        img: PIL Image (render đầy đủ độ phân giải của trang)
        fig_dir: thư mục lưu PNG (tự tạo khi có figure đầu tiên)
        page_label: nhãn trang trong tên file (vd "tr003", "img")
        rel_prefix: đường dẫn tương đối từ file TXT tới fig_dir để đưa vào
                    marker; nếu None dùng tên basename của fig_dir
        thr: ngưỡng score phát hiện figure
        regions: danh sách region figure ĐÃ phát hiện trước đó (vd full_pipeline
                 đã chạy layout) - truyền vào để không chạy lại model
        page: trang pdfplumber - nếu có, detect thêm trên ảnh nhúng
              (xem detect_figures_page)

    Returns:
        list [(y0, marker, rel_path), ...] với y0 = đỉnh vùng figure (pixel),
        marker dạng "[HÌNH k: <rel_prefix>/<ten_file>]", rel_path = đường dẫn
        tương đối tới file PNG (dùng cho markdown ![...](rel_path)).
        Trả [] nếu không có figure hoặc có lỗi (không bao giờ raise -
        pipeline OCR chính phải chạy tiếp được).
    """
    try:
        if regions is None:
            regions = detect_figures_page(page, img, thr=thr)
        if not regions:
            return []
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        os.makedirs(fig_dir, exist_ok=True)
        if rel_prefix is None:
            rel_prefix = os.path.basename(os.path.normpath(fig_dir))

        markers = []
        # Sắp xếp theo y để đánh số hình từ trên xuống
        for region in sorted(regions, key=lambda r: r["bbox"][1]):
            fig_img = crop_figure_region(img, region)
            if fig_img.width < FIGURE_MIN_SIZE or fig_img.height < FIGURE_MIN_SIZE:
                continue
            k = len(markers) + 1
            fname = f"{page_label}_fig{k}.png"
            fig_img.save(os.path.join(fig_dir, fname))
            rel_path = f"{rel_prefix}/{fname}"
            markers.append((region["bbox"][1], f"[HÌNH {k}: {rel_path}]", rel_path))
        return markers
    except Exception as e:
        logging.warning(f"extract_figures error (bo qua, OCR van chay binh thuong): {e}")
        return []
