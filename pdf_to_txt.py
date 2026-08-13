#
#  Pipeline: PDF / anh  (trong ./input)  ->  TXT  (trong ./output)
#  Dung OCR tieng Viet (DeepDoc + VietOCR), chay tren CPU.
#
#  Cach dung:
#     python pdf_to_txt.py                          # dung mac dinh ./input -> ./output
#     python pdf_to_txt.py --inputs ./input --output_dir ./output
#
#  Hoac chi can chay file: run.bat
#

import argparse
import os
import sys
import threading
import time

import numpy as np
from PIL import Image

# -------------------------------------------------------------------
# Cau hinh moi truong
# -------------------------------------------------------------------
# Chay tren CPU (giong t_ocr.py)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Dam bao import duoc module/ va utils/ khi chay tu bat ky thu muc nao
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -------------------------------------------------------------------
# Cac dinh dang ho tro
# -------------------------------------------------------------------
SUPPORTED_IMG = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp')
PDF_EXT = '.pdf'

# pdfplumber khong thread-safe -> dung lock khi mo PDF
_PDF_LOCK = threading.Lock()


def page_resolution(page, zoomin, max_long_edge=4000):
    """
    Tinh do phan giai (DPI) cho 1 trang.
    - zoomin: DPI mong muon = 72 * zoomin (vi du zoomin=3 -> 216 DPI).
    - max_long_edge: canh dai lon nhat (pixel) de tranh render trang qua to
      (model OCR se tu downscale ve 960px nen render sieu to la lang phi RAM).
    """
    long_edge_pt = max(float(page.width), float(page.height))
    desired_dpi = 72.0 * zoomin
    max_dpi = max_long_edge * 72.0 / long_edge_pt if long_edge_pt > 0 else desired_dpi
    resolution = min(desired_dpi, max_dpi)
    return max(resolution, 72.0)  # it nhat 72 DPI


def process_pdf_streaming(ocr, pdf_path, out_txt, zoomin, limit=None, max_long_edge=2500):
    """
    Render + OCR tung trang va ghi vao file TXT ngay (streaming).
    - Chi giu 1 trang trong RAM tai 1 thoi diem -> khong bi treo voi PDF nhieu trang.
    - Ghi (flush) sau moi trang: neu dung giua chang (Ctrl+C) van giu duoc phan da lam.
    Tra ve (out_txt, so_trang_da_xu_ly).
    """
    import pdfplumber

    n_done = 0
    with _PDF_LOCK:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            target = total if not limit else min(total, limit)
            stream_start = time.time()
            with open(out_txt, 'w', encoding='utf-8') as f:
                for pi, page in enumerate(pdf.pages, 1):
                    if limit and pi > limit:
                        break
                    resolution = page_resolution(page, zoomin, max_long_edge=max_long_edge)
                    img = page.to_image(resolution=resolution).annotated
                    txt = image_to_text(ocr, img)
                    del img  # giai phong anh khoi RAM ngay

                    body = txt.strip() if txt.strip() else '(trang khong phat hien van ban)'
                    if target > 1:
                        f.write(f'===== Trang {pi}/{target} =====\n\n')
                    f.write(body)
                    f.write('\n\n')
                    f.flush()
                    n_done = pi
                    # ETA don gian
                    elapsed = time.time() - stream_start
                    eta = elapsed / pi * (target - pi) if pi else 0
                    print(f'    - OCR trang {pi}/{target} xong'
                          f'  ({elapsed:.0f}s da qua, con lai ~{eta:.0f}s)')
    return out_txt, n_done


def image_to_text(ocr, img):
    """
    OCR mot anh PIL -> tra ve chuoi text (cac dong cach nhau bang xuong dong).
    ocr(...) tra ve danh sach [(box, (text, score)), ...]
    """
    result = ocr(np.array(img))
    if not result:
        return ''
    lines = [item[1][0] for item in result if item[1] and item[1][0]]
    return '\n'.join(lines)


def collect_inputs(input_dir):
    """Lay danh sach file PDF + anh trong thu muc input (khong de quy)."""
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext == PDF_EXT or ext in SUPPORTED_IMG:
            files.append(path)
    return files


def process_file(ocr, fpath, output_dir, zoomin, limit=None, max_long_edge=2500):
    """
    Xu ly 1 file (PDF hoac anh) -> ghi 1 file TXT.
    - PDF: xu ly streaming tung trang (tiet kiem RAM).
    - Anh: OCR truc tiep.
    Tra ve (out_txt, so_trang) hoac raise loi.
    """
    name = os.path.basename(fpath)
    stem = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1].lower()
    out_txt = os.path.join(output_dir, stem + '.txt')

    if ext == PDF_EXT:
        out_txt, n_pages = process_pdf_streaming(ocr, fpath, out_txt, zoomin, limit, max_long_edge=max_long_edge)
    else:
        img = Image.open(fpath).convert('RGB')
        txt = image_to_text(ocr, img)
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(txt)
        n_pages = 1

    return out_txt, n_pages


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline PDF/anh -> TXT dung OCR tieng Viet (DeepDoc + VietOCR).')
    parser.add_argument('--inputs', default='./input',
                        help='Thu muc chua file PDF/anh dau vao. Mac dinh: ./input')
    parser.add_argument('--output_dir', default='./output',
                        help='Thu muc luu file TXT. Mac dinh: ./output')
    parser.add_argument('--zoomin', type=int, default=8,
                        help='Do phan giai khi render PDF (72*zoomin DPI). Mac dinh: 8 (=576 DPI). '
                             'Tu dong giam xuong voi trang qua to de tiet kiem RAM. '
                             'Tang de text nho rai rac (de thi quet) net hon khi OCR.')
    # Luu y: voi text nho rai rac, render can DU NHIEU DPI va detector can det_limit_side cao.
    # Rib trai lai: max_long_edge=2500 + det_limit_side=960 khien scan text nho dinh line.
    # Test: render ~445 DPI (max_long_edge=5200) + det_limit_side=2048 -> tach du cac dong nho
    # (Question, dap an A-D) thanh box rieng. Chon gia tri mac dinh nay de duoc luon ket qua tot.
    parser.add_argument('--limit', type=int, default=None,
                        help='Chi OCR N trang dau tien cua moi PDF (dung de xem thu voi file lon).')
    parser.add_argument('--max_long_edge', type=int, default=5200,
                        help='Canh dai toi da (pixel) khi render PDF; trang to hon se bi giam DPI '
                             'de tiet kiem RAM. Mac dinh: 5200 (~445 DPI trang A4). '
                             'Tang len (vd 5200-7000) giu text nho net de OCR khoi mat/dinh line '
                             '(dap an A-D). Giam xuong (vd 2500) neu muon chay nhanh va tiet kiem RAM.')

    parser.add_argument('--det_limit_side', type=int, default=2048,
                        help='Canh dai toi da (pixel) cua buoc phat hien text (detector). Mac dinh: 2048. '
                             'Cao (2048) giu duoc text nho rai rac (dap an A-D) tach rieng, '
                             'khoi dinh line. Giam xuong (vd 960) neu muon nhanh hon.')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.inputs)
    output_dir = os.path.abspath(args.output_dir)

    # ---- Kiem tra thu muc input ----
    if not os.path.isdir(input_dir):
        print(f'[X] Khong tim thay thu muc input: {input_dir}')
        print('    Hay tao thu muc input va copy file PDF vao do.')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    files = collect_inputs(input_dir)
    if not files:
        print(f'[!] Khong co file PDF/anh nao trong: {input_dir}')
        print('    Vui long copy file PDF vao thu muc input roi chay lai run.bat.')
        sys.exit(0)

    # ---- In thong tin ----
    print('=' * 56)
    print('  Pipeline PDF -> TXT  (DeepDoc + VietOCR)')
    print('=' * 56)
    print(f'  Tim thay {len(files)} file de OCR')
    print(f'  Input : {input_dir}')
    print(f'  Output: {output_dir}')
    print(f'  Thiet bi: CPU')
    print('=' * 56)
    print('[*] Dang nap mo hinh OCR (lan dau se cham mot chut) ...')

    # ---- Nap mo hinh OCR ----
    from module.ocr import OCR   # import sau khi da sua sys.path
    ocr = OCR(det_limit_side_len=args.det_limit_side)
    print('[*] Nap mo hinh xong!\n')

    # ---- Xu ly tung file ----
    ok, fail = 0, 0
    total_start = time.time()

    for idx, fpath in enumerate(files, 1):
        name = os.path.basename(fpath)
        print(f'[{idx}/{len(files)}] {name}')
        t0 = time.time()
        try:
            out_txt, n_pages = process_file(ocr, fpath, output_dir, args.zoomin, args.limit,
                                                max_long_edge=args.max_long_edge)
            elapsed = time.time() - t0
            avg = elapsed / n_pages if n_pages else elapsed
            print(f'    -> Da luu: {os.path.basename(out_txt)}  '
                  f'({n_pages} trang, {elapsed:.1f}s, {avg:.1f}s/trang)')
            ok += 1
        except KeyboardInterrupt:
            print('\n    [!] DUNG BOI Ctrl+C. Phan da OCR da duoc luu vao output.')
            print(f'    Tong ket: Thanh cong: {ok} | Loi: {fail}')
            sys.exit(0)
        except Exception as e:
            print(f'    [X] LOI: {e}')
            fail += 1

    # ---- Tong ket ----
    total_elapsed = time.time() - total_start
    print('\n' + '=' * 56)
    print(f'  HOAN THANH!  Thanh cong: {ok} | Loi: {fail}')
    print(f'  Tong thoi gian: {total_elapsed:.1f}s')
    print(f'  Ket qua luu tai: {output_dir}')
    print('=' * 56)


if __name__ == '__main__':
    main()
