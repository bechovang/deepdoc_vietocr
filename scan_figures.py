#
#  Quet PDF/anh -> bao cao trang nao co figure (diagram/bieu do/anh minh hoa)
#  Chi chay mo hinh layout (onnx/layout.onnx), KHONG OCR -> nhanh hon pipeline nhieu.
#
#  Cach dung:
#     python scan_figures.py                            # quet ./input
#     python scan_figures.py --inputs ./input --limit 20
#     python scan_figures.py --inputs ./input --report bao_cao.txt
#

import argparse
import os
import sys
import time

from PIL import Image

# -------------------------------------------------------------------
# Cau hinh moi truong (giong pdf_to_txt.py)
# -------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pdf_to_txt import collect_inputs, page_resolution, PDF_EXT, _PDF_LOCK
from figure_mvp.figure_export import get_layout_recognizer, detect_figures, detect_figures_page


def scan_pdf(path, zoomin, max_long_edge, limit=None):
    """Quet tung trang PDF -> tra ve (so_trang_quet, {trang: so_figure})."""
    import pdfplumber

    found = {}
    with _PDF_LOCK:
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            target = total if not limit else min(total, limit)
            start = time.time()
            for pi, page in enumerate(pdf.pages, 1):
                if pi > target:
                    break
                resolution = page_resolution(page, zoomin, max_long_edge=max_long_edge)
                img = page.to_image(resolution=resolution).annotated
                # Detect ca tren anh nhung (exhibit nho trong screenshot de thi)
                regions = detect_figures_page(page, img)
                del img
                if regions:
                    found[pi] = len(regions)
                elapsed = time.time() - start
                eta = elapsed / pi * (target - pi) if pi else 0
                print(f'    - Trang {pi}/{target}'
                      f'  ({elapsed:.0f}s da qua, con lai ~{eta:.0f}s)')
    return target, found


def scan_image(path):
    """Quet 1 file anh -> tra ve (1, {1: so_figure})."""
    img = Image.open(path).convert('RGB')
    regions = detect_figures(img)
    return 1, ({1: len(regions)} if regions else {})


def main():
    parser = argparse.ArgumentParser(
        description='Quet PDF/anh, bao cao trang nao co figure (diagram/bieu do). '
                    'Chi chay mo hinh layout, khong OCR.')
    parser.add_argument('--inputs', default='./input',
                        help='Thu muc chua file PDF/anh dau vao. Mac dinh: ./input')
    parser.add_argument('--zoomin', type=int, default=5,
                        help='Do phan giai render khi quet (72*zoomin DPI). Mac dinh: 5 (=360 DPI). '
                             'Quet figure khong can DPI cao nhu OCR.')
    parser.add_argument('--max_long_edge', type=int, default=3000,
                        help='Canh dai toi da (pixel) khi render trang. Mac dinh: 3000.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Chi quet N trang dau tien cua moi PDF.')
    parser.add_argument('--report', default=None,
                        help='Duong dan file TXT de luu bao cao (ngoai in ra man hinh).')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.inputs)
    if not os.path.isdir(input_dir):
        print(f'[X] Khong tim thay thu muc input: {input_dir}')
        sys.exit(1)

    files = collect_inputs(input_dir)
    if not files:
        print(f'[!] Khong co file PDF/anh nao trong: {input_dir}')
        sys.exit(0)

    print('=' * 56)
    print('  Quet FIGURE (diagram/bieu do) - chi layout, khong OCR')
    print('=' * 56)
    print(f'  Tim thay {len(files)} file de quet')
    print(f'  Input : {input_dir}')
    print('[*] Dang nap mo hinh layout ...')
    get_layout_recognizer()  # nap 1 lan truoc khi quet
    print('[*] Nap mo hinh xong!\n')

    lines = []          # dong bao cao de luu file (neu co)
    total_figs = 0

    for idx, fpath in enumerate(files, 1):
        name = os.path.basename(fpath)
        print(f'[{idx}/{len(files)}] {name}')
        lines.append(f'{name}:')

        ext = os.path.splitext(name)[1].lower()
        if ext == PDF_EXT:
            n_pages, found = scan_pdf(fpath, args.zoomin, args.max_long_edge, args.limit)
            unit = 'trang'
        else:
            n_pages, found = scan_image(fpath)
            unit = 'anh'

        figs = sum(found.values())
        total_figs += figs
        if found:
            pages_str = ', '.join(f'{p} ({k} hinh)' if k > 1 else str(p)
                                  for p, k in sorted(found.items()))
            summary = f'    -> CO figure: {len(found)}/{n_pages} {unit}, {figs} hinh. Trang: {pages_str}'
        else:
            summary = f'    -> Khong co figure nao ({n_pages} {unit})'
        print(summary + '\n')
        lines.append(summary)
        lines.append('')

    print('=' * 56)
    print(f'  XONG!  Tong so figure phat hien: {total_figs}')
    print('=' * 56)

    if args.report:
        report_path = os.path.abspath(args.report)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'  Bao cao da luu tai: {report_path}')


if __name__ == '__main__':
    main()
