<p align="center">
  <a href="./README.md">Tiếng Việt</a> |
  <a href="./README_en.md">English</a> |
</p>

# *Deep*Doc + VietOCR - Công cụ OCR cho tiếng Việt nhanh và tiết kiệm chi phí

- [1. Giới thiệu](#1)
- [2. Kiến trúc kỹ thuật](#2)
- [3. Cài đặt và chạy thử](#3)

<a name="1"></a>

## 1. Giới thiệu

Với một loạt tài liệu từ nhiều nguồn khác nhau với nhiều định dạng khác nhau và cùng với các yêu cầu truy xuất đa dạng,  một công cụ trích xuất chính xác là rất cần thiết với bất kỳ doanh nghiệp nào. Hôm nay mình sẽ giới thiệu với các bạn công cụ DeepDoc, một công cụ OCR rất nhanh và tiết kiệm chi phí khi chỉ cần chạy trên CPU. Không những vậy còn có các tính năng kèm theo là Layout Recognizer (nhận diện bố cục) và Table Structure Recognizer (nhận diện cấu trúc bảng) giúp giữ định dạng văn bản sau OCR. 

Tuy nhiên DeepDoc chưa được chuẩn hóa cho tiếng Việt nên mình đã thay VietOCR và bản ONNX vào phần Text Recognizer để có thể nhận dạng văn bảng tiếng Việt tốt hơn. Bạn cũng có thể tham khảo DeepDoc phiên bản gốc tại [đây](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md). Thêm vào đó, DeepDoc bản chất là 1 phần xử lý dữ liệu cho luồng RAG thuộc dự án RAGFlow nên việc mình tách ra thành 1 git riêng cũng để ứng dụng có thể tùy chỉnh một cách thuận tiện hơn

<a name="2"></a>

## 2. Kiến trúc kỹ thuật
### 2.1 OCR
Phần này DeepDoc sử dụng PaddleOCR - Công cụ mã nguồn mở rất thông dụng được phát triển bởi Baidu - sau khi chuyển sang ONNX. Cơ bản thì ONNX (Open Neural Network Exchange) dạng mở cho mô hình AI, cho phép xuất – nhập mô hình giữa nhiều framework (PyTorch, TensorFlow, v.v.), giúp mô hình tương thích đa nền tảng, tối ưu tốc độ trên CPU/GPU và giảm chi phí hạ tầng khi triển khai (chúng ta sẽ không đi quá sâu về chủ đề này)

Bên DeepDoc họ không ghi rõ là sử dụng version bao nhiêu vì sau khi chuyển sang ONNX cũng khó để xác định lại. Để hiểu qua về cách hoạt động, mình sẽ tham khảo kiến trúc OCR PP-OCRv5 của bản mới nhất là PaddleOCR 3.0, bao gồm 4 phần chính:

- Image Preprocessing Module (Tiền xử lý ảnh): Cải thiện chất lượng ảnh, xử lý xoay/nghiêng bằng mô hình phân loại hướng (PP-LCNet) và unwarping (UVDoc).

- Text Detection (Phát hiện văn bản): Nâng cấp từ PP-OCRv4 nhờ backbone PP-HGNetV2, distillation từ - GOT-OCR2.0, và tăng cường dữ liệu (sinh tổng hợp, xoay, mờ, biến dạng). Giữ lại PFHead và DSR từ phiên bản trước.

- Text Line Orientation Classification (Phân loại hướng dòng chữ): Tự động phát hiện, sửa hướng dòng chữ (ngược, xoay) để chuẩn bị cho bước nhận dạng.

- Text Recognition (Nhận dạng văn bản): Kiến trúc 2 nhánh với PP-HGNetV2, huấn luyện bằng GTC-NRTR (attention) để hướng dẫn SVTR-HGNet (CTC, nhẹ, nhanh). Dữ liệu huấn luyện được tăng cường từ tài liệu, PDF, e-book, và sinh mẫu chữ viết tay.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\x6.png" width="900"/>
</div>

Chi tiết về PP-OCRv5, bạn có thể tham khảo tài liệu chính thức tại [đây](https://arxiv.org/html/2507.05595v1).

Như đã nói bên trên, phần Recognition của Paddle đã được thay bằng VietOCR và bản ONNX để việc nhận dạng chữ tiếng Việt chính xác hơn. Về VietOCR thì đó là 1 công cụ quá phổ biến cho OCR ở Việt Nam rồi, nên mình sẽ không đi sâu, các bạn có thể tìm hiểu thêm ở [đây](https://github.com/pbcquoc/vietocr). Còn phần chuyển sang định dạng ONNX cho VietOCR thì mình tham khảo từ bài viết [này.](https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw)

### 2.2 Layout Recognizer và Table Structure Recognizer
Phần này thì DeepDoc sử dụng YOLOv10 (You Only Look Once) - cũng là 1 phương pháp object detection (phát hiện đối tượng) phổ biến - phiên bản ONNX.

Kiến trúc cơ bản gồm 3 phần chính:

- Backbone: trích xuất đặc trưng từ ảnh, dùng thiết kế nhẹ và hiệu quả (giữ lại ý tưởng từ YOLOv8 nhưng cải tiến block để giảm tính toán).

- Neck: kết hợp đa cấp độ đặc trưng (FPN/PAN cải tiến) để phát hiện tốt cả vật thể nhỏ lẫn lớn.

- Head: sử dụng Anchor-Free decoupled head (tách nhánh classification và regression), tăng độ chính xác và dễ huấn luyện.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\af645ed9-7301-4ec4-81e7-cb996ddf2d7f.webp" width="900"/>
</div>


Trong DeepDoc, YOLOv10 được huấn luyện để nhận dạng các loại nhãn cho phần Layout Recognizer và Table Structure Recognizer cơ bản bao phủ hầu hết các trường hợp.

Đối với Layout Recognizer, có 10 loại:
- Text (Văn bản)
- Title (Tiêu đề)
- Image (Hình ảnh)
- Image Caption (Chú thích hình ảnh)
- Table (Bảng)
- Table Caption (Chú thích bảng)
- Header (Đầu đề)
- Footer (Chân trang)
- Reference (Tài liệu tham khảo)
- Equation (Phương trình)

Đối với Table Structure Recognizer, có 5 loại:
- Column (Cột)
- Row (Hàng)
- Column header (Đầu đề cột)
- Projected row header (Đầu đề hàng được chiếu)
- Spanning cell (Ô trải dài)


Để có thể hiểu rõ hơn về YOLOv10, bạn có thể tham khảo tài liệu chính thức tại [đây](https://arxiv.org/pdf/2405.14458).

<a name="3"></a>

## 3. Cài đặt và chạy thử

Đầu tiên bạn clone git về máy:
```bash
git clone https://github.com/hoaivannguyen/deepdoc_vietocr.git
```
Một số cài đặt trước khi chạy chương trình:
```bash
python t_ocr.py -h
usage: t_ocr.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR]

options:
  -h, --help            hiển thị thông báo trợ giúp này và thoát
  --inputs INPUTS       Thư mục lưu trữ hình ảnh hoặc tệp PDF hoặc đường dẫn tệp đến một hình ảnh hoặc tệp PDF duy nhất
  --output_dir OUTPUT_DIR
                        Thư mục lưu trữ hình ảnh đầu ra. Mặc định: './ocr_outputs'
```
```bash
python t_recognizer.py -h
usage: t_recognizer.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR] [--threshold THRESHOLD] [--mode {layout,tsr}]

options:
  -h, --help            hiển thị thông báo trợ giúp này và thoát
  --inputs INPUTS       Thư mục lưu trữ hình ảnh hoặc tệp PDF hoặc đường dẫn tệp đến một hình ảnh hoặc tệp PDF duy nhất
  --output_dir OUTPUT_DIR
                        Thư mục lưu trữ hình ảnh đầu ra. Mặc định: './layouts_outputs'
  --threshold THRESHOLD
                        Ngưỡng để lọc ra các phát hiện. Mặc định: 0.5
  --mode {layout,tsr}   Chế độ tác vụ: nhận dạng bố cục (layout) hoặc nhận dạng cấu trúc bảng (tsr)
```
### 3.0. Pipeline PDF → TXT (khuyên dùng)

Đây là cách đơn giản và nhanh nhất để chuyển một loạt file PDF (hoặc ảnh) thành file text tiếng Việt. So với `t_ocr.py` (mục 3.1), pipeline này được thiết kế gọn và thân thiện hơn:

- **Chỉ xuất TXT** — không sinh ảnh debug thừa.
- **Gộp các trang PDF thành 1 file `.txt`** (mỗi PDF → 1 TXT), các trang ngăn cách bằng `===== Trang i/N =====`.
- **In tiến độ ra màn hình** từng trang kèm thời gian ước tính còn lại (ETA).
- **Xử lý streaming từng trang** → không bị treo do hết RAM với PDF dày/nặng.
- **Ghi tăng dần**: nếu dừng giữa chừng (Ctrl+C), phần đã OCR vẫn được lưu vào file.

#### Cách dùng nhanh (Windows)

1. Copy các file PDF (hoặc ảnh) cần OCR vào thư mục **`input/`** (ở thư mục gốc dự án).
2. Double-click **`run.bat`** (hoặc chạy trong Command Prompt).
3. Mỗi file ra **1 file `.txt` cùng tên** trong thư mục **`output/`**.

```
deepdoc_vietocr/
├── input/          ← bỏ PDF / ảnh vào đây
│   └── tai-lieu.pdf
├── output/         ← kết quả TXT ra đây
│   └── tai-lieu.txt
├── pdf_to_txt.py   ← script pipeline
└── run.bat         ← double-click để chạy (Windows)
```

#### Chạy bằng dòng lệnh

Lệnh tối giản (dùng cấu hình mặc định):
```bash
python pdf_to_txt.py --inputs ./input --output_dir ./output
```

Cấu hình **khuyên dùng** cho PDF có text nhỏ/dày (đề thi, giấy tờ scan…). `run.bat` đã áp dụng sẵn cấu hình này — cho ra nhiều chữ hơn và giảm lỗi cắt dòng:
```bash
python pdf_to_txt.py --inputs ./input --output_dir ./output \
    --zoomin 4 --max_long_edge 3400 --det_limit_side 1536
```

#### Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--inputs` | `./input` | Thư mục chứa file PDF/ảnh đầu vào |
| `--output_dir` | `./output` | Thư mục lưu file TXT kết quả |
| `--zoomin` | `6` | Độ phân giải render PDF (72 × zoomin DPI). Tự động giảm với trang quá to để tiết kiệm RAM. |
| `--limit` | (không giới hạn) | Chỉ OCR N trang đầu của mỗi PDF — hữu ích để xem thử file dày |
| `--max_long_edge` | `2500` | Cạnh dài tối đa (pixel) khi render PDF; trang to hơn sẽ bị giảm DPI để tiết kiệm RAM. Tăng lên (vd `3400`, `5200`) nếu muốn DPI cao hơn. |
| `--det_limit_side` | `960` | Cạnh dài tối đa (pixel) ở bước phát hiện text (detector). Tăng lên (vd `1536`) để giảm lỗi cắt dòng ở text nhỏ/dày; đổi lại chậm hơn một chút. |

Ví dụ xem thử 5 trang đầu của một PDF dày:
```bash
python pdf_to_txt.py --inputs ./input --limit 5
```

#### Định dạng hỗ trợ

- **PDF:** `.pdf`
- **Ảnh:** `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.webp`

#### Xử lý PDF nhiều trang / dung lượng lớn

Pipeline render và OCR **từng trang một** (chỉ giữ một trang trong RAM tại một thời điểm), nên có thể xử lý PDF hàng trăm trang mà không bị treo. Độ phân giải render được **tự động giới hạn** ở mức hợp lý qua `--max_long_edge` (mặc định 2500px). Lưu ý: render siêu cao thường vô ích — xem phân tích chi tiết ở mục **Hiệu chỉnh chất lượng** bên dưới. Trong quá trình chạy, màn hình sẽ in tiến độ dạng:

```
[1/1] tai-lieu.pdf
    - OCR trang 5/439 xong  (8s da qua, con lai ~250s)
```

Cần dừng sớm thì nhấn **Ctrl+C** — phần đã OCR sẽ được lưu ngay vào file TXT.

#### Ghi chú

- Mặc định chạy trên **CPU**, nhận dạng bằng **VietOCR Seq2seq**. Muốn đổi sang Transformer/ONNX thì chỉnh trong `module/ocr.py` (xem mục 3.1).
- `run.bat` tự dùng `venv` (nếu có); không cần kích hoạt môi trường ảo bằng tay.
- Dữ liệu trong `input/` và `output/` mặc định bị `.gitignore` bỏ qua (chỉ track thư mục).

#### Hiệu chỉnh chất lượng (DPI & detector)

Có **hai biến độc lập** quyết định chất lượng OCR — cần hiểu rõ để không tăng DPI vô ích:

1. **DPI render** (`--zoomin`, `--max_long_edge`) — độ phân giải khi raster hóa trang PDF.
2. **Độ phân giải detector** (`--det_limit_side`) — kích thước (pixel) mà bước phát hiện text downscale về trước khi tìm hộp chữ.

**Khi nào tăng DPI là vô ích?**

- Nếu nội dung PDF là **ảnh raster nhúng** (ví dụ đề thi FuOverflow: mỗi câu hỏi là một ảnh `1920×~720px`, tương đương **271 DPI**), thì render trang ở 432/600/800 DPI **chỉ là upscale** cái ảnh đó — không thêm chi tiết, chỉ tốn RAM và chậm hơn. DPI hiệu dụng bị chốt ở độ phân giải gốc của ảnh.
- Nếu nội dung là **text vector**, chữ vốn đã sắc ở mọi DPI; ~288 DPI là dư sức. Lỗi lúc này thường do detector cắt hộp chữ quá ngắn (`"MULTIPLE C"` thay vì `"MULTIPLE CHOICE"`), không phải do mờ.

**Cấu hình khuyên dùng** (đã kiểm chứng): `--zoomin 4 --max_long_edge 3400 --det_limit_side 1536`.

Kết quả đo trên một PDF đề thi (120 trang):

| Cấu hình | Số từ | `"MULTIPLE CHOICE"` đầy đủ | Tốc độ |
|---|---|---|---|
| 432 DPI + detector 960 (cũ) | 2.723 | 19/60 trang | ~0.45 s/trang |
| 288 DPI + detector 1536 (mới) | **3.957 (+45%)** | **60/60 trang** | ~0.8 s/trang |

→ Detector 1536 lấy thêm gần 50% chữ và sửa hết lỗi cắt dòng, đổi lại chậm ~1.8×. Nếu cần nhanh hơn, thử `--det_limit_side 1280` (điểm cân bằng).

Nếu sau khi tăng detector mà **vẫn sai chữ ở vùng ảnh 271-DPI** (vd `"lisled"` thay vì `listed`), đó là giới hạn của ảnh gốc — cần nhánh nâng cao riêng (rút ảnh native rồi upscale) chứ không phải tăng DPI trang.

### 3.1. OCR
Để chạy thử OCR, bạn có thể sử dụng lệnh sau:
 ```bash
python t_ocr.py --inputs=path_to_images_or_pdfs --output_dir=path_to_store_result
```
Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF, hoặc một hình ảnh hoặc PDF. Đầu ra sẽ gồm 1 ảnh với các bounding box được nhận diện và 1 file txt chứa văn bản được OCR.
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 171633.png" width="900"/>
</div>

Mình đang để mặc định là VietOCR Seq2seq vì hiện đang chạy tương đối nhanh và chính xác. Bạn có thể đổi sang VietOCR Transformer trong module/ocr.py nhưng mình không đề xuất vì thời gian xử lý lâu hơn rất nhiều mà độ chuẩn xác không tănng lên là mấy. Nếu bạn muốn nhanh nhất có thể chuyển sang sử dụng bản ONNX bằng việc import ocr_onnx thay vì ocr nhưng độ chính xác sẽ giảm đi 1 chút.

### 3.2. Layout Recognizer (Nhận diện bố cục)
Hãy thử lệnh sau để xem kết quả Layout Recognizer:
```bash
python t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=layout --output_dir=path_to_store_result
```
Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF, hoặc một hình ảnh hoặc PDF. Đầu ra sẽ gồm 1 ảnh với các gán nhãn như dưới đây:
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\49806-Article Text-153529-1-10-20200804_page-0002.jpg" width="1000"/>
</div>

## 3.3 Table Structure Recognizer
Hãy thử lệnh sau để xem kết quả TSR.
```bash
python t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=tsr --output_dir=path_to_store_result
```

Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF, hoặc một hình ảnh hoặc PDF. Đầu ra sẽ là 1 ảnh với gán nhãn và 1 file markdown với nội dung bảng
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 182132.png" width="1000"/>
</div>

### 3.4. Full Pipeline — Layout + Table + Equation

Pipeline tổng hợp kết hợp toàn bộ khả năng của DeepDoc: nhận diện bố cục (layout), trích xuất bảng (table markdown) và nhận dạng công thức toán học (LaTeX) chỉ trong một lần chạy.

```bash
python full_pipeline.py --inputs=path_to_images_or_pdfs --output_dir=./output --threshold=0.5
```

Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF. Đầu ra là file `.md` với nội dung gồm:
- **Văn bản OCR** từ các vùng không phải bảng/công thức
- **Bảng** được trích xuất dưới dạng markdown (giữ nguyên cấu trúc cột/hàng)
- **Công thức toán** được nhận dạng thành mã LaTeX (`$$...$$`)

#### Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--inputs` | (bắt buộc) | Thư mục chứa file PDF/ảnh đầu vào |
| `--output_dir` | `./table_markdown_outputs` | Thư mục lưu file markdown kết quả |
| `--threshold` | `0.5` | Ngưỡng lọc phát hiện layout |
| `--zoomin` | `6` | Độ phân giải render PDF (72 × zoomin DPI) |

#### Yêu cầu bổ sung (cho nhận dạng công thức)

Khi xử lý tài liệu có công thức toán, pipeline cần thêm gói **pix2tex**:

```bash
pip install pix2tex
```

Nếu chưa cài `pix2tex`, pipeline vẫn chạy được nhưng vùng công thức sẽ bị bỏ qua (ghi là `_(equation)_`).

## 4. Build EXE (cho khách hàng)

Để phân phối cho khách không cần cài Python, bạn có thể đóng gói toàn bộ dự án thành thư mục chứa file `.exe` chạy độc lập bằng **PyInstaller**.

### Yêu cầu

- Đã cài đặt đầy đủ dependencies trong `venv`
- Internet (để pip install PyInstaller nếu chưa có)
- ~2 GB dung lượng ổ đĩa trống cho quá trình build

### Build 1-click

```bash
python build_exe.py
```

Script sẽ tự động:
1. Cài PyInstaller (nếu chưa có)
2. Dọn sạch bản build cũ
3. Đóng gói toàn bộ code + model weights + dependencies
4. Xuất ra thư mục `dist/DeepDoc_VietOCR/`

Quá trình build mất khoảng **10–30 phút** tùy cấu hình máy.

### Kết quả

Sau khi build, thư mục `dist/DeepDoc_VietOCR/` gồm:

```
dist/DeepDoc_VietOCR/
├── DeepDoc_VietOCR.exe    # GUI (không console, double-click)
├── pdf_to_txt.exe          # CLI (có console, chạy bằng cmd)
├── _internal/               # Python runtime + dependencies + models
│   ├── onnx/                # Model weights (~407 MB)
│   ├── torch/               # PyTorch runtime (da prune CUDA)
│   ├── onnxruntime/         # ONNX Runtime
│   └── ...
├── input/                   # Tao thu muc nay, bo PDF vao
└── output/                  # Ket qua TXT tu dong vao day
```

**Tổng dung lượng:** ~1.5–1.8 GB (nén 7-Zip còn ~700–900 MB)

### Cách dùng cho khách

#### GUI (khuyên dùng)

1. Giải nén thư mục `DeepDoc_VietOCR`
2. Tạo thư mục `input/` (nếu chưa có)
3. Copy file PDF hoặc ảnh vào `input/`
4. Double-click **`DeepDoc_VietOCR.exe`**
5. Trong cửa sổ GUI: bấm **"📂 Thêm file..."** hoặc **"📁 Thêm thư mục..."**
6. Chọn thư mục output (mặc định `./output`)
7. Bấm **"▶ Bắt đầu OCR"**

#### CLI

```bash
pdf_to_txt.exe --inputs ./input --output_dir ./output
```

### Ghi chú

- File `.exe` đã chạy được ngay trên máy **không cài Python**
- Có thể nén thư mục `DeepDoc_VietOCR` bằng 7-Zip / WinRAR để gửi cho khách
- Nếu khách cần xử lý công thức toán, cài thêm `pip install pix2tex` trước khi build (xem mục 3.4)

## Kết
Hy vọng các bạn thấy công cụ hữu ích và áp dụng được vào thực tế. Nếu có góp ý hãy để lại dưới phần bình luận. Cảm ơn các bạn đã đọc bài viết! 


## Tài liệu tham khảo
DeepDoc repo: https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md

PP-OCRv5: https://arxiv.org/html/2507.05595v1

VietOCR: https://github.com/pbcquoc/vietocr

VietOCR ONNX: https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw

YOLOv10: https://arxiv.org/pdf/2405.14458
