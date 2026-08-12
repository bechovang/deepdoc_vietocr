<p align="center">
  <a href="./README.md">Tiếng Việt</a> |
  <a href="./README_en.md">English</a> |
</p>

# *Deep*Doc + VietOCR - Fast and Cost-effective OCR Tool for Vietnamese

- [1. Introduction](#1)
- [2. Architecture](#2)
- [3. Installation & Running](#3)

<a name="1"></a>

## 1. Introduction

With a wide range of documents from various sources and formats, along with diverse retrieval requirements, an accurate extraction tool is essential for any business. Today, I'd like to introduce DeepDoc, a very fast and cost-efficient OCR tool that only requires running on a CPU. In addition, it also comes with Layout Recognizer and Table Structure Recognizer features, which help preserve the document's formatting after OCR.

However, DeepDoc has not yet been standardized for Vietnamese, so I replaced the Text Recognizer with VietOCR and the ONNX version to achieve better Vietnamese text recognition. You can also check out the original version of DeepDoc [here](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md). Moreover, since DeepDoc is essentially a data processing component for the RAG pipeline in the RAGFlow project, I separated it into an independent Git repository so the application can be customized more conveniently.

<a name="2"></a>

## 2. Architecture
### 2.1 OCR
In this part, DeepDoc uses PaddleOCR - a very popular open-source tool developed by Baidu - after converting it into ONNX. Basically, ONNX (Open Neural Network Exchange) is an open format for AI models, allowing export and import of models between multiple frameworks (PyTorch, TensorFlow, etc.). It enables cross-platform compatibility, optimizes inference speed on CPU/GPU, and reduces infrastructure costs when deployed (we won't go too deep into this topic here).

DeepDoc does not specify which version it uses, since after conversion to ONNX it's difficult to determine. To get an idea of how it works, I'll refer to the OCR architecture PP-OCRv5 from the latest PaddleOCR 3.0, which includes four main components:

- Image Preprocessing Module: Enhances image quality, handles rotation/skew using orientation classification (PP-LCNet) and unwarping (UVDoc).

- Text Detection: Upgraded from PP-OCRv4 with backbone PP-HGNetV2, knowledge distillation from GOT-OCR2.0, and data augmentation (synthetic generation, rotation, blur, distortion). Retains PFHead and DSR from the previous version.

- Text Line Orientation Classification: Automatically detects and corrects text line orientation (flipped, rotated) to prepare for recognition.

- Text Recognition: Two-branch architecture with PP-HGNetV2, trained with GTC-NRTR (attention-based) to guide SVTR-HGNet (CTC, lightweight, fast). Training data is augmented with documents, PDFs, e-books, and synthetic handwriting samples.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\x6.png" width="900"/>
</div>

For more details about PP-OCRv5, you can refer to the official documentation [here](https://arxiv.org/html/2507.05595v1).

As mentioned above, the Recognition module of Paddle has been replaced with VietOCR and its ONNX version to achieve more accurate Vietnamese text recognition. VietOCR is already a very popular OCR tool in Vietnam, so I won't go into details here - you can explore more about it [here](https://github.com/pbcquoc/vietocr). For the process of converting VietOCR into the ONNX format, I referred to [this article](https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw).

### 2.2 Layout Recognizer & Table Structure Recognizer
In this part, DeepDoc uses YOLOv10 (You Only Look Once) - also a popular object detection method - in its ONNX version.

The basic architecture consists of three main components:
- Backbone: Extracts features from the image, using a lightweight and efficient design (retaining the ideas from YOLOv8 but improving the blocks to reduce computation).
- Neck: Combines multi-scale features (an improved FPN/PAN) to detect both small and large objects effectively.
- Head: Uses an anchor-free decoupled head (separating classification and regression branches), which improves accuracy and makes training easier.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\af645ed9-7301-4ec4-81e7-cb996ddf2d7f.webp" width="900"/>
</div>


In DeepDoc, YOLOv10 is trained to recognize label types for both Layout Recognizer and Table Structure Recognizer, covering most common cases.

For Layout Recognizer, there are 10 categories:
- Text
- Title
- Image
- Image Caption
- Table
- Table Caption
- Header
- Footer
- Reference
- Equation

For Table Structure Recognition, there are 5 types:
- Column
- Row
- Column header
- Projected row header
- Spanning cell

To understand more about YOLOv10, you can refer to the official documentation [here](https://arxiv.org/pdf/2405.14458).

<a name="3"></a>

## 3. Installation and Testing

First, clone the git repository:
```bash
git clone https://github.com/hoaivannguyen/deepdoc_vietocr.git
```
Some setup options before running the program:
```bash
python t_ocr.py -h
usage: t_ocr.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR]

options:
  -h, --help            Display this help message and exit
  --inputs INPUTS       Directory containing images or PDF files, or a file path to a single image or PDF file
  --output_dir OUTPUT_DIR
                        Directory to store output images. Default:'./ocr_outputs'
```
```bash
python t_recognizer.py -h
usage: t_recognizer.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR] [--threshold THRESHOLD] [--mode {layout,tsr}]

options:
  -h, --help            Display this help message and exit
  --inputs INPUTS       Directory containing images or PDF files, or a file path to a single image or PDF file
  --output_dir OUTPUT_DIR
                        Directory to store output images. Default: './layouts_outputs'
  --threshold THRESHOLD
                        Threshold for filtering detections. Default: 0.5
  --mode {layout,tsr}   Task mode: layout recognizer (layout) or table structure recognizer (tsr)
```
### 3.0. PDF → TXT Pipeline (recommended)

This is the simplest and fastest way to convert a batch of PDF files (or images) into Vietnamese text. Compared to `t_ocr.py` (section 3.1), this pipeline is more compact and user-friendly:

- **Outputs TXT only** — no extra debug images.
- **Merges PDF pages into a single `.txt`** (one PDF → one TXT); pages are separated by `===== Trang i/N =====`.
- **Prints per-page progress** to the console with an estimated time remaining (ETA).
- **Streams pages one by one** → no out-of-memory crashes on long/heavy PDFs.
- **Incremental writes**: if you stop mid-run (Ctrl+C), everything OCR'd so far is still saved.

#### Quick start (Windows)

1. Copy the PDF (or image) files you want to OCR into the **`input/`** folder (at the project root).
2. Double-click **`run.bat`** (or run it from Command Prompt).
3. Each file produces **one `.txt` file with the same name** in the **`output/`** folder.

```
deepdoc_vietocr/
├── input/          <- put PDF / images here
│   └── document.pdf
├── output/         <- TXT results appear here
│   └── document.txt
├── pdf_to_txt.py   <- pipeline script
└── run.bat         <- double-click to run (Windows)
```

#### Run from the command line

Minimal command (default settings):
```bash
python pdf_to_txt.py --inputs ./input --output_dir ./output
```

**Recommended** config for PDFs with small/dense text (exams, scanned documents…). `run.bat` already uses this — recovers more text and reduces line truncation:
```bash
python pdf_to_txt.py --inputs ./input --output_dir ./output \
    --zoomin 4 --max_long_edge 3400 --det_limit_side 1536
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--inputs` | `./input` | Directory containing input PDF/image files |
| `--output_dir` | `./output` | Directory to store output TXT files |
| `--zoomin` | `6` | PDF render resolution (72 × zoomin DPI). Automatically reduced for oversized pages to save RAM. |
| `--limit` | (no limit) | OCR only the first N pages of each PDF — useful for previewing large files |
| `--max_long_edge` | `2500` | Maximum long edge (pixels) when rendering a PDF; larger pages are downscaled to save RAM. Increase (e.g. `3400`, `5200`) for higher DPI. |
| `--det_limit_side` | `960` | Maximum long edge (pixels) at the text-detection step. Increase (e.g. `1536`) to reduce line truncation on small/dense text, at the cost of some speed. |

Example: preview the first 5 pages of a long PDF:
```bash
python pdf_to_txt.py --inputs ./input --limit 5
```

#### Supported formats

- **PDF:** `.pdf`
- **Images:** `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.webp`

#### Handling large / multi-page PDFs

The pipeline renders and OCRs **one page at a time** (only one page is kept in memory), so it can process PDFs with hundreds of pages without freezing. The render resolution is **automatically capped** to a sensible level via `--max_long_edge` (default 2500px). Note that rendering at very high resolution is usually pointless — see **Tuning quality** below. During the run, progress is printed like this:

```
[1/1] document.pdf
    - OCR trang 5/439 xong  (8s da qua, con lai ~250s)
```

Press **Ctrl+C** to stop early — everything OCR'd so far is saved to the TXT file immediately.

#### Notes

- Runs on **CPU** by default, recognizing with **VietOCR Seq2seq**. To switch to the Transformer/ONNX variant, edit `module/ocr.py` (see section 3.1).
- `run.bat` automatically uses `venv` if present; no need to activate the virtual environment manually.
- Contents of `input/` and `output/` are ignored by `.gitignore` by default (only the folders are tracked).

#### Tuning quality (DPI & detector)

OCR quality is governed by **two independent knobs** — understand them to avoid raising DPI for nothing:

1. **Render DPI** (`--zoomin`, `--max_long_edge`) — resolution used to rasterize the PDF page.
2. **Detector resolution** (`--det_limit_side`) — size (pixels) the text-detection step downscales to before finding text boxes.

**When is raising DPI pointless?**

- If the PDF content is an **embedded raster image** (e.g. FuOverflow exams: each question is a `1920×~720px` image, i.e. **271 DPI**), then rendering the page at 432/600/800 DPI **only upscales** that image — it adds no detail, just wastes RAM and time. The effective DPI is capped by the image's native resolution.
- If the content is **vector text**, the glyphs are already sharp at any DPI; ~288 DPI is more than enough. Errors here usually come from the detector cutting text boxes too short (`"MULTIPLE C"` instead of `"MULTIPLE CHOICE"`), not from blurriness.

**Recommended config** (verified): `--zoomin 4 --max_long_edge 3400 --det_limit_side 1536`.

Measured on one exam PDF (120 pages):

| Config | Words | `"MULTIPLE CHOICE"` complete | Speed |
|---|---|---|---|
| 432 DPI + detector 960 (old) | 2,723 | 19/60 pages | ~0.45 s/page |
| 288 DPI + detector 1536 (new) | **3,957 (+45%)** | **60/60 pages** | ~0.8 s/page |

→ Detector 1536 recovers ~50% more text and fixes all line truncation, at ~1.8× the time. If you need it faster, try `--det_limit_side 1280` (a good middle ground).

If, after raising the detector, **text in the 271-DPI image region is still misread** (e.g. `"lisled"` instead of `listed`), that's a limit of the source image — it needs a separate advanced path (extract the native image then upscale), not a higher page DPI.

### 3.1. OCR
To test OCR, you can use the following command:
 ```bash
python t_ocr.py --inputs=path_to_images_or_pdfs --output_dir=path_to_store_result
```
The input can be a directory containing images or PDFs, or a single image or PDF file. The output will include 1 image with the detected bounding boxes and 1 text file containing the OCR text.
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 171633.png" width="900"/>
</div>

I'm currently using VietOCR Seq2seq as the default since it runs relatively fast and accurately. You can switch to VietOCR Transformer in module/ocr.py, but I don't recommend it because the processing time is much longer while the accuracy doesn't improve significantly. If you want maximum speed, you can switch to the ONNX version by importing ocr_onnx instead of ocr, though the accuracy will decrease slightly.

### 3.2. Layout Recognizer
Try the following command to see the result of the Layout Recognizer:
```bash
python t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=layout --output_dir=path_to_store_result
```
The input can be a directory containing images or PDFs, or a single image or PDF file. The output will include 1 image with the detected labels as shown below:
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\49806-Article Text-153529-1-10-20200804_page-0002.jpg" width="1000"/>
</div>

### 3.3. Table Structure Recognizer
Try the following command to see the TSR result:
```bash
python t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=tsr --output_dir=path_to_store_result
```

The input can be a directory containing images or PDFs, or a single image or PDF file. The output will include 1 image with the detected labels and 1 markdown file with the table content.
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 182132.png" width="1000"/>
</div>

### 3.4. Full Pipeline — Layout + Table + Equation

A comprehensive pipeline that combines DeepDoc's full capabilities: layout recognition, table extraction (markdown), and mathematical equation recognition (LaTeX) in a single run.

```bash
python full_pipeline.py --inputs=path_to_images_or_pdfs --output_dir=./output --threshold=0.5
```

The input can be a directory containing images or PDFs. The output is a `.md` file containing:
- **OCR text** from non-table/non-equation regions
- **Tables** extracted as markdown (preserving column/row structure)
- **Math equations** recognized as LaTeX code (`$$...$$`)

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--inputs` | (required) | Directory containing input PDF/image files |
| `--output_dir` | `./table_markdown_outputs` | Directory to store output markdown files |
| `--threshold` | `0.5` | Detection confidence threshold |
| `--zoomin` | `6` | PDF render resolution (72 × zoomin DPI) |

#### Additional requirement (for equation recognition)

When processing documents with math equations, the pipeline requires **pix2tex**:

```bash
pip install pix2tex
```

If `pix2tex` is not installed, the pipeline still runs but equation regions will be skipped (outputting `_(equation)_`).

## 4. Build EXE (for distribution)

To distribute to clients who don't have Python installed, you can package the entire project into a standalone folder containing `.exe` files using **PyInstaller**.

### Requirements

- All dependencies installed in `venv`
- Internet connection (to pip install PyInstaller if missing)
- ~2 GB free disk space for the build process

### One-click build

```bash
python build_exe.py
```

The script will automatically:
1. Install PyInstaller (if not present)
2. Clean old builds
3. Package all code + model weights + dependencies
4. Output to `dist/DeepDoc_VietOCR/`

The build process takes approximately **10–30 minutes** depending on your machine.

### Result

After building, the `dist/DeepDoc_VietOCR/` folder contains:

```
dist/DeepDoc_VietOCR/
├── DeepDoc_VietOCR.exe    # GUI (no console, double-click)
├── pdf_to_txt.exe          # CLI (with console)
├── _internal/               # Python runtime + dependencies + models
│   ├── onnx/                # Model weights (~407 MB)
│   ├── torch/               # PyTorch (CUDA pruned)
│   ├── onnxruntime/         # ONNX Runtime
│   └── ...
├── input/                   # Create this folder, put PDFs here
└── output/                  # TXT results will appear here
```

**Total size:** ~1.5–1.8 GB (7-Zip compressed: ~700–900 MB)

### Client usage

#### GUI (recommended)

1. Extract the `DeepDoc_VietOCR` folder
2. Create an `input/` folder (if not present)
3. Copy PDF or image files into `input/`
4. Double-click **`DeepDoc_VietOCR.exe`**
5. Click **"📂 Thêm file..."** or **"📁 Thêm thư mục..."** to select files
6. Choose output directory (default `./output`)
7. Click **"▶ Bắt đầu OCR"**

#### CLI

```bash
pdf_to_txt.exe --inputs ./input --output_dir ./output
```

### Notes

- The `.exe` runs on machines **without Python installed**
- Compress the `DeepDoc_VietOCR` folder with 7-Zip / WinRAR before sending to clients
- For equation recognition support, install `pip install pix2tex` before building (see section 3.4)

## Conclusion
I hope you find this tool useful and applicable in practice. If you have any feedback, please leave it in the comments below. Thank you for reading!

## References
DeepDoc repo: https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md

PP-OCRv5: https://arxiv.org/html/2507.05595v1

VietOCR: https://github.com/pbcquoc/vietocr

VietOCR ONNX: https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw

YOLOv10: https://arxiv.org/pdf/2405.14458
