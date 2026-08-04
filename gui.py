"""
GUI Tkinter cho pipeline OCR (DeepDoc + VietOCR).
Cho phep khach chon file PDF/anh, chay OCR, theo doi tien trinh.
"""

import os
import sys
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

# Dam bao import duoc module/ va utils/
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU

SUPPORTED_IMG = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp')
PDF_EXT = '.pdf'

# -------------------------------------------------------------------
# Worker: chay OCR tren thread rieng, day message vao queue
# -------------------------------------------------------------------
class OcrWorker:
    def __init__(self, file_paths, output_dir, zoomin, msg_queue):
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.zoomin = zoomin
        self.msg_queue = msg_queue
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.msg_queue.put(("status", "Đang nạp mô hình OCR..."))
            self.msg_queue.put(("log", "[*] Đang nạp mô hình OCR (lần đầu sẽ chậm)...\n"))

            # Import o day de tranh cham khi GUI khoi dong
            from pdf_to_txt import process_file, image_to_text
            from module.ocr import OCR
            from PIL import Image

            ocr = OCR()
            self.msg_queue.put(("status", "Đã nạp mô hình xong!"))
            self.msg_queue.put(("log", "[*] Nạp mô hình xong!\n\n"))

            if self._stop:
                self.msg_queue.put(("done", (0, 0)))
                return

            total = len(self.file_paths)
            ok, fail = 0, 0
            total_start = time.time()

            os.makedirs(self.output_dir, exist_ok=True)

            for idx, fpath in enumerate(self.file_paths, 1):
                if self._stop:
                    self.msg_queue.put(("log", "\n[!] Đã dừng bởi người dùng.\n"))
                    break

                name = os.path.basename(fpath)
                self.msg_queue.put(("log", f"[{idx}/{total}] {name}\n"))
                self.msg_queue.put(("status", f"Đang xử lý: {name}"))
                self.msg_queue.put(("progress", (idx - 1) / total * 100))

                t0 = time.time()
                try:
                    out_txt, n_pages = process_file(ocr, fpath, self.output_dir, self.zoomin)
                    elapsed = time.time() - t0
                    avg = elapsed / n_pages if n_pages else elapsed
                    self.msg_queue.put(("log", f"    -> {os.path.basename(out_txt)}  "
                                               f"({n_pages} trang, {elapsed:.1f}s, {avg:.1f}s/trang)\n"))
                    self.msg_queue.put(("progress", idx / total * 100))
                    ok += 1
                except Exception as e:
                    self.msg_queue.put(("log", f"    [X] LỖI: {e}\n"))
                    fail += 1

            total_elapsed = time.time() - total_start
            self.msg_queue.put(("log", "\n" + "=" * 56 + "\n"))
            self.msg_queue.put(("log", f"  HOÀN THÀNH!  Thành công: {ok} | Lỗi: {fail}\n"))
            self.msg_queue.put(("log", f"  Tổng thời gian: {total_elapsed:.1f}s\n"))
            self.msg_queue.put(("log", f"  Kết quả lưu tại: {self.output_dir}\n"))
            self.msg_queue.put(("log", "=" * 56 + "\n\n"))
            self.msg_queue.put(("status", "Hoàn thành!"))
            self.msg_queue.put(("done", (ok, fail)))

        except Exception as e:
            self.msg_queue.put(("log", f"\n[X] LỖI NGHIÊM TRỌNG: {e}\n"))
            self.msg_queue.put(("status", "Lỗi!"))
            self.msg_queue.put(("done", (0, 0)))


# -------------------------------------------------------------------
# GUI chinh
# -------------------------------------------------------------------
class OcrGui:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepDoc + VietOCR - OCR Tool")
        self.root.geometry("780x620")
        self.root.minsize(640, 480)

        # Icon (neu co)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        self.file_paths = []  # danh sach duong dan file
        self.output_dir = os.path.join(ROOT_DIR, "output")
        self.zoomin = tk.IntVar(value=5)
        self.worker = None
        self.worker_thread = None
        self.msg_queue = queue.Queue()
        self.running = False

        self._build_ui()
        self._poll_queue()

    # ---- Xay dung giao dien ----
    def _build_ui(self):
        # ---------- Khung chon file ----------
        frame_files = ttk.LabelFrame(self.root, text=" 1. Chọn file PDF / Ảnh ", padding=8)
        frame_files.pack(fill="x", padx=10, pady=(10, 5))

        btn_row = ttk.Frame(frame_files)
        btn_row.pack(fill="x", pady=(0, 5))

        ttk.Button(btn_row, text="📂 Thêm file...", command=self._add_files).pack(side="left", padx=(0, 5))
        ttk.Button(btn_row, text="📁 Thêm thư mục...", command=self._add_folder).pack(side="left", padx=(0, 5))
        ttk.Button(btn_row, text="🗑️ Xóa hết", command=self._clear_files).pack(side="left", padx=(0, 5))

        ttk.Label(btn_row, text="Zoomin:").pack(side="right", padx=(10, 3))
        ttk.Spinbox(btn_row, from_=1, to=12, textvariable=self.zoomin, width=4).pack(side="right")

        self.lbl_count = ttk.Label(frame_files, text="0 file", foreground="#666")
        self.lbl_count.pack(anchor="w")

        # Listbox file
        list_frame = ttk.Frame(frame_files)
        list_frame.pack(fill="x")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.file_listbox = tk.Listbox(
            list_frame, height=5, selectmode="extended",
            yscrollcommand=scrollbar.set, font=("Consolas", 9)
        )
        scrollbar.config(command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.pack(side="left", fill="x", expand=True)

        # ---------- Khung thu muc output ----------
        frame_out = ttk.LabelFrame(self.root, text=" 2. Thư mục kết quả ", padding=8)
        frame_out.pack(fill="x", padx=10, pady=5)

        out_row = ttk.Frame(frame_out)
        out_row.pack(fill="x")

        self.out_path_var = tk.StringVar(value=self.output_dir)
        ttk.Entry(out_row, textvariable=self.out_path_var).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(out_row, text="📁 Chọn...", command=self._choose_output_dir).pack(side="right")

        # ---------- Khung dieu khien ----------
        frame_ctrl = ttk.Frame(self.root)
        frame_ctrl.pack(fill="x", padx=10, pady=5)

        self.btn_start = ttk.Button(frame_ctrl, text="▶ Bắt đầu OCR", command=self._start_ocr)
        self.btn_start.pack(side="left", padx=(0, 5))

        self.btn_stop = ttk.Button(frame_ctrl, text="⏹ Dừng", command=self._stop_ocr, state="disabled")
        self.btn_stop.pack(side="left", padx=(0, 5))

        self.btn_open = ttk.Button(frame_ctrl, text="📂 Mở thư mục kết quả", command=self._open_output, state="disabled")
        self.btn_open.pack(side="right")

        # ---------- Progress ----------
        frame_progress = ttk.Frame(self.root)
        frame_progress.pack(fill="x", padx=10, pady=(0, 5))

        self.progress = ttk.Progressbar(frame_progress, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 3))

        self.lbl_status = ttk.Label(frame_progress, text="Sẵn sàng", foreground="#555")
        self.lbl_status.pack(anchor="w")

        # ---------- Log ----------
        frame_log = ttk.LabelFrame(self.root, text=" Log ", padding=4)
        frame_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_area = scrolledtext.ScrolledText(
            frame_log, wrap="word", state="disabled",
            font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white"
        )
        self.log_area.pack(fill="both", expand=True)

        # Tag colors
        self.log_area.tag_config("info", foreground="#4ec9b0")
        self.log_area.tag_config("warn", foreground="#ce9178")
        self.log_area.tag_config("error", foreground="#f44747")
        self.log_area.tag_config("done", foreground="#6a9955", font=("Consolas", 9, "bold"))

        self._log("[i] Sẵn sàng. Chọn file PDF hoặc ảnh để bắt đầu.\n", "info")
        self._log("[i] Mặc định chạy trên CPU (DeepDoc + VietOCR).\n", "info")

    # ---- Cac ham xu ly ----
    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Chọn file PDF / Ảnh",
            filetypes=[("PDF & Images", "*.pdf *.jpg *.jpeg *.png *.tif *.tiff *.bmp *.gif *.webp"),
                       ("PDF", "*.pdf"),
                       ("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.gif *.webp")]
        )
        for f in files:
            if f not in self.file_paths:
                self.file_paths.append(f)
                self.file_listbox.insert("end", os.path.basename(f))
        self._update_count()

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa file PDF / Ảnh")
        if not folder:
            return
        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext == PDF_EXT or ext in SUPPORTED_IMG:
                full = os.path.join(folder, name)
                if full not in self.file_paths:
                    self.file_paths.append(full)
                    self.file_listbox.insert("end", name)
        self._update_count()

    def _clear_files(self):
        self.file_paths.clear()
        self.file_listbox.delete(0, "end")
        self._update_count()

    def _update_count(self):
        n = len(self.file_paths)
        self.lbl_count.config(text=f"{n} file" if n <= 1 else f"{n} files")

    def _choose_output_dir(self):
        d = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
        if d:
            self.output_dir = d
            self.out_path_var.set(d)

    def _open_output(self):
        if os.path.isdir(self.output_dir):
            os.startfile(self.output_dir)

    # ---- OCR Thread ----
    def _start_ocr(self):
        if not self.file_paths:
            self._log("[!] Chưa có file nào được chọn!\n", "warn")
            return

        if self.running:
            return

        self.output_dir = self.out_path_var.get().strip() or self.output_dir
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_open.config(state="disabled")
        self.progress["value"] = 0
        self.log_area.config(state="normal")
        self.log_area.delete(1.0, "end")
        self.log_area.config(state="disabled")

        self.msg_queue = queue.Queue()
        self.worker = OcrWorker(
            file_paths=list(self.file_paths),
            output_dir=self.output_dir,
            zoomin=self.zoomin.get(),
            msg_queue=self.msg_queue
        )
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()

        # Hien thi thong tin
        self._log("=" * 56 + "\n", "done")
        self._log(f"  Pipeline PDF -> TXT  (DeepDoc + VietOCR)\n", "done")
        self._log("=" * 56 + "\n", "done")
        self._log(f"  Số file: {len(self.file_paths)}\n")
        self._log(f"  Output : {self.output_dir}\n")
        self._log(f"  Zoomin : {self.zoomin.get()}\n")
        self._log("=" * 56 + "\n\n", "done")

    def _stop_ocr(self):
        if self.worker and self.running:
            self.worker.stop()
            self._log("[!] Đang dừng... (đợi file hiện tại xong)\n", "warn")

    def _on_ocr_done(self):
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_open.config(state="normal")
        self.lbl_status.config(text="Hoàn thành ✓")

    # ---- Poll queue tu thread OCR ----
    def _poll_queue(self):
        try:
            while True:
                msg_type, msg_data = self.msg_queue.get_nowait()

                if msg_type == "log":
                    self._log(msg_data)
                elif msg_type == "status":
                    self.lbl_status.config(text=msg_data)
                elif msg_type == "progress":
                    self.progress["value"] = msg_data
                elif msg_type == "done":
                    ok, fail = msg_data
                    self._on_ocr_done()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    def _log(self, text, tag=None):
        """Ghi log vao ScrolledText."""
        self.log_area.config(state="normal")
        self.log_area.insert("end", text, tag) if tag else self.log_area.insert("end", text)
        self.log_area.see("end")
        self.log_area.config(state="disabled")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = OcrGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()