import hashlib
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from huffman_compress import (
    compress_pure_huffman,
    compress_to_file,
    compress_huffman_lz77,
    export_huffman_tree_pdf,
)
from huffman_decompress import decompress_from_file

def _format_bytes(n: int) -> str:
    if n < 0:
        return "Không xác định"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1

    if i == 0:
        return f"{int(v)} {units[i]}"
    return f"{v:.2f} {units[i]}"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _files_equal(a: str, b: str) -> bool:
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
    except OSError:
        return False

    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ca = fa.read(1024 * 1024)
            cb = fb.read(1024 * 1024)
            if ca != cb:
                return False
            if not ca:
                return True


def _suggest_tree_pdf_path(compressed_path: str) -> str:
    base, _ext = os.path.splitext(compressed_path)
    candidate = base + "_huffman_tree.pdf"
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        alt = f"{base}_huffman_tree_{i}.pdf"
        if not os.path.exists(alt):
            return alt
        i += 1

    
class HuffmanApp(tk.Tk):
    # Khởi tạo cửa sổ UI và cấu hình style + biến trạng thái cho ứng dụng.
    def __init__(self) -> None:
        super().__init__()
        self.title("Huffman Coding – Nén & Giải nén")
        self.geometry("900x650")
        self.minsize(820, 520)
        self._center_window()

        self._bg = "#f8fafc"
        self._card = "#ffffff"
        self._text = "#0f172a"
        self._muted = "#64748b"
        self._accent = "#2563eb"
        self._border = "#e2e8f0"
        self.configure(bg=self._bg)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background=self._bg)
        style.configure("TLabel", background=self._bg, foreground=self._text)
        style.configure("HeaderTitle.TLabel", font=("Segoe UI", 18, "bold"), background=self._bg, foreground=self._text)
        style.configure("HeaderSubtitle.TLabel", font=("Segoe UI", 10), background=self._bg, foreground=self._muted)
        style.configure("Status.TLabel", font=("Segoe UI", 9), background=self._bg, foreground=self._muted)
        style.configure("Percent.TLabel", font=("Segoe UI", 9, "bold"), background=self._bg, foreground=self._muted)

        style.configure("Card.TLabel", background=self._card, foreground=self._text)
        style.configure("CardMuted.TLabel", background=self._card, foreground=self._muted)
        style.configure("CardKey.TLabel", font=("Segoe UI", 10, "bold"), background=self._card, foreground=self._text)
        style.configure("CardValueMono.TLabel", font=("Consolas", 9), background=self._card, foreground=self._text)
        style.configure("CardValue.TLabel", font=("Segoe UI", 9), background=self._card, foreground=self._text)
        style.configure("CardAccent.TLabel", font=("Segoe UI", 9, "bold"), background=self._card, foreground=self._accent)

        style.configure(
            "TButton",
            padding=(12, 8),
            relief="flat",
            borderwidth=1,
            bordercolor=self._border,
            lightcolor=self._border,
            darkcolor=self._border,
        )
        style.map(
            "TButton",
            background=[("active", "#eef2ff")],
        )

        style.configure(
            "Ghost.TButton",
            padding=(10, 8),
            background=self._card,
            foreground=self._text,
            relief="solid",
            borderwidth=1,
            bordercolor=self._border,
            lightcolor=self._border,
            darkcolor=self._border,
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#f1f5f9")],
        )

        style.configure(
            "TEntry",
            padding=(10, 8),
            fieldbackground=self._card,
            foreground=self._text,
            bordercolor=self._border,
            lightcolor=self._border,
            darkcolor=self._border,
        )

        style.configure("App.TNotebook", background=self._bg, borderwidth=0)
        style.configure(
            "App.TNotebook.Tab",
            padding=(16, 10),
            background=self._bg,
            foreground=self._muted,
            borderwidth=0,
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", self._card), ("active", "#f1f5f9")],
            foreground=[("selected", self._text), ("active", self._text)],
        )

        style.configure("Card.TFrame", background=self._card)
        style.configure(
            "Card.TLabelframe",
            background=self._card,
            bordercolor=self._border,
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=self._card,
            foreground=self._text,
            font=("Segoe UI", 10, "bold"),
        )

        style.configure(
            "Accent.TButton",
            background=self._accent,
            foreground="#ffffff",
            padding=(12, 8),
            font=("Segoe UI", 10, "bold"),
            relief="solid",
            borderwidth=1,
            bordercolor="#1d4ed8",
            lightcolor="#1d4ed8",
            darkcolor="#1d4ed8",
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")],
            foreground=[("disabled", "#f9fafb")],
        )

        style.configure(
            "App.Horizontal.TProgressbar",
            troughcolor="#e5e7eb",
            background=self._accent,
            bordercolor=self._border,
            lightcolor=self._border,
            darkcolor=self._border,
        )

        style.configure("TLabelframe", background=self._bg, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=self._bg, font=("Segoe UI", 10, "bold"))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", font=("Consolas", 10))

        self._last_compress_suggest = ""
        self._last_compress_input = ""
        self._last_decompress_suggest = ""

        self.compress_input_var = tk.StringVar()
        self.compress_output_var = tk.StringVar()
        self.export_tree_pdf_var = tk.BooleanVar(value=True)

        self.compress_input_var.trace_add("write", self._on_compress_input_changed)

        self.decompress_input_var = tk.StringVar()
        self.decompress_output_var = tk.StringVar()

        self.stat_file_var = tk.StringVar(value="-")
        self.stat_original_var = tk.StringVar(value="-")
        self.stat_compressed_var = tk.StringVar(value="-")
        self.stat_ratio_var = tk.StringVar(value="-")

        self.decomp_file_var = tk.StringVar(value="-")
        self.decomp_src_size_var = tk.StringVar(value="-")
        self.decomp_dst_size_var = tk.StringVar(value="-")
        self.decomp_ratio_var = tk.StringVar(value="-")

        self.progress: ttk.Progressbar | None = None
        self.progress_percent_var = tk.StringVar(value="0%")
        self.status_var = tk.StringVar(value="Sẵn sàng.")
        self.compress_button: ttk.Button | None = None
        self.compress_lz77_button: ttk.Button | None = None
        self.decompress_button: ttk.Button | None = None

        self._build_ui()

    # Chuẩn hoá đường dẫn để so sánh
    def _norm_path(self, path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    # Kiểm tra hợp lệ đường dẫn khi nén (chống trùng src/dst, xác nhận ghi đè).
    def _validate_compress_paths(self, src: str, dst: str) -> bool:
        if not src:
            messagebox.showwarning("Thiếu file nguồn", "Vui lòng chọn file văn bản cần nén.")
            return False
        if not dst:
            messagebox.showwarning("Thiếu file đích", "Vui lòng chọn file nén đích (.huff).")
            return False

        if self._norm_path(src) == self._norm_path(dst):
            messagebox.showerror(
                "Trùng tên file",
                "File nguồn và file đích đang trùng nhau. Vui lòng chọn file đích khác.",
            )
            return False

        if os.path.exists(dst):
            ok = messagebox.askyesno(
                "File đã tồn tại",
                "File đích đã tồn tại. Bạn có muốn ghi đè không?",
            )
            if not ok:
                return False

        return True

    # Kiểm tra hợp lệ đường dẫn khi giải nén (chống trùng src/dst, xác nhận ghi đè).
    def _validate_decompress_paths(self, src: str, dst: str) -> bool:
        if not src:
            messagebox.showwarning("Thiếu file nguồn", "Vui lòng chọn file .huff cần giải nén.")
            return False
        if not dst:
            messagebox.showwarning("Thiếu file đích", "Vui lòng chọn file giải nén đích (.txt).")
            return False

        if self._norm_path(src) == self._norm_path(dst):
            messagebox.showerror(
                "Trùng tên file",
                "File nguồn và file đích đang trùng nhau. Vui lòng chọn file đích khác.",
            )
            return False

        if os.path.exists(dst):
            ok = messagebox.askyesno(
                "File đã tồn tại",
                "File đích đã tồn tại. Bạn có muốn ghi đè không?",
            )
            if not ok:
                return False

        return True

    # Căn cửa sổ ra giữa màn hình một cách tương đối.
    def _center_window(self) -> None:
        self.update_idletasks()
        width = self.winfo_width() or 900
        height = self.winfo_height() or 620
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2) - 40
        self.geometry(f"{width}x{height}+{x}+{y}")

    # Dựng toàn bộ giao diện (header, tabs, form nén/giải nén, status/progress).
    def _build_ui(self) -> None:
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        title_label = ttk.Label(
            header_frame,
            text="Huffman Coding – Nén & Giải nén file văn bản",
            style="HeaderTitle.TLabel",
            anchor="center",
        )
        title_label.pack(fill=tk.X, pady=(0, 5))

        subtitle = ttk.Label(
            header_frame,
            text="Chọn tab để NÉN (.txt → .huff) hoặc GIẢI NÉN (.huff → .txt)",
            style="HeaderSubtitle.TLabel",
            anchor="center",
            justify="center",
            wraplength=820,
        )
        subtitle.pack(fill=tk.X)

        notebook = ttk.Notebook(self, style="App.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        tab_compress = ttk.Frame(notebook, style="TFrame")
        tab_decompress = ttk.Frame(notebook, style="TFrame")
        notebook.add(tab_compress, text="NÉN", padding=10)
        notebook.add(tab_decompress, text="GIẢI NÉN", padding=10)

        frm_compress = ttk.LabelFrame(tab_compress, text="NÉN FILE", padding=15, style="Card.TLabelframe")
        frm_compress.pack(fill=tk.X, padx=10, pady=(10, 10))

        frm_compress.columnconfigure(1, weight=1, minsize=400)
        frm_compress.columnconfigure(2, weight=0)

        ttk.Label(frm_compress, text="File văn bản nguồn (.txt):", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 8)
        )
        entry1 = ttk.Entry(frm_compress, textvariable=self.compress_input_var, font=("Segoe UI", 9))
        entry1.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(0, 8))
        btn1 = ttk.Button(frm_compress, text="Chọn file", command=self.browse_compress_input, style="Ghost.TButton")
        btn1.grid(row=0, column=2, sticky="e", pady=(0, 8))

        ttk.Label(frm_compress, text="File nén đích (.huff):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(0, 12)
        )
        entry2 = ttk.Entry(frm_compress, textvariable=self.compress_output_var, font=("Segoe UI", 9))
        entry2.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 12))
        btn2 = ttk.Button(frm_compress, text="Chọn file", command=self.browse_compress_output, style="Ghost.TButton")
        btn2.grid(row=1, column=2, sticky="e", pady=(0, 12))

        button_frame = ttk.Frame(frm_compress)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(5, 0))

        export_chk = ttk.Checkbutton(
            button_frame,
            text="Huffman Tree File (PDF)",
            variable=self.export_tree_pdf_var,
        )
        export_chk.pack(side=tk.LEFT, padx=(0, 14))
        
        self.compress_button = ttk.Button(
            button_frame,
            text="Nén Huffman thuần",
            command=self.handle_compress_pure,
            width=20,
            style="Ghost.TButton",
        )
        self.compress_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.compress_lz77_button = ttk.Button(
            button_frame,
            text="Nén Huffman + LZ77",
            command=self.handle_compress_lz77,
            width=20,
            style="Accent.TButton",
        )
        self.compress_lz77_button.pack(side=tk.LEFT)

        stats_frame = ttk.LabelFrame(tab_compress, text="Thông tin kết quả nén", padding=15, style="Card.TLabelframe")
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        stats_frame.columnconfigure(0, weight=0, minsize=150)
        stats_frame.columnconfigure(1, weight=1)

        ttk.Label(stats_frame, text="File gốc:", style="CardKey.TLabel").grid(
            row=0, column=0, sticky="e", padx=(0, 15), pady=(0, 10)
        )
        ttk.Label(stats_frame, textvariable=self.stat_file_var, style="CardValueMono.TLabel").grid(
            row=0, column=1, sticky="w", pady=(0, 10)
        )

        ttk.Label(stats_frame, text="Kích thước gốc:", style="CardKey.TLabel").grid(
            row=1, column=0, sticky="e", padx=(0, 15), pady=(0, 10)
        )
        ttk.Label(stats_frame, textvariable=self.stat_original_var, style="CardValue.TLabel").grid(
            row=1, column=1, sticky="w", pady=(0, 10)
        )

        ttk.Label(stats_frame, text="Kích thước sau nén:", style="CardKey.TLabel").grid(
            row=2, column=0, sticky="e", padx=(0, 15), pady=(0, 10)
        )
        ttk.Label(stats_frame, textvariable=self.stat_compressed_var, style="CardValue.TLabel").grid(
            row=2, column=1, sticky="w", pady=(0, 10)
        )

        ttk.Label(stats_frame, text="Tỉ lệ nén:", style="CardKey.TLabel").grid(
            row=3, column=0, sticky="e", padx=(0, 15), pady=(0, 0)
        )
        ttk.Label(stats_frame, textvariable=self.stat_ratio_var, style="CardAccent.TLabel").grid(
            row=3, column=1, sticky="w", pady=(0, 0)
        )

        frm_decompress = ttk.LabelFrame(tab_decompress, text="GIẢI NÉN FILE", padding=15, style="Card.TLabelframe")
        frm_decompress.pack(fill=tk.X, padx=10, pady=(10, 10))

        frm_decompress.columnconfigure(1, weight=1, minsize=400)
        frm_decompress.columnconfigure(2, weight=0)

        ttk.Label(frm_decompress, text="File nén nguồn (.huff):", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 8)
        )
        entry3 = ttk.Entry(frm_decompress, textvariable=self.decompress_input_var, font=("Segoe UI", 9))
        entry3.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(0, 8))
        btn3 = ttk.Button(frm_decompress, text="Chọn file", command=self.browse_decompress_input, style="Ghost.TButton")
        btn3.grid(row=0, column=2, sticky="e", pady=(0, 8))

        ttk.Label(frm_decompress, text="File giải nén đích (.txt):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(0, 12)
        )
        entry4 = ttk.Entry(frm_decompress, textvariable=self.decompress_output_var, font=("Segoe UI", 9))
        entry4.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 12))
        btn4 = ttk.Button(frm_decompress, text="Chọn file", command=self.browse_decompress_output, style="Ghost.TButton")
        btn4.grid(row=1, column=2, sticky="e", pady=(0, 12))

        self.decompress_button = ttk.Button(
            frm_decompress,
            text="Giải nén (Decompress)",
            command=self.handle_decompress,
            width=20,
            style="Accent.TButton",
        )
        self.decompress_button.grid(row=2, column=0, columnspan=3, pady=(5, 0))

        compare_button = ttk.Button(
            frm_decompress,
            text="So sánh với file gốc",
            command=self.handle_compare_with_original,
            width=20,
            style="Ghost.TButton",
        )
        compare_button.grid(row=3, column=0, columnspan=3, pady=(10, 0))

        dec_stats = ttk.LabelFrame(tab_decompress, text="Thông tin kết quả giải nén", padding=15, style="Card.TLabelframe")
        dec_stats.pack(fill=tk.X, padx=10, pady=(0, 10))

        dec_stats.columnconfigure(0, weight=0, minsize=150)
        dec_stats.columnconfigure(1, weight=1)

        ttk.Label(dec_stats, text="File nén (.huff):", style="CardKey.TLabel").grid(
            row=0, column=0, sticky="e", padx=(0, 15), pady=(0, 10)
        )
        ttk.Label(dec_stats, textvariable=self.decomp_file_var, style="CardValue.TLabel").grid(
            row=0, column=1, sticky="w", pady=(0, 10)
        )

        ttk.Label(dec_stats, text="Kích thước file nén:", style="CardKey.TLabel").grid(
            row=1, column=0, sticky="e", padx=(0, 15), pady=(0, 10)
        )
        ttk.Label(dec_stats, textvariable=self.decomp_src_size_var, style="CardValue.TLabel").grid(
            row=1, column=1, sticky="w", pady=(0, 10)
        )

        ttk.Label(dec_stats, text="Kích thước sau giải nén:", style="CardKey.TLabel").grid(
            row=2, column=0, sticky="e", padx=(0, 15), pady=(0, 0)
        )
        ttk.Label(dec_stats, textvariable=self.decomp_dst_size_var, style="CardValue.TLabel").grid(
            row=2, column=1, sticky="w", pady=(0, 0)
        )

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        progress_row = ttk.Frame(bottom_frame)
        progress_row.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))
        progress_row.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100, style="App.Horizontal.TProgressbar")
        self.progress.grid(row=0, column=0, sticky="ew")

        status_label = ttk.Label(
            bottom_frame,
            textvariable=self.status_var,
            anchor="w",
            style="Status.TLabel",
        )
        status_label.pack(fill=tk.X, side=tk.TOP)

    # Chọn file nguồn để nén (.txt).
    def browse_compress_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn file văn bản cần nén",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            prev_suggest = self._last_compress_suggest
            self.compress_input_var.set(path)
            self._update_compress_output_suggestion(prev_suggest)

    # Tự cập nhật gợi ý tên file nén đích khi người dùng đổi file nguồn nén.
    def _on_compress_input_changed(self, *_args) -> None:
        src = self.compress_input_var.get()
        if not src:
            return
        if not os.path.exists(src):
            return
        norm = self._norm_path(src)
        if norm == self._last_compress_input:
            return
        new_suggest = self.suggest_compress_output_path()
        if new_suggest:
            self.compress_output_var.set(new_suggest)
            self._last_compress_suggest = new_suggest
        self._last_compress_input = norm

    # Chọn file đích để lưu kết quả nén (.huff).
    def browse_compress_output(self) -> None:
        initial = self.suggest_compress_output_path()
        path = filedialog.asksaveasfilename(
            title="Chọn file nén đích (.huff)",
            initialfile=initial,
            defaultextension=".huff",
            filetypes=[("Huffman files", "*.huff"), ("All files", "*.*")],
        )
        if path:
            self.compress_output_var.set(path)

    # Gợi ý đường dẫn file nén đích dựa trên file nguồn.
    def suggest_compress_output_path(self) -> str:
        src = self.compress_input_var.get()
        if not src:
            return ""
        src_dir, src_name = os.path.split(src)
        base, ext = os.path.splitext(src_name)
        if not base:
            return src 
        out_name = base + ".huff"
        return os.path.join(src_dir, out_name) if src_dir else out_name

    # Cập nhật tên file nén đích khi đổi file nguồn.
    def _update_compress_output_suggestion(self, prev_suggest: str) -> None:
        new_suggest = self.suggest_compress_output_path()
        current_out = self.compress_output_var.get()
        if not current_out or current_out == prev_suggest:
            self.compress_output_var.set(new_suggest)
        self._last_compress_suggest = new_suggest

    # Chọn file nén nguồn để giải nén (.huff).
    def browse_decompress_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn file .huff cần giải nén",
            filetypes=[("Huffman files", "*.huff"), ("All files", "*.*")],
        )
        if path:
            prev_suggest = self._last_decompress_suggest
            self.decompress_input_var.set(path)
            self._update_decompress_output_suggestion(prev_suggest)

    # Chọn file đích để lưu kết quả giải nén (.txt).
    def browse_decompress_output(self) -> None:
        initial = self.suggest_decompress_output_path()
        path = filedialog.asksaveasfilename(
            title="Chọn file văn bản đích (.txt)",
            initialfile=initial,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.decompress_output_var.set(path)

    # Gợi ý đường dẫn file giải nén đích dựa trên file nguồn.
    def suggest_decompress_output_path(self) -> str:
        src = self.decompress_input_var.get()
        if not src:
            return ""
        src_dir, src_name = os.path.split(src)
        base, ext = os.path.splitext(src_name)
        if not base:
            return src
        if ext.lower() == ".huff":
            out_name = base + "_decompressed.txt"
        else:
            out_name = base + ".txt"
        return os.path.join(src_dir, out_name) if src_dir else out_name

    # Cập nhật tên file giải nén đích khi đổi file nguồn
    def _update_decompress_output_suggestion(self, prev_suggest: str) -> None:
        new_suggest = self.suggest_decompress_output_path()
        current_out = self.decompress_output_var.get()
        if not current_out or current_out == prev_suggest:
            self.decompress_output_var.set(new_suggest)
        self._last_decompress_suggest = new_suggest

    #  Quản lý trạng thái / tiến trình chung của ứng dụng.
    def set_busy(self, busy: bool, message: str = "") -> None:
        if message:
            self.status_var.set(message)
        else:
            self.status_var.set("Sẵn sàng.")

        state = "disabled" if busy else "normal"
        if self.compress_button is not None:
            self.compress_button.config(state=state)
        if self.compress_lz77_button is not None:
            self.compress_lz77_button.config(state=state)
        if self.decompress_button is not None:
            self.decompress_button.config(state=state)

        if self.progress is not None:
            mode = str(self.progress["mode"])
            if mode == "indeterminate":
                if busy:
                    self.progress.start(10)
                else:
                    self.progress.stop()

        if not busy:
            self.progress_percent_var.set("0%")

        self.update_idletasks()

    # Xử lý nén Huffman.
    def handle_compress_pure(self) -> None:
        src = self.compress_input_var.get()
        dst = self.compress_output_var.get()
        if not src:
            messagebox.showwarning("Thiếu file nguồn", "Vui lòng chọn file văn bản cần nén.")
            return
        prev_suggest = self._last_compress_suggest
        new_suggest = self.suggest_compress_output_path()
        if not dst or dst == prev_suggest:
            dst = new_suggest
            self.compress_output_var.set(dst)
        self._last_compress_suggest = new_suggest
        if not self._validate_compress_paths(src, dst):
            return
        if self.progress is not None:
            self.progress.config(mode="determinate", maximum=100, value=0)
        self.set_busy(True, "Đang nén file (Huffman), vui lòng chờ 0%...")

        def worker() -> None:
            try:
                def progress_cb(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    percent = int(done * 100 / total)
                    if percent > 100:
                        percent = 100

                    def ui_update() -> None:
                        if self.progress is not None:
                            self.progress["value"] = percent
                        self.progress_percent_var.set(f"{percent}%")
                        self.status_var.set(f"Đang nén file (Huffman), vui lòng chờ {percent}%...")

                    self.after(0, ui_update)

                freq_table, codes = compress_pure_huffman(src, dst, progress_callback=progress_cb)

                pdf_path = ""
                if self.export_tree_pdf_var.get():
                    pdf_path = _suggest_tree_pdf_path(dst)
                    try:
                        export_huffman_tree_pdf(
                            freq_table,
                            pdf_path,
                            codes=codes,
                            title=f"Huffman tree: {os.path.basename(src)}",
                        )
                    except Exception as e: 
                        pdf_path = ""

                        def warn_pdf() -> None:
                            messagebox.showwarning(
                                "Cảnh báo",
                                "Nén thành công nhưng không xuất được cây Huffman ra PDF.\n" + str(e),
                            )

                        self.after(0, warn_pdf)
                try:
                    original_size = os.path.getsize(src)
                except OSError:
                    original_size = -1
                try:
                    compressed_size = os.path.getsize(dst)
                except OSError:
                    compressed_size = -1

                def on_done() -> None:
                    self.stat_file_var.set(src)
                    self.stat_original_var.set(
                        _format_bytes(original_size)
                    )
                    self.stat_compressed_var.set(
                        _format_bytes(compressed_size)
                    )
                    if original_size > 0 and compressed_size >= 0:
                        ratio = (1 - compressed_size / original_size) * 100
                        self.stat_ratio_var.set(f"{ratio:.2f}% dung lượng giảm")
                    else:
                        self.stat_ratio_var.set("Không xác định")
                    if self.progress is not None:
                        self.progress["value"] = 100
                    self.progress_percent_var.set("100%")
                    self.set_busy(False, "Nén xong.")

                    msg = f"Đã nén file (Huffman):\n{src}\n=>\n{dst}"
                    if pdf_path:
                        msg += f"\n\nCây Huffman (PDF):\n{pdf_path}"
                    messagebox.showinfo(
                        "Thành công",
                        msg,
                    )

                self.after(0, on_done)
            except Exception as e:

                def on_error() -> None:
                    self.set_busy(False, "Lỗi khi nén.")
                    messagebox.showerror("Lỗi nén", str(e))

                self.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def handle_compress(self) -> None:
        src = self.compress_input_var.get()
        dst = self.compress_output_var.get()
        if not src:
            messagebox.showwarning("Thiếu file nguồn", "Vui lòng chọn file văn bản cần nén.")
            return
        prev_suggest = self._last_compress_suggest
        new_suggest = self.suggest_compress_output_path()
        if not dst or dst == prev_suggest:
            dst = new_suggest
            self.compress_output_var.set(dst)
        self._last_compress_suggest = new_suggest
        if not self._validate_compress_paths(src, dst):
            return
        if self.progress is not None:
            self.progress.config(mode="determinate", maximum=100, value=0)
        self.set_busy(True, "Đang nén file, vui lòng chờ 0%...")

        def worker() -> None:
            try:
                def progress_cb(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    percent = int(done * 100 / total)
                    if percent > 100:
                        percent = 100

                    def ui_update() -> None:
                        if self.progress is not None:
                            self.progress["value"] = percent
                        self.progress_percent_var.set(f"{percent}%")
                        self.status_var.set(f"Đang nén file, vui lòng chờ {percent}%...")

                    self.after(0, ui_update)

                freq_table, codes = compress_to_file(src, dst, progress_callback=progress_cb)

                pdf_path = ""
                if self.export_tree_pdf_var.get():
                    pdf_path = _suggest_tree_pdf_path(dst)
                    try:
                        export_huffman_tree_pdf(
                            freq_table,
                            pdf_path,
                            codes=codes,
                            title=f"Huffman tree: {os.path.basename(src)}",
                        )
                    except Exception as e:  
                        pdf_path = ""

                        def warn_pdf() -> None:
                            messagebox.showwarning(
                                "Cảnh báo",
                                "Nén thành công nhưng không xuất được cây Huffman ra PDF.\n" + str(e),
                            )

                        self.after(0, warn_pdf)

                try:
                    original_size = os.path.getsize(src)
                except OSError:
                    original_size = -1
                try:
                    compressed_size = os.path.getsize(dst)
                except OSError:
                    compressed_size = -1

                def on_done() -> None:
                    self.stat_file_var.set(src)
                    self.stat_original_var.set(_format_bytes(original_size))
                    self.stat_compressed_var.set(_format_bytes(compressed_size))
                    if original_size > 0 and compressed_size >= 0:
                        ratio = (1 - compressed_size / original_size) * 100
                        self.stat_ratio_var.set(f"{ratio:.2f}% dung lượng giảm")
                    else:
                        self.stat_ratio_var.set("Không xác định")
                    if self.progress is not None:
                        self.progress["value"] = 100
                    self.progress_percent_var.set("100%")
                    self.set_busy(False, "Nén xong.")

                    msg = f"Đã nén file:\n{src}\n=>\n{dst}"
                    if pdf_path:
                        msg += f"\n\nCây Huffman (PDF):\n{pdf_path}"
                    messagebox.showinfo("Thành công", msg)

                self.after(0, on_done)
            except Exception as e:  

                def on_error() -> None:
                    self.set_busy(False, "Lỗi khi nén.")
                    messagebox.showerror("Lỗi nén", str(e))

                self.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    # Xử lý nén Huffman + LZ77.
    def handle_compress_lz77(self) -> None:
        src = self.compress_input_var.get()
        dst = self.compress_output_var.get()
        if not src:
            messagebox.showwarning("Thiếu file nguồn", "Vui lòng chọn file văn bản cần nén.")
            return
        prev_suggest = self._last_compress_suggest
        new_suggest = self.suggest_compress_output_path()
        if not dst or dst == prev_suggest:
            dst = new_suggest
            self.compress_output_var.set(dst)
        self._last_compress_suggest = new_suggest
        if not self._validate_compress_paths(src, dst):
            return
        if self.progress is not None:
            self.progress.config(mode="determinate", maximum=100, value=0)
        self.set_busy(True, "Đang nén file (Huffman + LZ77), vui lòng chờ 0%...")

        def worker() -> None:
            try:
                def progress_cb(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    percent = int(done * 100 / total)
                    if percent > 100:
                        percent = 100

                    def ui_update() -> None:
                        if self.progress is not None:
                            self.progress["value"] = percent
                        self.progress_percent_var.set(f"{percent}%")
                        self.status_var.set(f"Đang nén file (Huffman + LZ77), vui lòng chờ {percent}%...")

                    self.after(0, ui_update)

                freq_table, codes = compress_huffman_lz77(
                    src,
                    dst,
                    progress_callback=progress_cb,
                    fast=True,
                )

                pdf_path = ""
                if self.export_tree_pdf_var.get() and freq_table:
                    pdf_path = _suggest_tree_pdf_path(dst)
                    try:
                        export_huffman_tree_pdf(
                            freq_table,
                            pdf_path,
                            codes=codes,
                            title=f"Huffman tree (LZ77+Huffman-bytes): {os.path.basename(src)}",
                        )
                    except Exception as e:
                        pdf_path = ""

                        def warn_pdf() -> None:
                            messagebox.showwarning(
                                "Cảnh báo",
                                "Nén thành công nhưng không xuất được cây Huffman ra PDF.\n" + str(e),
                            )

                        self.after(0, warn_pdf)
                try:
                    original_size = os.path.getsize(src)
                except OSError:
                    original_size = -1
                try:
                    compressed_size = os.path.getsize(dst)
                except OSError:
                    compressed_size = -1

                def on_done() -> None:
                    self.stat_file_var.set(src)
                    self.stat_original_var.set(
                        _format_bytes(original_size)
                    )
                    self.stat_compressed_var.set(
                        _format_bytes(compressed_size)
                    )
                    if original_size > 0 and compressed_size >= 0:
                        ratio = (1 - compressed_size / original_size) * 100
                        self.stat_ratio_var.set(f"{ratio:.2f}% dung lượng giảm")
                    else:
                        self.stat_ratio_var.set("Không xác định")
                    if self.progress is not None:
                        self.progress["value"] = 100
                    self.progress_percent_var.set("100%")
                    self.set_busy(False, "Nén xong.")
                    msg = f"Đã nén file (Huffman + LZ77):\n{src}\n=>\n{dst}"
                    if pdf_path:
                        msg += f"\n\nCây Huffman (PDF):\n{pdf_path}"
                    messagebox.showinfo("Thành công", msg)

                self.after(0, on_done)
            except Exception as e:

                def on_error() -> None:
                    self.set_busy(False, "Lỗi khi nén.")
                    messagebox.showerror("Lỗi nén", str(e))

                self.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    # Giải nén file .huff ra .txt 
    def handle_decompress(self) -> None:
        src = self.decompress_input_var.get()
        dst = self.decompress_output_var.get()
        if not src:
            messagebox.showwarning("Thiếu file nguồn", "Vui lòng chọn file .huff cần giải nén.")
            return
        if not dst:
            dst = self.suggest_decompress_output_path()
            self.decompress_output_var.set(dst)
        if not self._validate_decompress_paths(src, dst):
            return
        if self.progress is not None:
            self.progress.config(mode="determinate", maximum=100, value=0)
        self.set_busy(True, "Đang giải nén file, vui lòng chờ 0%...")

        def worker() -> None:
            try:
                def progress_cb(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    percent = int(done * 100 / total)
                    if percent > 100:
                        percent = 100

                    def ui_update() -> None:
                        if self.progress is not None:
                            self.progress["value"] = percent
                        self.progress_percent_var.set(f"{percent}%")
                        self.status_var.set(f"Đang giải nén file, vui lòng chờ {percent}%...")

                    self.after(0, ui_update)

                decompress_from_file(src, dst, progress_callback=progress_cb)
                try:
                    src_size = os.path.getsize(src)
                except OSError:
                    src_size = -1
                try:
                    dst_size = os.path.getsize(dst)
                except OSError:
                    dst_size = -1

                def on_done() -> None:
                    self.decomp_file_var.set(src)
                    self.decomp_src_size_var.set(
                        _format_bytes(src_size)
                    )
                    self.decomp_dst_size_var.set(
                        _format_bytes(dst_size)
                    )
                    if self.progress is not None:
                        self.progress["value"] = 100
                    self.progress_percent_var.set("100%")
                    self.set_busy(False, "Giải nén xong.")
                    messagebox.showinfo("Thành công", f"Đã giải nén file:\n{src}\n=>\n{dst}")

                self.after(0, on_done)
            except Exception as e: 

                def on_error() -> None:
                    self.set_busy(False, "Lỗi khi giải nén.")
                    messagebox.showerror("Lỗi giải nén", str(e))

                self.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def handle_compare_with_original(self) -> None:
        decompressed = self.decompress_output_var.get()
        if not decompressed:
            messagebox.showwarning("Thiếu file", "Chưa có file giải nén để so sánh.")
            return
        if not os.path.exists(decompressed):
            messagebox.showwarning("Thiếu file", "File giải nén không tồn tại.")
            return

        original = filedialog.askopenfilename(
            title="Chọn file gốc để so sánh",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not original:
            return
        if not os.path.exists(original):
            messagebox.showwarning("Thiếu file", "File gốc không tồn tại.")
            return

        try:
            equal = _files_equal(original, decompressed)
            osz = os.path.getsize(original)
            dsz = os.path.getsize(decompressed)
            oh = _sha256_file(original)
            dh = _sha256_file(decompressed)
        except Exception as e:
            messagebox.showerror("Lỗi", "Không thể so sánh file.\n" + str(e))
            return

        msg = (
            f"File gốc:\n{original}\n\n"
            f"File giải nén:\n{decompressed}\n\n"
            f"Kích thước (gốc / giải nén): {osz} / {dsz} bytes\n"
            f"SHA-256 (gốc): {oh}\n"
            f"SHA-256 (giải nén): {dh}\n"
        )

        if equal:
            messagebox.showinfo("Kết quả so sánh", "HAI FILE GIỐNG NHAU (100%).\n\n" + msg)
        else:
            messagebox.showwarning("Kết quả so sánh", "HAI FILE KHÁC NHAU.\n\n" + msg)
if __name__ == "__main__":
    app = HuffmanApp()
    app.mainloop()

