from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from processor import GoshuinProcessor, ProcessResult

if hasattr(Image, "Resampling"):
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:  # pragma: no cover - Pillow backward compatibility
    _RESAMPLE_LANCZOS = Image.LANCZOS


class GoshuinScanApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GoshuinScan-OSS")
        self.root.geometry("980x700")
        self.root.minsize(860, 560)

        self.image_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str((Path.cwd() / "outputs").resolve()))
        self.use_gpu_var = tk.BooleanVar(value=torch.cuda.is_available())

        self._event_queue: queue.Queue = queue.Queue()
        self._processing = False
        self._preview_labels: dict[str, ttk.Label] = {}
        self._preview_images: dict[str, Image.Image | None] = {
            "input": None,
            "enhanced": None,
            "transparent": None,
        }
        self._preview_photo_refs: dict[str, ImageTk.PhotoImage] = {}

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="御朱印画像処理ツール", font=("Segoe UI", 14, "bold"))
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            container,
            text="処理フロー: 視角補正 (RMBG) -> docTR 補正 -> 背景除去 (黒墨/朱印保持)",
        )
        subtitle.pack(anchor=tk.W, pady=(2, 12))

        input_frame = ttk.LabelFrame(container, text="入力", padding=10)
        input_frame.pack(fill=tk.X)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="画像:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        image_entry = ttk.Entry(input_frame, textvariable=self.image_path_var)
        image_entry.grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Button(input_frame, text="画像を選択", command=self._select_image).grid(
            row=0, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )

        ttk.Label(input_frame, text="出力フォルダー:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4
        )
        output_entry = ttk.Entry(input_frame, textvariable=self.output_dir_var)
        output_entry.grid(row=1, column=1, sticky=tk.EW, pady=4)
        ttk.Button(input_frame, text="フォルダーを選択", command=self._select_output_dir).grid(
            row=1, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )

        options_frame = ttk.Frame(container)
        options_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            options_frame,
            text="GPU (CUDA) を使用",
            variable=self.use_gpu_var,
        ).pack(anchor=tk.W)

        actions_frame = ttk.Frame(container)
        actions_frame.pack(fill=tk.X, pady=(12, 0))
        self.process_button = ttk.Button(actions_frame, text="処理開始", command=self._start_process)
        self.process_button.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(actions_frame, mode="indeterminate", length=180)
        self.progress.pack(side=tk.LEFT, padx=(12, 0))

        preview_frame = ttk.LabelFrame(container, text="プレビュー", padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.configure(height=300)
        preview_frame.grid_propagate(False)

        preview_content = ttk.Frame(preview_frame)
        preview_content.grid(row=0, column=0, sticky=tk.NSEW)
        preview_content.columnconfigure(0, weight=1)
        preview_content.columnconfigure(1, weight=1)
        preview_content.columnconfigure(2, weight=1)
        preview_content.rowconfigure(0, weight=1)

        self._create_preview_panel(
            parent=preview_content,
            key="input",
            title="入力画像",
            placeholder="画像を選択してください。",
            column=0,
        )
        self._create_preview_panel(
            parent=preview_content,
            key="enhanced",
            title="補正画像",
            placeholder="処理後の補正画像がここに表示されます。",
            column=1,
        )
        self._create_preview_panel(
            parent=preview_content,
            key="transparent",
            title="透過画像",
            placeholder="処理後の透過画像がここに表示されます。",
            column=2,
        )

        log_frame = ttk.LabelFrame(container, text="ログ", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)

        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scroll.set)

    def _select_image(self) -> None:
        path = filedialog.askopenfilename(
            title="御朱印画像を選択",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.image_path_var.set(path)
            self._update_preview_from_path("input", Path(path))
            self._set_preview_placeholder("enhanced", "処理後の補正画像がここに表示されます。")
            self._set_preview_placeholder("transparent", "処理後の透過画像がここに表示されます。")

    def _select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="出力フォルダーを選択")
        if path:
            self.output_dir_var.set(path)

    def _start_process(self) -> None:
        if self._processing:
            return

        image_text = self.image_path_var.get().strip()
        output_text = self.output_dir_var.get().strip()

        image_path = Path(image_text)
        output_dir = Path(output_text)

        if not image_path.exists() or not image_path.is_file():
            messagebox.showerror("入力エラー", "有効な画像ファイルを選択してください。")
            return

        if not output_text:
            messagebox.showerror("入力エラー", "出力フォルダーを選択してください。")
            return

        self._update_preview_from_path("input", image_path)
        self._set_processing(True)
        self._append_log(f"処理開始: {image_path}")

        worker = threading.Thread(
            target=self._process_worker,
            args=(image_path, output_dir, self.use_gpu_var.get()),
            daemon=True,
        )
        worker.start()

    def _process_worker(self, image_path: Path, output_dir: Path, use_gpu: bool) -> None:
        try:
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            self._event_queue.put(("log", f"推論デバイス: {device}"))
            processor = GoshuinProcessor(
                device=device,
                log_callback=lambda msg: self._event_queue.put(("log", msg)),
            )
            result = processor.process(image_path, output_dir)
            self._event_queue.put(("done", result))
        except Exception as exc:  # pragma: no cover - GUI runtime path
            self._event_queue.put(("error", str(exc)))

    def _poll_events(self) -> None:
        while True:
            try:
                event_type, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "log":
                self._append_log(payload)
            elif event_type == "done":
                self._on_process_done(payload)
            elif event_type == "error":
                self._on_process_error(payload)

        self.root.after(100, self._poll_events)

    def _on_process_done(self, result: ProcessResult) -> None:
        self._set_processing(False)
        self._append_log("処理が完了しました。")
        self._append_log(f"補正画像: {result.enhanced_path}")
        self._append_log(f"透過画像: {result.transparent_path}")
        
        self._update_preview_from_path("enhanced", result.enhanced_path)
        self._update_preview_from_path("transparent", result.transparent_path)
        
        # 強制的にUIを更新して、ダイアログ表示前にプレビューを描画させる
        self.root.update_idletasks()
        
        messagebox.showinfo(
            "完了",
            (
                "処理が完了しました。\n\n"
                f"補正画像:\n{result.enhanced_path}\n\n"
                f"透過画像:\n{result.transparent_path}"
            ),
        )

    def _on_process_error(self, error_message: str) -> None:
        self._set_processing(False)
        self._append_log(f"処理失敗: {error_message}")
        messagebox.showerror("処理失敗", error_message)

    def _set_processing(self, processing: bool) -> None:
        self._processing = processing
        if processing:
            self.process_button.configure(state=tk.DISABLED)
            self.progress.start(10)
        else:
            self.process_button.configure(state=tk.NORMAL)
            self.progress.stop()

    def _create_preview_panel(
        self,
        parent: ttk.Frame,
        key: str,
        title: str,
        placeholder: str,
        column: int,
    ) -> None:
        panel = ttk.LabelFrame(parent, text=title, padding=6)
        panel.grid(row=0, column=column, sticky=tk.NSEW, padx=4)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        preview_label = ttk.Label(panel, anchor=tk.CENTER, justify=tk.CENTER)
        preview_label.grid(row=0, column=0, sticky=tk.NSEW)
        preview_label.bind("<Configure>", lambda _event, image_key=key: self._render_preview(image_key))
        self._preview_labels[key] = preview_label
        self._set_preview_placeholder(key, placeholder)

    def _set_preview_placeholder(self, key: str, message: str) -> None:
        label = self._preview_labels[key]
        self._preview_images[key] = None
        self._preview_photo_refs.pop(key, None)
        label.configure(image="", text=message, wraplength=520)

    def _update_preview_from_path(self, key: str, path: Path) -> None:
        try:
            with Image.open(path) as pil_image:
                if key == "transparent":
                    loaded = pil_image.convert("RGBA")
                else:
                    loaded = pil_image.convert("RGB")
        except Exception as exc:
            self._set_preview_placeholder(key, f"プレビューの読み込みに失敗しました。\n{exc}")
            return

        self._preview_images[key] = loaded
        self._render_preview(key, force=True)

    def _render_preview(self, key: str, force: bool = False) -> None:
        label = self._preview_labels[key]
        source_image = self._preview_images.get(key)
        if source_image is None:
            return

        w, h = label.winfo_width(), label.winfo_height()

        # ウィンドウがまだ描画されていない場合はリトライ
        if w <= 10 or h <= 10:
            self.root.after(60, lambda image_key=key: self._render_preview(image_key, force=force))
            return

        target_width = max(w - 16, 1)
        target_height = max(h - 16, 1)

        # 画面リサイズ等の場合、現在の画像のサイズと目標サイズを比較し無駄な再描画を防ぐ
        if not force:
            current_tk = self._preview_photo_refs.get(key)
            if current_tk and abs(current_tk.width() - target_width) < 10 and abs(current_tk.height() - target_height) < 10:
                return

        preview = source_image.copy()
        preview.thumbnail((target_width, target_height), _RESAMPLE_LANCZOS)
        preview_tk = ImageTk.PhotoImage(preview)
        self._preview_photo_refs[key] = preview_tk
        label.configure(image=preview_tk, text="")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    GoshuinScanApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
