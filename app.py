from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch

from processor import GoshuinProcessor, ProcessResult


class GoshuinScanApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GoshuinScan-OSS")
        self.root.geometry("900x560")
        self.root.minsize(780, 460)

        self.image_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str((Path.cwd() / "outputs").resolve()))
        self.use_gpu_var = tk.BooleanVar(value=torch.cuda.is_available())

        self._event_queue: queue.Queue = queue.Queue()
        self._processing = False

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="御朱印画像処理ツール", font=("Segoe UI", 14, "bold"))
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            container,
            text="処理フロー: docTR ドキュメント補正 -> RMBG-2.0 前景抽出 (黒墨/朱印保持)",
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
