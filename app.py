from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from processor import GoshuinProcessor, ProcessResult

def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_PROJECT_ROOT = Path(__file__).resolve().parent
_load_dotenv(_PROJECT_ROOT / ".env")

# LoRA モデルのパス設定 (.env で上書き可能)
_LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH")
_LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH")
_LORA_PROMPT = (
    "この御朱印はなにですか？日本語で出力してください。"
    "必ず以下のJSON形式で出力してください:\n"
    '{"name":"神社またはお寺名","date":"日付","text":"最も目立つ文字","mark":"印の名前"}'
)

_SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

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
        self.folder_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str((Path.cwd() / "outputs").resolve()))
        self.use_gpu_var = tk.BooleanVar(value=torch.cuda.is_available())
        self.use_ai_var  = tk.BooleanVar(value=False)

        self._event_queue: queue.Queue = queue.Queue()
        self._processing = False
        self._lora_model = None   # lazy load: (model, processor)
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

        ttk.Label(input_frame, text="画像フォルダー:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4
        )
        folder_entry = ttk.Entry(input_frame, textvariable=self.folder_path_var)
        folder_entry.grid(row=1, column=1, sticky=tk.EW, pady=4)
        ttk.Button(input_frame, text="フォルダーを選択", command=self._select_input_dir).grid(
            row=1, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )

        ttk.Label(input_frame, text="出力フォルダー:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 8), pady=4
        )
        output_entry = ttk.Entry(input_frame, textvariable=self.output_dir_var)
        output_entry.grid(row=2, column=1, sticky=tk.EW, pady=4)
        ttk.Button(input_frame, text="フォルダーを選択", command=self._select_output_dir).grid(
            row=2, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )
        ttk.Label(
            input_frame,
            text="※ 画像フォルダーを指定した場合は、フォルダー内の画像を一括処理します。",
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))

        options_frame = ttk.Frame(container)
        options_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            options_frame,
            text="GPU (CUDA) を使用",
            variable=self.use_gpu_var,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            options_frame,
            text="AI 識別 (LoRA モデル)",
            variable=self.use_ai_var,
        ).pack(side=tk.LEFT, padx=(24, 0))

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

        # AI 識別結果パネル
        ai_frame = ttk.LabelFrame(container, text="AI 識別結果 (LoRA)", padding=8)
        ai_frame.pack(fill=tk.X, pady=(10, 0))
        ai_frame.columnconfigure(0, weight=1)
        ai_frame.columnconfigure(1, weight=1)
        ai_frame.columnconfigure(2, weight=1)
        ai_frame.columnconfigure(3, weight=1)

        self._ai_labels: dict[str, ttk.Label] = {}
        for col, (key, display) in enumerate([
            ("name", "神社・寺名"),
            ("date", "日付"),
            ("text", "主な文字"),
            ("mark", "印"),
        ]):
            cell = ttk.Frame(ai_frame)
            cell.grid(row=0, column=col, sticky=tk.EW, padx=6)
            ttk.Label(cell, text=display, font=("Segoe UI", 8), foreground="gray").pack(anchor=tk.W)
            val_label = ttk.Label(cell, text="--", font=("Segoe UI", 10, "bold"), wraplength=200, justify=tk.LEFT)
            val_label.pack(anchor=tk.W)
            self._ai_labels[key] = val_label

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
            self.folder_path_var.set("")
            self._update_preview_from_path("input", Path(path))
            self._set_preview_placeholder("enhanced", "処理後の補正画像がここに表示されます。")
            self._set_preview_placeholder("transparent", "処理後の透過画像がここに表示されます。")

    def _select_input_dir(self) -> None:
        path = filedialog.askdirectory(title="画像フォルダーを選択")
        if path:
            folder_path = Path(path)
            self.folder_path_var.set(path)
            self.image_path_var.set("")

            images = self._collect_images_from_folder(folder_path)
            if images:
                self._update_preview_from_path("input", images[0])
                self._append_log(f"フォルダー内の検出画像数: {len(images)}")
            else:
                self._set_preview_placeholder("input", "フォルダー内に画像が見つかりません。")

            self._set_preview_placeholder("enhanced", "処理後の補正画像がここに表示されます。")
            self._set_preview_placeholder("transparent", "処理後の透過画像がここに表示されます。")

    def _select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="出力フォルダーを選択")
        if path:
            self.output_dir_var.set(path)

    @staticmethod
    def _collect_images_from_folder(folder_path: Path) -> list[Path]:
        image_paths: list[Path] = []
        for p in sorted(folder_path.iterdir()):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.append(p)
        return image_paths

    def _start_process(self) -> None:
        if self._processing:
            return

        image_text = self.image_path_var.get().strip()
        folder_text = self.folder_path_var.get().strip()
        output_text = self.output_dir_var.get().strip()

        output_dir = Path(output_text)

        if not output_text:
            messagebox.showerror("入力エラー", "出力フォルダーを選択してください。")
            return

        image_paths: list[Path]
        if folder_text:
            folder_path = Path(folder_text)
            if not folder_path.exists() or not folder_path.is_dir():
                messagebox.showerror("入力エラー", "有効な画像フォルダーを選択してください。")
                return
            image_paths = self._collect_images_from_folder(folder_path)
            if not image_paths:
                messagebox.showerror("入力エラー", "指定フォルダー内に処理可能な画像がありません。")
                return
        else:
            image_path = Path(image_text)
            if not image_path.exists() or not image_path.is_file():
                messagebox.showerror("入力エラー", "有効な画像ファイルを選択してください。")
                return
            image_paths = [image_path]

        self._update_preview_from_path("input", image_paths[0])
        self._set_processing(True)
        self._append_log(f"処理開始: {len(image_paths)} 件")

        worker = threading.Thread(
            target=self._process_worker,
            args=(image_paths, output_dir, self.use_gpu_var.get(), self.use_ai_var.get()),
            daemon=True,
        )
        worker.start()

    def _process_worker(
        self, image_paths: list[Path], output_dir: Path, use_gpu: bool, use_ai: bool = False
    ) -> None:
        try:
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            self._event_queue.put(("log", f"推論デバイス: {device}"))
            processor = GoshuinProcessor(
                device=device,
                log_callback=lambda msg: self._event_queue.put(("log", msg)),
            )
            total = len(image_paths)
            success = 0
            failed: list[tuple[str, str]] = []

            for index, image_path in enumerate(image_paths, start=1):
                self._event_queue.put(("log", f"[{index}/{total}] 処理開始: {image_path}"))
                try:
                    result = processor.process(image_path, output_dir)
                    success += 1
                    self._event_queue.put(
                        (
                            "done_one",
                            {
                                "index": index,
                                "total": total,
                                "image_path": image_path,
                                "result": result,
                            },
                        )
                    )
                    if use_ai:
                        self._event_queue.put(("log", f"[{index}/{total}] AI 識別開始..."))
                        ai_result = self._run_lora_infer(str(image_path))
                        self._event_queue.put(("ai_result", ai_result))
                except Exception as exc:
                    failed.append((str(image_path), str(exc)))
                    self._event_queue.put(("log", f"[{index}/{total}] 処理失敗: {image_path} -> {exc}"))

            self._event_queue.put(
                (
                    "batch_done",
                    {
                        "total": total,
                        "success": success,
                        "failed": failed,
                    },
                )
            )

        except Exception as exc:  # pragma: no cover - GUI runtime path
            self._event_queue.put(("error", str(exc)))

    def _get_lora_model(self):
        """LoRA モデルを遅延ロードしてキャッシュする。"""
        if self._lora_model is None:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            from peft import PeftModel
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                _LORA_MODEL_PATH, torch_dtype=dtype, device_map="auto"
            )
            model = PeftModel.from_pretrained(base, _LORA_ADAPTER_PATH)
            model.eval()
            lora_processor = AutoProcessor.from_pretrained(_LORA_MODEL_PATH)
            self._lora_model = (model, lora_processor)
        return self._lora_model

    def _run_lora_infer(self, image_path: str) -> dict:
        """LoRA モデルで御朱印を識別し、辞書を返す。"""
        try:
            model, lora_processor = self._get_lora_model()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": _LORA_PROMPT},
                ],
            }]
            inputs = lora_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=256)
            trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
            text = lora_processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            # JSONブロックを抽出してパース
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {"name": text, "date": "", "text": "", "mark": ""}
        except Exception as exc:
            return {"name": f"エラー: {exc}", "date": "", "text": "", "mark": ""}

    def _poll_events(self) -> None:
        while True:
            try:
                event_type, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "log":
                self._append_log(payload)
            elif event_type == "done_one":
                self._on_process_done(payload)
            elif event_type == "batch_done":
                self._on_batch_done(payload)
            elif event_type == "error":
                self._on_process_error(payload)
            elif event_type == "ai_result":
                self._update_ai_result(payload)

        self.root.after(100, self._poll_events)

    def _on_process_done(self, payload: dict) -> None:
        index = payload["index"]
        total = payload["total"]
        image_path = payload["image_path"]
        result: ProcessResult = payload["result"]

        self._append_log(f"[{index}/{total}] 完了: {Path(image_path).name}")
        self._append_log(f"[{index}/{total}] 補正画像: {result.enhanced_path}")
        self._append_log(f"[{index}/{total}] 透過画像: {result.transparent_path}")
        
        self._update_preview_from_path("input", Path(image_path))
        self._update_preview_from_path("enhanced", result.enhanced_path)
        self._update_preview_from_path("transparent", result.transparent_path)
        
        # バッチ中でもプレビューを即時更新する
        self.root.update_idletasks()

    def _on_batch_done(self, payload: dict) -> None:
        self._set_processing(False)
        total = payload["total"]
        success = payload["success"]
        failed: list[tuple[str, str]] = payload["failed"]

        self._append_log(f"バッチ処理終了: 成功 {success}/{total}")
        if failed:
            for failed_path, failed_reason in failed:
                self._append_log(f"失敗: {failed_path} -> {failed_reason}")
            messagebox.showwarning(
                "完了（一部失敗）",
                f"バッチ処理が完了しました。\n成功: {success}/{total}\n失敗: {len(failed)}",
            )
        else:
            messagebox.showinfo("完了", f"バッチ処理が完了しました。\n成功: {success}/{total}")

    def _update_ai_result(self, data: dict) -> None:
        """AI識別結果パネルを更新する。"""
        fields = {"name": "神社・寺名", "date": "日付", "text": "主な文字", "mark": "印"}
        for key in fields:
            value = data.get(key, "--") or "--"
            self._ai_labels[key].configure(text=value)
        self._append_log(f"AI識別完了: {json.dumps(data, ensure_ascii=False)}")

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
