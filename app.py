from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageTk

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


def _configure_turbojpeg_path() -> None:
    configured = (
        os.getenv("PYTURBOJPEG_LIBRARY_PATH")
        or os.getenv("TURBOJPEG_LIB_PATH")
        or os.getenv("TURBOJPEG")
        or os.getenv("TURBOJPEG_LIB")
    )
    if not configured:
        return
    dll_path = Path(configured.strip().strip('"'))
    if not dll_path.exists():
        return

    normalized = str(dll_path)
    os.environ["TURBOJPEG"] = normalized
    os.environ["TURBOJPEG_LIB"] = normalized

    bin_dir = str(dll_path.parent)
    current_path = os.environ.get("PATH", "")
    path_items = current_path.split(os.pathsep) if current_path else []
    if bin_dir not in path_items:
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir


_configure_turbojpeg_path()

from processor import (
    GoshuinProcessor,
    ProcessResult,
    extract_goshuin_color_options,
)

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
_MAIN_PREVIEW_PANEL_WIDTH = 250
_MAIN_PREVIEW_PANEL_HEIGHT = 250

if hasattr(Image, "Resampling"):
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:  # pragma: no cover - Pillow backward compatibility
    _RESAMPLE_LANCZOS = Image.LANCZOS


class GoshuinScanApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GoshuinScan-OSS")
        self.root.geometry("1460x860")
        self.root.minsize(1240, 700)

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
        self._color_options: list[dict[str, Any]] = []
        self._selected_color_ids: set[int] = set()
        self._color_buttons: dict[int, tk.Button] = {}
        self._color_button_frames: dict[int, tk.Frame] = {}
        self._color_palette_container: Optional[tk.Frame] = None
        self._color_palette_hint_var = tk.StringVar(value="画像を選択すると色域候補が表示されます。")
        self._color_preview_source = None
        self._log_window: Optional[tk.Toplevel] = None
        self._log_text_widget: Optional[tk.Text] = None
        self._log_lines: list[str] = []

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        top_area = ttk.Frame(container)
        top_area.grid(row=0, column=0, sticky=tk.NSEW)
        top_area.columnconfigure(0, weight=2)
        top_area.columnconfigure(1, weight=3)
        top_area.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(top_area)
        left_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))

        right_panel = ttk.Frame(top_area)
        right_panel.grid(row=0, column=1, sticky=tk.NSEW)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=4)
        right_panel.rowconfigure(1, weight=1)

        title = ttk.Label(left_panel, text="御朱印画像処理ツール", font=("Segoe UI", 14, "bold"))
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            left_panel,
            text="処理フロー: DocAligner+UVDoc 幾何補正 -> docTR 補正 -> 背景除去",
        )
        subtitle.pack(anchor=tk.W, pady=(2, 10))

        input_frame = ttk.LabelFrame(left_panel, text="入力/出力", padding=10)
        input_frame.pack(fill=tk.X)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="画像:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Entry(input_frame, textvariable=self.image_path_var).grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Button(input_frame, text="画像を選択", command=self._select_image).grid(
            row=0, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )

        ttk.Label(input_frame, text="画像フォルダー:").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Entry(input_frame, textvariable=self.folder_path_var).grid(row=1, column=1, sticky=tk.EW, pady=4)
        ttk.Button(input_frame, text="フォルダーを選択", command=self._select_input_dir).grid(
            row=1, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )

        ttk.Label(input_frame, text="出力フォルダー:").grid(row=2, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Entry(input_frame, textvariable=self.output_dir_var).grid(row=2, column=1, sticky=tk.EW, pady=4)
        ttk.Button(input_frame, text="フォルダーを選択", command=self._select_output_dir).grid(
            row=2, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )
        ttk.Label(
            input_frame,
            text="※ フォルダー指定時は一括処理。単画像時のみ色域選択を適用。",
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))

        color_frame = ttk.LabelFrame(left_panel, text="保持色の選択", padding=8)
        color_frame.pack(fill=tk.X, pady=(10, 0))
        self._color_palette_container = tk.Frame(color_frame)
        self._color_palette_container.pack(fill=tk.X, anchor=tk.W)
        ttk.Label(color_frame, textvariable=self._color_palette_hint_var).pack(anchor=tk.W, pady=(6, 0))

        options_frame = ttk.Frame(left_panel)
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

        actions_frame = ttk.Frame(left_panel)
        actions_frame.pack(fill=tk.X, pady=(12, 0))
        self.process_button = ttk.Button(actions_frame, text="処理開始", command=self._start_process)
        self.process_button.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(actions_frame, mode="indeterminate", length=180)
        self.progress.pack(side=tk.LEFT, padx=(12, 0))
        self.log_button = ttk.Button(actions_frame, text="ログを開く", command=self._open_log_window)
        self.log_button.pack(side=tk.LEFT, padx=(12, 0))

        preview_frame = ttk.LabelFrame(right_panel, text="画像表示", padding=8)
        preview_frame.grid(row=0, column=0, sticky=tk.NW)
        preview_frame.configure(
            width=(_MAIN_PREVIEW_PANEL_WIDTH * 3) + 34,
            height=_MAIN_PREVIEW_PANEL_HEIGHT + 46,
        )
        preview_frame.grid_propagate(False)
        preview_frame.columnconfigure(0, weight=0)
        preview_frame.rowconfigure(0, weight=0)

        preview_content = ttk.Frame(preview_frame)
        preview_content.grid(row=0, column=0, sticky=tk.NW)
        preview_content.configure(
            width=(_MAIN_PREVIEW_PANEL_WIDTH * 3) + 18,
            height=_MAIN_PREVIEW_PANEL_HEIGHT + 16,
        )
        preview_content.grid_propagate(False)
        preview_content.columnconfigure(0, weight=0)
        preview_content.columnconfigure(1, weight=0)
        preview_content.columnconfigure(2, weight=0)
        preview_content.rowconfigure(0, weight=0)

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

        ai_frame = ttk.LabelFrame(right_panel, text="解析結果 (LoRA)", padding=8)
        ai_frame.grid(row=1, column=0, sticky=tk.NSEW, pady=(10, 0))
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
            val_label = ttk.Label(cell, text="--", font=("Segoe UI", 10, "bold"), wraplength=220, justify=tk.LEFT)
            val_label.pack(anchor=tk.W)
            self._ai_labels[key] = val_label

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
            self._refresh_color_options_for_image(Path(path))

    def _select_input_dir(self) -> None:
        path = filedialog.askdirectory(title="画像フォルダーを選択")
        if path:
            folder_path = Path(path)
            self.folder_path_var.set(path)
            self.image_path_var.set("")
            self._clear_color_options("フォルダー一括処理では色選択は適用されません。")

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

    @staticmethod
    def _bgr_to_hex(color_bgr: list[int]) -> str:
        b, g, r = [int(x) for x in color_bgr[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _pick_default_keep_color_ids(color_options: list[dict[str, Any]]) -> set[int]:
        black_candidates: list[tuple[float, int]] = []
        red_candidates: list[tuple[float, int]] = []

        for option in color_options:
            color_id = int(option.get("id", -1))
            bgr = option.get("bgr")
            if color_id < 0 or not isinstance(bgr, list) or len(bgr) < 3:
                continue

            b, g, r = [int(np.clip(v, 0, 255)) for v in bgr[:3]]
            ratio = float(option.get("ratio", 0.0))
            is_background = bool(option.get("is_background", False))
            if is_background:
                continue

            # 輝度(Value)と彩度(Saturation)の計算
            v_max = max(r, g, b)
            v_min = min(r, g, b)
            sat = v_max - v_min

            # 1. 金粉・背景色の除外ロジック (Gold/Yellowish Background Filter)
            # 御朱印の和紙に含まれる金粉や黄ばみを除去するためのフィルター
            # 金色は通常 R > G > B かつ B が極端に低い特性を持つ
            is_gold_noise = (r > 130 and g > 110 and b < 110)
            
            # 2. 墨書き（黒色系）の判定 (Ink/Black Candidates)
            # 明度が低ければ、多少の彩度（茶色寄り）も許容して掠れを保持する
            if v_max <= 140 and not is_gold_noise:
                # 彩度が低いほど純粋な黒に近いと判定
                black_score = (140 - v_max) * 2.0 + (80 - sat) * 0.5 + ratio * 120.0
                # 褐色残渣を防ぐため、Rが極端に高い場合はスコアを減らす
                if r > b + 30:
                    black_score -= 40
                
                if black_score > 0:
                    black_candidates.append((black_score, color_id))

            # 3. 朱印（赤色系）の判定 (Stamp/Red Candidates)
            # R成分が支配的であるかを厳密にチェック
            if r >= 70 and r > g and not is_gold_noise:
                # 赤色の純度（Redness）を計算
                redness = r - max(g, b)
                if redness > 15:
                    # 彩度と出現率を考慮したスコアリング
                    red_score = redness * 2.5 + (r * 0.4) + ratio * 100.0
                    # 背景の黄色（RとGが近い）との混同を避ける
                    if abs(r - g) < 25:
                        red_score -= 50
                        
                    if red_score > 0:
                        red_candidates.append((red_score, color_id))

        selected: set[int] = set()

        if black_candidates:
            black_candidates.sort(key=lambda x: x[0], reverse=True)
            selected.add(black_candidates[0][1])

        if red_candidates:
            red_candidates.sort(key=lambda x: x[0], reverse=True)
            for _score, color_id in red_candidates[:2]:
                selected.add(color_id)

        return selected

    def _clear_color_options(self, hint: str = "画像を選択すると色域候補が表示されます。") -> None:
        self._color_options = []
        self._selected_color_ids.clear()
        self._color_buttons.clear()
        self._color_button_frames.clear()
        self._color_palette_hint_var.set(hint)
        self._color_preview_source = None
        if self._color_palette_container is None:
            return
        for child in self._color_palette_container.winfo_children():
            child.destroy()

    def _render_color_blocks(self) -> None:
        if self._color_palette_container is None:
            return
        for child in self._color_palette_container.winfo_children():
            child.destroy()
        self._color_buttons.clear()
        self._color_button_frames.clear()

        if not self._color_options:
            return

        unselected_border = self._color_palette_container.cget("bg")
        columns = 12
        for idx, option in enumerate(self._color_options):
            color_id = int(option["id"])
            color_hex = self._bgr_to_hex(option["bgr"])
            border = tk.Frame(
                self._color_palette_container,
                bg=unselected_border,
                highlightthickness=0,
                bd=0,
            )
            border.grid(row=idx // columns, column=idx % columns, padx=3, pady=3, sticky=tk.W)
            button = tk.Button(
                border,
                width=2,
                height=1,
                bg=color_hex,
                activebackground=color_hex,
                relief=tk.FLAT,
                bd=0,
                highlightthickness=0,
                command=lambda cid=color_id: self._toggle_color_selection(cid),
            )
            button.pack(padx=2, pady=2)
            self._color_buttons[color_id] = button
            self._color_button_frames[color_id] = border
        self._update_color_block_styles()

    def _toggle_color_selection(self, color_id: int) -> None:
        if color_id in self._selected_color_ids:
            self._selected_color_ids.remove(color_id)
        else:
            self._selected_color_ids.add(color_id)
        self._update_color_block_styles()

    def _update_color_block_styles(self) -> None:
        if self._color_palette_container is None:
            return
        unselected_border = self._color_palette_container.cget("bg")
        selected_border = "#2583f5"
        for color_id, button in self._color_buttons.items():
            border = self._color_button_frames.get(color_id)
            if border is not None:
                border.configure(bg=selected_border if color_id in self._selected_color_ids else unselected_border)
            button.configure(relief=tk.FLAT, bd=0)

    def _set_input_preview_from_bgr_array(self, image_bgr: np.ndarray) -> None:
        rgb = image_bgr[:, :, ::-1]
        self._preview_images["input"] = Image.fromarray(rgb, mode="RGB")
        self._render_preview("input", force=True)

    def _refresh_color_options_for_image(self, image_path: Path) -> None:
        image = GoshuinProcessor._read_image_unicode(image_path, 1)
        if image is None:
            self._clear_color_options("色域候補の解析に失敗しました。")
            return

        self._color_preview_source = image
        options = extract_goshuin_color_options(image)
        if not options:
            self._clear_color_options("色域候補を抽出できませんでした。")
            return

        self._color_options = options
        self._selected_color_ids = self._pick_default_keep_color_ids(options)
        if self._selected_color_ids:
            self._color_palette_hint_var.set(
                "黒字と朱印に近い色を既定で選択済みです。必要に応じて変更してください。"
            )
        else:
            self._color_palette_hint_var.set("保持したい色を選択してください（選択色は最終透過で残します）。")
        self._render_color_blocks()

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

        color_options_payload: list[dict[str, Any]] = []
        selected_color_ids_payload: list[int] = []
        if not folder_text and self._color_options and self._selected_color_ids:
            color_options_payload = [dict(opt) for opt in self._color_options]
            selected_color_ids_payload = sorted(self._selected_color_ids)

        self._update_preview_from_path("input", image_paths[0])
        self._set_processing(True)
        self._append_log(f"処理開始: {len(image_paths)} 件")

        worker = threading.Thread(
            target=self._process_worker,
            args=(
                image_paths,
                output_dir,
                self.use_gpu_var.get(),
                self.use_ai_var.get(),
                color_options_payload,
                selected_color_ids_payload,
            ),
            daemon=True,
        )
        worker.start()

    def _process_worker(
        self,
        image_paths: list[Path],
        output_dir: Path,
        use_gpu: bool,
        use_ai: bool = False,
        color_options: Optional[list[dict[str, Any]]] = None,
        selected_color_ids: Optional[list[int]] = None,
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
                    result = processor.process(
                        image_path,
                        output_dir,
                        color_options=color_options if total == 1 else None,
                        selected_color_ids=selected_color_ids if total == 1 else None,
                    )
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
        panel.grid(row=0, column=column, sticky=tk.NW, padx=4)
        panel.configure(width=_MAIN_PREVIEW_PANEL_WIDTH, height=_MAIN_PREVIEW_PANEL_HEIGHT)
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=0)
        panel.rowconfigure(0, weight=0)

        preview_label = ttk.Label(panel, anchor=tk.CENTER, justify=tk.CENTER)
        preview_label.grid(row=0, column=0, sticky=tk.NW)
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

        target_width = max(_MAIN_PREVIEW_PANEL_WIDTH - 18, 1)
        target_height = max(_MAIN_PREVIEW_PANEL_HEIGHT - 38, 1)

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

    def _open_log_window(self) -> None:
        if self._log_window is not None and self._log_window.winfo_exists():
            self._log_window.deiconify()
            self._log_window.lift()
            self._log_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        self._log_window = window
        window.title("ログ")
        window.geometry("550x260")
        window.minsize(380, 180)

        frame = ttk.Frame(window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        text_widget = tk.Text(frame, wrap=tk.WORD, state=tk.NORMAL, font=("Consolas", 10))
        text_widget.grid(row=0, column=0, sticky=tk.NSEW)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        text_widget.configure(yscrollcommand=scroll.set)

        if self._log_lines:
            text_widget.insert(tk.END, "\n".join(self._log_lines) + "\n")
        text_widget.see(tk.END)
        text_widget.configure(state=tk.DISABLED)
        self._log_text_widget = text_widget

        def _on_close() -> None:
            self._close_log_window()

        window.protocol("WM_DELETE_WINDOW", _on_close)

    def _close_log_window(self) -> None:
        if self._log_window is not None and self._log_window.winfo_exists():
            self._log_window.destroy()
        self._log_window = None
        self._log_text_widget = None

    def _append_log(self, message: str) -> None:
        line = str(message)
        self._log_lines.append(line)
        if len(self._log_lines) > 3000:
            self._log_lines = self._log_lines[-2500:]

        if self._log_text_widget is None or not self._log_text_widget.winfo_exists():
            self._log_text_widget = None
            return

        self._log_text_widget.configure(state=tk.NORMAL)
        self._log_text_widget.insert(tk.END, f"{line}\n")
        self._log_text_widget.see(tk.END)
        self._log_text_widget.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    GoshuinScanApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
