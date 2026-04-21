from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

try:
    from paddleocr import TextImageUnwarping

    _PADDLEOCR_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - runtime dependency check
    TextImageUnwarping = None
    _PADDLEOCR_IMPORT_ERROR = exc

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    _DOCTR_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - runtime dependency check
    DocumentFile = None
    ocr_predictor = None
    _DOCTR_IMPORT_ERROR = exc


@dataclass
class ProcessResult:
    enhanced_path: Path
    transparent_path: Path


def _lab_to_bgr_tuple(lab_color: np.ndarray) -> tuple[int, int, int]:
    # cv2.cvtColor with COLOR_Lab2BGR expects OpenCV-Lab encoded range for uint8:
    # L in [0,255], a in [0,255], b in [0,255].
    lab_encoded = np.clip(np.rint(lab_color), 0, 255).astype(np.uint8)
    lab_pixel = np.array([[lab_encoded]], dtype=np.uint8)
    bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)[0, 0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def extract_goshuin_color_options(
    image: np.ndarray,
    max_colors: int = 8,
    min_ratio: float = 0.012,
) -> list[dict[str, Any]]:
    if image is None or image.size == 0:
        return []
    if image.ndim != 3 or image.shape[2] < 3:
        return []

    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side > 720:
        scale = 720.0 / float(max_side)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    num_pixels = lab.shape[0]
    if num_pixels < 64:
        return []

    dynamic_k = int(np.clip(np.sqrt(num_pixels) / 60.0, 3, max_colors))
    k = int(max(3, min(max_colors, dynamic_k)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.4)
    _compactness, labels, centers = cv2.kmeans(
        lab,
        k,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(-1)
    options: list[dict[str, Any]] = []

    for cluster_idx in range(k):
        cluster_mask = labels == cluster_idx
        ratio = float(cluster_mask.mean())
        if ratio < min_ratio:
            continue

        cluster_points = lab[cluster_mask]
        center = centers[cluster_idx]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        radius = float(np.clip(np.percentile(distances, 88) * 1.35 + 6.0, 12.0, 60.0))

        l_val, a_val, b_val = float(center[0]), float(center[1]), float(center[2])
        chroma = float(np.hypot(a_val - 128.0, b_val - 128.0))
        is_background = bool((l_val >= 176 and chroma <= 14 and ratio >= 0.05) or (l_val >= 190 and chroma <= 20))

        options.append(
            {
                "ratio": ratio,
                "lab": [float(center[0]), float(center[1]), float(center[2])],
                "radius": radius,
                "bgr": list(_lab_to_bgr_tuple(center)),
                "is_background": is_background,
            }
        )

    # Merge nearby color options so users see fewer but broader color blocks.
    merged: list[dict[str, Any]] = []
    merge_distance = 18.0
    for option in sorted(options, key=lambda item: item["ratio"], reverse=True):
        center = np.array(option["lab"], dtype=np.float32)
        merged_index = -1
        for idx, base in enumerate(merged):
            base_center = np.array(base["lab"], dtype=np.float32)
            if np.linalg.norm(center - base_center) <= merge_distance:
                merged_index = idx
                break

        if merged_index < 0:
            merged.append(dict(option))
            continue

        base = merged[merged_index]
        total_ratio = float(base["ratio"]) + float(option["ratio"])
        if total_ratio <= 0:
            continue
        blended_center = (
            np.array(base["lab"], dtype=np.float32) * float(base["ratio"])
            + np.array(option["lab"], dtype=np.float32) * float(option["ratio"])
        ) / total_ratio

        base["ratio"] = total_ratio
        base["lab"] = [float(blended_center[0]), float(blended_center[1]), float(blended_center[2])]
        base["radius"] = float(np.clip(max(float(base["radius"]), float(option["radius"])) * 1.08, 12.0, 66.0))
        base["bgr"] = list(_lab_to_bgr_tuple(np.array(base["lab"], dtype=np.float32)))

        l_val = float(base["lab"][0])
        a_val = float(base["lab"][1])
        b_val = float(base["lab"][2])
        chroma = float(np.hypot(a_val - 128.0, b_val - 128.0))
        base["is_background"] = bool((l_val >= 176 and chroma <= 14 and total_ratio >= 0.05) or (l_val >= 190 and chroma <= 20))

    options = [item for item in merged if float(item["ratio"]) >= min_ratio * 0.8]
    options.sort(key=lambda item: item["ratio"], reverse=True)
    options = options[:max_colors]
    for idx, item in enumerate(options):
        item["id"] = idx
    return options


def build_selected_color_mask(
    image: np.ndarray,
    color_options: list[dict[str, Any]],
    selected_color_ids: list[int],
) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        return np.zeros(image.shape[:2], dtype=np.float32)

    selected_ids = {int(i) for i in selected_color_ids}
    if not selected_ids:
        return np.zeros(image.shape[:2], dtype=np.float32)

    valid_options = []
    for option in color_options:
        center_vals = option.get("lab")
        if isinstance(center_vals, list) and len(center_vals) == 3:
            valid_options.append(option)

    if not valid_options:
        return np.zeros(image.shape[:2], dtype=np.float32)

    # Ensure deterministic mapping between option id and center index.
    valid_options.sort(key=lambda o: int(o.get("id", 0)))
    option_ids = [int(o.get("id", -1)) for o in valid_options]
    selected_indices = [idx for idx, oid in enumerate(option_ids) if oid in selected_ids]

    if not selected_indices:
        return np.zeros(image.shape[:2], dtype=np.float32)
    if len(selected_indices) == len(valid_options):
        return np.ones(image.shape[:2], dtype=np.float32)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    centers = np.array([o["lab"] for o in valid_options], dtype=np.float32)

    # Full partition: every pixel belongs to nearest color option.
    diff = lab_image[:, :, None, :] - centers[None, None, :, :]
    dist = np.linalg.norm(diff, axis=3)
    nearest_idx = np.argmin(dist, axis=2)
    hard = np.isin(nearest_idx, selected_indices).astype(np.float32)

    # Hard partition: each pixel is assigned to one nearest color option.
    # This guarantees full coverage with no residual semi-transparent band.
    return hard


def GoshuinSensoryExtractor(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("GoshuinSensoryExtractor: image is empty.")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("GoshuinSensoryExtractor: expected BGR image with 3 channels.")

    bgr = image[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    l_ch, a_ch, b_ch = cv2.split(lab)

    a_i16 = a_ch.astype(np.int16)
    b_i16 = b_ch.astype(np.int16)

    black_mask = (v <= 86) & (l_ch <= 112)

    red_mask = (cv2.inRange(hsv, (0, 42, 20), (15, 255, 255)) > 0) | (
        cv2.inRange(hsv, (160, 45, 20), (180, 255, 255)) > 0
    )

    colorful_ink_mask = (
        (s >= 44)
        & (v >= 30)
        & (((h >= 46) & (h <= 159)) | ((h >= 161) & (h <= 175)))
    )

    paper_mask = (
        (s <= 34)
        & (v >= 155)
        & (np.abs(a_i16 - 128) <= 12)
        & (np.abs(b_i16 - 128) <= 14)
    )

    l_blur = cv2.GaussianBlur(l_ch, (0, 0), sigmaX=1.2)
    lap = cv2.Laplacian(l_blur, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0.0, 1.0, cv2.NORM_MINMAX)
    edge_mask = cv2.Canny(l_blur, 60, 140) > 0
    texture_mask = (lap_norm > 0.09) | edge_mask
    texture_strong_mask = (lap_norm > 0.13) | edge_mask

    anchor_seed = black_mask | red_mask | colorful_ink_mask
    anchor_u8 = (anchor_seed.astype(np.uint8) * 255)
    anchor_near = cv2.dilate(
        anchor_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    ) > 0

    initial_mask = (black_mask | red_mask | colorful_ink_mask) & (~paper_mask | texture_mask)

    mask_u8 = initial_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    filtered = np.zeros_like(mask_u8)
    min_area = max(32, int(image.shape[0] * image.shape[1] * 0.00003))
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label_id] = 255

    alpha = filtered.astype(np.float32) / 255.0
    fine_detail = (texture_strong_mask & anchor_near).astype(np.float32) * 0.28
    alpha = np.maximum(alpha, fine_detail)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8)
    return np.clip(alpha, 0.0, 1.0)


class GoshuinProcessor:
    def __init__(
        self,
        device: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self._log_callback = log_callback or (lambda _msg: None)

        self._doctr_predictor = None
        self._uvdoc_model = None
        self._rmbg_model = None
        self._rmbg_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ]
        )

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        source_image_override: Optional[np.ndarray] = None,
        color_options: Optional[list[dict[str, Any]]] = None,
        selected_color_ids: Optional[list[int]] = None,
    ) -> ProcessResult:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if source_image_override is not None:
            source = source_image_override.copy()
            self._log(f"入力画像を読み込み: {image_path} (色域選択済み入力を使用)")
        else:
            self._log(f"入力画像を読み込み: {image_path}")
            source = self._read_image_unicode(image_path, cv2.IMREAD_COLOR)
            if source is None:
                raise ValueError(f"画像を読み込めません: {image_path}")

        self._log("ステップ 1/3: UVDoc 幾何補正")
        try:
            source = self._uvdoc_geometric_correction(source)
            self._log("UVDoc による幾何補正が完了しました。")
        except Exception as exc:
            details = self._collect_exception_text(exc)
            hint = self._build_uvdoc_hint(details)
            if hint:
                details = f"{details}\n{hint}"
            self._log(
                "UVDoc 補正に失敗したため、従来の RMBG パース補正にフォールバックします。\n"
                f"詳細: {details}"
            )
            initial_mask = self._predict_foreground_mask(source)
            source = self._perspective_correction(source, initial_mask)

        self._log("ステップ 2/3: docTR ドキュメント補正")
        enhanced = self._doctr_document_enhancement(source)
        enhanced_path = output_dir / f"{image_path.stem}_enhanced_doctr.png"
        self._write_image_unicode(enhanced_path, enhanced)
        self._log(f"保存済み: {enhanced_path}")

        self._log("ステップ 3/3: RMBG-2.0 背景除去 (黒墨/朱印抽出)")
        transparent = self._build_transparent_ink_stamp(
            source,
            color_options=color_options,
            selected_color_ids=selected_color_ids,
        )
        transparent_path = output_dir / f"{image_path.stem}_ink_stamp_transparent.png"
        self._write_image_unicode(transparent_path, transparent)
        self._log(f"保存済み: {transparent_path}")

        return ProcessResult(
            enhanced_path=enhanced_path,
            transparent_path=transparent_path,
        )

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _log(self, message: str) -> None:
        self._log_callback(message)

    def _load_uvdoc_model(self):
        if self._uvdoc_model is not None:
            return self._uvdoc_model
        if TextImageUnwarping is None:
            import_error_text = str(_PADDLEOCR_IMPORT_ERROR) if _PADDLEOCR_IMPORT_ERROR else "不明な import エラー"
            raise RuntimeError(
                "UVDoc (PaddleOCR) が利用できません。\n"
                f"原因: {import_error_text}\n"
                "対処: `pip install paddleocr` と `pip install paddlepaddle`（または `paddlepaddle-gpu`）を実行してください。"
            ) from _PADDLEOCR_IMPORT_ERROR

        model = None
        model_options = [
            {"model_name": "UVDoc", "device": "gpu" if self.device == "cuda" else "cpu"},
            {"model_name": "UVDoc"},
        ]
        last_exc: Optional[Exception] = None
        for kwargs in model_options:
            try:
                model = TextImageUnwarping(**kwargs)
                break
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                # Some versions may reject `device`; retry with minimal args.
                last_exc = exc
                if "device" in kwargs:
                    continue
                raise

        if model is None:
            details = self._collect_exception_text(last_exc) if last_exc else "不明な初期化エラー"
            raise RuntimeError(f"UVDoc モデルの初期化に失敗しました。詳細: {details}") from last_exc

        self._uvdoc_model = model
        return model

    def _uvdoc_geometric_correction(self, image: np.ndarray) -> np.ndarray:
        model = self._load_uvdoc_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "uvdoc_input.png"
            output_dir = temp_root / "uvdoc_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_image_unicode(input_path, image)

            try:
                results = model.predict(str(input_path), batch_size=1)
            except TypeError:
                results = model.predict(str(input_path))

            result_item = None
            for item in results:
                result_item = item
                break

            if result_item is None:
                raise RuntimeError("UVDoc から有効な推論結果が得られませんでした。")

            if hasattr(result_item, "save_to_img"):
                try:
                    result_item.save_to_img(save_path=str(output_dir))
                    saved_path = self._find_first_image_file(output_dir)
                    if saved_path is not None:
                        saved = self._read_image_unicode(saved_path, cv2.IMREAD_COLOR)
                        if saved is not None:
                            return saved
                except Exception:
                    pass

            extracted = self._extract_image_from_uvdoc_result(result_item)
            if extracted is not None:
                return extracted

            raise RuntimeError("UVDoc の出力画像を復元できませんでした。")

    @staticmethod
    def _find_first_image_file(root: Path) -> Optional[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                return path
        return None

    @staticmethod
    def _normalize_image_array(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None

        array: Optional[np.ndarray]
        if isinstance(value, np.ndarray):
            array = value
        elif hasattr(value, "numpy"):
            try:
                array = value.numpy()
            except Exception:
                return None
        else:
            return None

        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        elif array.ndim != 3:
            return None

        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            max_val = float(array.max()) if array.size else 0.0
            if max_val <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return array

    @classmethod
    def _extract_image_from_uvdoc_result(cls, result_item: Any) -> Optional[np.ndarray]:
        candidates: list[Any] = [result_item]
        if hasattr(result_item, "res"):
            candidates.append(getattr(result_item, "res"))

        for candidate in candidates:
            normalized = cls._normalize_image_array(candidate)
            if normalized is not None:
                return normalized

            if isinstance(candidate, dict):
                for key in ("doctr_img", "output_img", "img", "image", "res", "result"):
                    if key not in candidate:
                        continue
                    normalized = cls._normalize_image_array(candidate.get(key))
                    if normalized is not None:
                        return normalized

        return None

    def _load_doctr_predictor(self):
        if self._doctr_predictor is not None:
            return self._doctr_predictor
        if ocr_predictor is None or DocumentFile is None:
            raise RuntimeError(
                "docTR が利用できません。requirements.txt の依存関係をインストールして再実行してください。"
            ) from _DOCTR_IMPORT_ERROR

        predictor = None
        predictor_options = [
            {
                "pretrained": True,
                "detect_orientation": True,
                "straighten_pages": True,
            },
            {
                "pretrained": True,
                "detect_orientation": True,
            },
            {
                "pretrained": True,
            },
        ]
        for kwargs in predictor_options:
            try:
                predictor = ocr_predictor(**kwargs)
                break
            except TypeError:
                continue

        if predictor is None:
            raise RuntimeError("現在の docTR バージョンは ocr_predictor の初期化パラメータと互換性がありません。")

        # docTR predictor is not always a plain nn.Module across versions.
        # This best-effort move keeps compatibility with multiple releases.
        try:
            predictor = predictor.to(torch.device(self.device))
        except Exception:
            pass

        self._doctr_predictor = predictor
        return predictor

    @staticmethod
    def _read_image_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        try:
            buffer = np.fromfile(str(path), dtype=np.uint8)
            if buffer.size == 0:
                return None
            image = cv2.imdecode(buffer, flags)
            return image
        except Exception:
            return None

    @staticmethod
    def _write_image_unicode(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower() or ".png"
        ext = suffix if suffix.startswith(".") else f".{suffix}"
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            raise ValueError(f"画像をエンコードできません: {path}")
        encoded.tofile(str(path))

    def _load_rmbg_model(self):
        if self._rmbg_model is not None:
            return self._rmbg_model

        model_id = os.getenv("RMBG_MODEL_ID", "briaai/RMBG-2.0").strip() or "briaai/RMBG-2.0"
        token = self._resolve_hf_token()

        kwargs = {"trust_remote_code": True}
        if token:
            kwargs["token"] = token

        try:
            model = AutoModelForImageSegmentation.from_pretrained(model_id, **kwargs)
        except TypeError:
            # Backward compatibility for old Transformers versions.
            if "token" in kwargs:
                legacy_kwargs = dict(kwargs)
                legacy_kwargs["use_auth_token"] = legacy_kwargs.pop("token")
                model = AutoModelForImageSegmentation.from_pretrained(model_id, **legacy_kwargs)
            else:
                raise
        except HfHubHTTPError as exc:
            raise RuntimeError(self._build_rmbg_hf_error_message(model_id, token, exc)) from exc
        except Exception as exc:
            message = str(exc)
            lower_message = message.lower()
            if (
                "gated repo" in lower_message
                or "401" in lower_message
                or "403" in lower_message
                or "couldn't connect" in lower_message
                or "offline mode" in lower_message
            ):
                raise RuntimeError(self._build_rmbg_hf_error_message(model_id, token, exc)) from exc
            raise

        model.to(torch.device(self.device))
        model.eval()
        self._rmbg_model = model
        return model

    @staticmethod
    def _resolve_hf_token() -> Optional[str]:
        env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if env_token:
            return env_token
        stored_token = get_token()
        return stored_token if stored_token else None

    @staticmethod
    def _build_rmbg_hf_error_message(model_id: str, token: Optional[str], exc: Exception) -> str:
        details = GoshuinProcessor._collect_exception_text(exc)
        lowered = details.lower()

        # Additional probe to turn generic "can't connect" exceptions into actionable causes.
        probe_details = ""
        if "couldn't connect" in lowered or "offline mode" in lowered:
            try:
                hf_hub_download(repo_id=model_id, filename="config.json", token=token)
            except Exception as probe_exc:
                probe_details = GoshuinProcessor._collect_exception_text(probe_exc)
                details = probe_details
                lowered = probe_details.lower()

        fallback_hint = ""
        if model_id == "briaai/RMBG-2.0":
            fallback_hint = (
                "\n暫定対応: 環境変数 `RMBG_MODEL_ID=briaai/RMBG-1.4` を設定すると gated 制限を回避できます。"
            )

        if "public gated repositories" in lowered:
            return (
                "現在の Hugging Face token は有効ですが、public gated repositories の読み取り権限が不足しています。\n"
                "https://huggingface.co/settings/tokens で fine-grained token を編集し、"
                "public gated repos への read 権限を有効化するか、通常の Read token を再作成して、"
                "`hf auth login` で再ログインしてください。\n"
                f"モデル: {model_id}\n元エラー: {details}{fallback_hint}"
            )

        if "not in the authorized list" in lowered or "gated repo" in lowered:
            return (
                f"{model_id} は gated model で、現在のアカウントは許可リストに含まれていません。\n"
                f"https://huggingface.co/{model_id} でアクセス申請し承認後、`hf auth login` を実行してください。\n"
                f"元エラー: {details}{fallback_hint}"
            )

        if "401" in lowered or "invalid token" in lowered:
            return (
                "Hugging Face 認証に失敗しました。token が有効で、`hf auth login` で現在の環境に保存されているか確認してください。\n"
                f"モデル: {model_id}\n元エラー: {details}{fallback_hint}"
            )

        if "couldn't connect" in lowered or "failed to establish a new connection" in lowered:
            return (
                "Hugging Face に接続できません。ネットワーク、プロキシ、ファイアウォール設定を確認し、"
                "必要に応じて `HTTPS_PROXY` を設定して再試行してください。\n"
                f"モデル: {model_id}\n元エラー: {details}{fallback_hint}"
            )

        return (
            f"モデル {model_id} の読み込みに失敗しました。\n"
            "アカウント権限、token 権限、ネットワーク設定を確認してください。\n"
            f"元エラー: {details}{fallback_hint}"
        )

    @staticmethod
    def _collect_exception_text(exc: Exception) -> str:
        if exc is None:
            return ""
        seen: set[int] = set()
        parts: list[str] = []
        stack: list[BaseException] = [exc]

        while stack:
            current = stack.pop(0)
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)

            text = str(current).strip()
            if text:
                parts.append(text)

            if current.__cause__ is not None:
                stack.append(current.__cause__)
            if current.__context__ is not None:
                stack.append(current.__context__)

        return "\n".join(parts)

    @staticmethod
    def _build_uvdoc_hint(details: str) -> str:
        lowered = details.lower()

        if "no module named 'paddleocr'" in lowered:
            return "ヒント: `pip install paddleocr` を実行してから再試行してください。"
        if "no module named 'paddle'" in lowered or "import paddle" in lowered:
            return (
                "ヒント: PaddlePaddle が未導入です。"
                "`pip install paddlepaddle`（CPU）または `pip install paddlepaddle-gpu`（GPU）を実行してください。"
            )
        if "textimageunwarping" in lowered and "cannot import name" in lowered:
            return "ヒント: PaddleOCR のバージョンが古い可能性があります。`pip install -U paddleocr` を実行してください。"
        if "couldn't connect" in lowered or "failed to establish a new connection" in lowered:
            return (
                "ヒント: モデルダウンロード時のネットワーク接続に失敗しています。"
                "必要なら `PADDLE_PDX_MODEL_SOURCE=BOS` を設定して再試行してください。"
            )
        if "permission denied" in lowered or "winerror 5" in lowered:
            return "ヒント: 権限不足の可能性があります。管理者権限または書き込み可能な作業ディレクトリで実行してください。"
        return ""

    def _doctr_document_enhancement(self, image: np.ndarray) -> np.ndarray:
        oriented = image
        try:
            predictor = self._load_doctr_predictor()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "input.png"
                cv2.imwrite(str(temp_path), image)
                doc = DocumentFile.from_images(str(temp_path))
                result = predictor(doc)

            angle = self._extract_orientation_angle(result)
            if abs(angle) >= 1.0:
                self._log(f"docTR がページ角度 {angle:.1f}° を検出。回転補正を実行します。")
                oriented = self._rotate_bound(image, -angle)
            else:
                self._log("docTR のページ角度は 0° 付近のため、回転補正をスキップします。")
        except Exception as exc:
            self._log(f"docTR 補正に失敗したため、クラシック補正にフォールバックします: {exc}")

        return self._classical_document_enhancement(oriented)

    @staticmethod
    def _extract_orientation_angle(result) -> float:
        try:
            exported = result.export()
        except Exception:
            return 0.0

        pages = exported.get("pages", [])
        if not pages:
            return 0.0

        orientation = pages[0].get("orientation")
        if isinstance(orientation, dict):
            value = orientation.get("value", 0.0)
        else:
            value = orientation

        if isinstance(value, (int, float)):
            if value in (0, 1, 2, 3):
                return float(value) * 90.0
            return float(value)
        return 0.0

    @staticmethod
    def _classical_document_enhancement(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        return sharpened

    def _build_transparent_ink_stamp(
        self,
        image: np.ndarray,
        color_options: Optional[list[dict[str, Any]]] = None,
        selected_color_ids: Optional[list[int]] = None,
    ) -> np.ndarray:
        foreground_mask = self._predict_foreground_mask(image)

        ink_stamp_mask = self._extract_ink_stamp_mask(image)
        alpha = np.clip(foreground_mask * ink_stamp_mask, 0.0, 1.0)
        alpha = np.where(alpha > 0.09, alpha, 0.0)

        if color_options and selected_color_ids:
            selected_mask = build_selected_color_mask(image, color_options, selected_color_ids)
            if float(selected_mask.max()) > 0.02:
                # Keep selected colors opaque in final transparent output.
                alpha = np.maximum(alpha, np.clip(selected_mask, 0.0, 1.0))
            else:
                self._log("選択した保持色マスクが空のため、既定抽出のみを使用します。")

        alpha_u8 = (alpha * 255).astype(np.uint8)

        output = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        output[:, :, 3] = alpha_u8
        output[alpha_u8 == 0, :3] = 0
        return output

    def _predict_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        model = self._load_rmbg_model()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        input_tensor = self._rmbg_transform(pil_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            raw_output = model(input_tensor)

        tensor = self._find_tensor(raw_output)
        if tensor is None:
            raise RuntimeError("RMBG-2.0 の出力構造を解析できません。")

        tensor = torch.sigmoid(tensor)
        while tensor.ndim > 2:
            tensor = tensor[0]
        mask = tensor.detach().float().cpu().numpy()

        height, width = image.shape[:2]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)
        mask = np.clip(mask, 0.0, 1.0)
        return mask

    @staticmethod
    def _find_tensor(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in reversed(output):
                tensor = GoshuinProcessor._find_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(output, dict):
            preferred_keys = ("preds", "logits", "mask", "out", "alpha")
            for key in preferred_keys:
                if key in output:
                    tensor = GoshuinProcessor._find_tensor(output[key])
                    if tensor is not None:
                        return tensor
            for value in output.values():
                tensor = GoshuinProcessor._find_tensor(value)
                if tensor is not None:
                    return tensor
            return None

        for attr in ("preds", "logits", "out", "mask"):
            if hasattr(output, attr):
                tensor = GoshuinProcessor._find_tensor(getattr(output, attr))
                if tensor is not None:
                    return tensor
        return None

    def _perspective_correction(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask * 255).astype(np.uint8)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
            
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < (mask.shape[0] * mask.shape[1] * 0.1):
            return image
            
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            pts = np.int0(box)
            
        rect_pts = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect_pts[0] = pts[np.argmin(s)]
        rect_pts[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect_pts[1] = pts[np.argmin(diff)]
        rect_pts[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect_pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth < 10 or maxHeight < 10:
            return image
            
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect_pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        self._log("ページ境界を検出し、視角（パース）補正を適用しました。")
        return warped

    @staticmethod
    def _extract_ink_stamp_mask(image: np.ndarray) -> np.ndarray:
        return GoshuinSensoryExtractor(image)

    @staticmethod
    def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_value = abs(matrix[0, 0])
        sin_value = abs(matrix[0, 1])
        new_width = int((height * sin_value) + (width * cos_value))
        new_height = int((height * cos_value) + (width * sin_value))

        matrix[0, 2] += (new_width / 2.0) - center[0]
        matrix[1, 2] += (new_height / 2.0) - center[1]

        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
