from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

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


class GoshuinProcessor:
    def __init__(
        self,
        device: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self._log_callback = log_callback or (lambda _msg: None)

        self._doctr_predictor = None
        self._rmbg_model = None
        self._rmbg_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ]
        )

    def process(self, image_path: str | Path, output_dir: str | Path) -> ProcessResult:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"入力画像を読み込み: {image_path}")
        source = self._read_image_unicode(image_path, cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        self._log("ステップ 1/3: 視角の補正と切り抜き (RMBG)")
        initial_mask = self._predict_foreground_mask(source)
        source = self._perspective_correction(source, initial_mask)

        self._log("ステップ 2/3: docTR ドキュメント補正")
        enhanced = self._doctr_document_enhancement(source)
        enhanced_path = output_dir / f"{image_path.stem}_enhanced_doctr.png"
        self._write_image_unicode(enhanced_path, enhanced)
        self._log(f"保存済み: {enhanced_path}")

        self._log("ステップ 3/3: RMBG-2.0 背景除去 (黒墨/朱印抽出)")
        transparent = self._build_transparent_ink_stamp(source)
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

    def _build_transparent_ink_stamp(self, image: np.ndarray) -> np.ndarray:
        foreground_mask = self._predict_foreground_mask(image)
        ink_stamp_mask = self._extract_ink_stamp_mask(image)

        alpha = np.clip(foreground_mask * ink_stamp_mask, 0.0, 1.0)
        alpha = np.where(alpha > 0.08, alpha, 0.0)
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
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 赤色（朱印）の抽出
        red_1 = cv2.inRange(hsv, (0, 50, 35), (15, 255, 255))
        red_2 = cv2.inRange(hsv, (165, 50, 35), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_1, red_2)

        # 黒色（墨）の抽出
        # 1. 太い筆文字や完全に暗い領域のためのグローバル閾値
        black_mask_dark = cv2.inRange(gray, 0, 100)
        
        # 2. 細い線（絵など）やかすれた墨のための適応的閾値
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        black_mask_adaptive = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
        
        black_mask = cv2.bitwise_or(black_mask_dark, black_mask_adaptive)

        combined = cv2.bitwise_or(red_mask, black_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 小さなノイズを除去し、途切れた線を繋ぐ
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        return combined.astype(np.float32) / 255.0

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
