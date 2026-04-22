from __future__ import annotations

import ctypes.util
import os
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    try:
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
    except Exception:
        # sitecustomize must stay fail-safe.
        return


def _configure_turbojpeg() -> None:
    original_find_library = ctypes.util.find_library

    configured = (
        os.getenv("PYTURBOJPEG_LIBRARY_PATH")
        or os.getenv("TURBOJPEG_LIB_PATH")
        or os.getenv("TURBOJPEG")
        or os.getenv("TURBOJPEG_LIB")
    )
    resolved_path: str | None = None
    if configured:
        dll_path = Path(configured.strip().strip('"'))
        if dll_path.exists():
            resolved_path = str(dll_path)
            os.environ["TURBOJPEG"] = resolved_path
            os.environ["TURBOJPEG_LIB"] = resolved_path

            bin_dir = str(dll_path.parent)
            current_path = os.environ.get("PATH", "")
            path_items = current_path.split(os.pathsep) if current_path else []
            if bin_dir not in path_items:
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir

            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass

    if resolved_path is None:
        discovered = original_find_library("libturbojpeg")
        if discovered:
            resolved_path = discovered

    def _patched_find_library(name: str):
        lowered = (name or "").lower()
        if lowered in {"turbojpeg", "libturbojpeg"}:
            if resolved_path:
                return resolved_path
            fallback = original_find_library("libturbojpeg")
            if fallback:
                return fallback
        return original_find_library(name)

    ctypes.util.find_library = _patched_find_library


for _dotenv in (
    Path.cwd() / ".env",
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
):
    _load_dotenv(_dotenv)
_configure_turbojpeg()
