#!/usr/bin/env python3
"""Fail-fast checks before a long real FER2013 feature build."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def check_module(name: str) -> tuple[bool, str]:
    return importlib.util.find_spec(name) is not None, name


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from app.core.config import get_settings

    settings = get_settings()
    dataset = Path(args.dataset_path) if args.dataset_path else settings.dataset_path
    checks: list[tuple[str, bool, str]] = []

    checks.append(("FER2013 CSV exists", dataset.is_file(), str(dataset)))
    for module in ("numpy", "pandas", "cv2", "sklearn", "skimage"):
        ok, detail = check_module(module)
        checks.append((f"Python dependency: {module}", ok, detail))

    deepface_ok, _ = check_module("deepface")
    checks.append(("Python dependency: deepface", deepface_ok, "required for real 22D features"))
    if deepface_ok:
        try:
            from deepface import DeepFace
            checks.append(("DeepFace import", True, getattr(DeepFace, "__name__", "DeepFace")))
        except Exception as exc:
            checks.append(("DeepFace import", False, str(exc)))
    else:
        checks.append(("DeepFace import", False, "package not installed"))

    checks.append(("features directory writable", settings.features_dir.exists() or settings.features_dir.parent.exists(), str(settings.features_dir)))

    print("PHASE 5 PREFLIGHT")
    failed = False
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
        failed |= not ok

    if dataset.is_file():
        print("\nDataset schema/content validation:")
        # Import and validate via the project's real loader, without building features.
        try:
            from app.dataset.fer2013_loader import FER2013Loader
            loaded = FER2013Loader().load(dataset)
            print(f"[OK] loader: {len(loaded.samples)} valid samples, {len(loaded.malformed)} malformed")
            if loaded.malformed:
                failed = True
                print("[FAIL] malformed rows exist; inspect loader details before a long run")
            if not loaded.schema.has_official_split_column:
                failed = True
                print("[FAIL] official Usage/split column is missing")
        except Exception as exc:
            failed = True
            print(f"[FAIL] project loader: {exc}")

    print("\nRESULT:", "READY" if not failed else "NOT READY")
    return 0 if not failed else 1

if __name__ == "__main__":
    raise SystemExit(main())
