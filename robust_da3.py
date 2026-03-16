#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[2]


# ANSI color helpers for terminal output
ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def info_print(msg: str) -> None:
    print(f"{ANSI_GREEN}{msg}{ANSI_RESET}")


def safe_empty_cache() -> None:
    """Aggressively free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@dataclass(frozen=True)
class ExperimentConfig:
    image_dir: Path
    exp_name: str = "demo_result_da3"
    model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    export_format: Optional[str] = None
    export_feat_layers: Tuple[int, ...] = ()
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    use_ray_pose: bool = False
    ref_view_strategy: str = "saddle_balanced"


def list_image_paths(images_dir: Path) -> List[Path]:
    """List image files inside the given directory."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    valid_suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix in valid_suffixes],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise RuntimeError(f"No image files found in {images_dir}")
    return image_paths


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_rgb_images(images: np.ndarray, out_dir: Path, prefix: str = "img") -> None:
    """Save a batch of RGB uint8 images with shape (N, H, W, 3)."""
    _ensure_dir(out_dir)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected images with shape (N, H, W, 3), got {images.shape}")
    for i, img in enumerate(images):
        img_pil = Image.fromarray(img)
        img_pil.save(out_dir / f"{prefix}_{i:04d}.png")


class RobustDA3Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = select_device()
        self.model = DepthAnything3.from_pretrained(self.config.model_name)
        self.model = self.model.to(device=self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

    def _run_inference(self, image_paths: Sequence[Path]) -> Dict[str, Any]:
        export_dir = self.out_dir
        export_format = self.config.export_format
        export_feat_layers = list(self.config.export_feat_layers)

        kwargs: Dict[str, Any] = {
            "image": [str(p) for p in image_paths],
            "process_res": self.config.process_res,
            "process_res_method": self.config.process_res_method,
            "use_ray_pose": self.config.use_ray_pose,
            "ref_view_strategy": self.config.ref_view_strategy,
        }
        if export_format:
            kwargs["export_dir"] = str(export_dir)
            kwargs["export_format"] = export_format
        if export_feat_layers:
            kwargs["export_feat_layers"] = export_feat_layers

        info_print("[INFO] Running DA3 inference...")
        with torch.inference_mode():
            prediction = self.model.inference(**kwargs)

        outputs: Dict[str, Any] = {
            "depth": getattr(prediction, "depth", None),
            "conf": getattr(prediction, "conf", None),
            "extrinsics": getattr(prediction, "extrinsics", None),
            "intrinsics": getattr(prediction, "intrinsics", None),
            "processed_images": getattr(prediction, "processed_images", None),
            "aux": getattr(prediction, "aux", None),
        }
        return outputs

    def _save_predictions(self, outputs: Dict[str, Any]) -> None:
        pred_path = self.out_dir / "predictions.npz"

        npz_data: Dict[str, Any] = {}
        for key in ("depth", "conf", "extrinsics", "intrinsics"):
            val = outputs.get(key, None)
            if isinstance(val, np.ndarray):
                npz_data[key] = val
        if npz_data:
            np.savez(pred_path, **npz_data)
            info_print(f"[INFO] Saved predictions to {pred_path}")
        else:
            info_print("[WARN] No core predictions found to save.")

        processed_images = outputs.get("processed_images", None)
        if isinstance(processed_images, np.ndarray):
            try:
                _save_rgb_images(processed_images, self.out_dir / "processed_images")
                info_print("[INFO] Saved processed input images.")
            except Exception as exc:
                print(f"[WARN] Failed to save processed images: {exc}")

        aux = outputs.get("aux", None)
        if isinstance(aux, dict):
            feat_dir = self.out_dir / "features"
            _ensure_dir(feat_dir)
            feat_saved = 0
            for layer_idx in self.config.export_feat_layers:
                key = f"feat_layer_{layer_idx}"
                if key in aux and isinstance(aux[key], np.ndarray):
                    np.save(feat_dir / f"{key}.npy", aux[key])
                    feat_saved += 1
            if feat_saved > 0:
                info_print(f"[INFO] Saved {feat_saved} feature layers to {feat_dir}")
            else:
                info_print("[INFO] No feature layers found in aux outputs.")

        metadata = {
            "config": asdict(self.config),
            "shapes": {
                "depth": getattr(outputs.get("depth", None), "shape", None),
                "conf": getattr(outputs.get("conf", None), "shape", None),
                "extrinsics": getattr(outputs.get("extrinsics", None), "shape", None),
                "intrinsics": getattr(outputs.get("intrinsics", None), "shape", None),
                "processed_images": getattr(outputs.get("processed_images", None), "shape", None),
            },
        }
        with open(self.out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    def run_demo(self) -> None:
        self.out_dir = Path(self.config.exp_name)
        _ensure_dir(self.out_dir)

        image_paths = list_image_paths(self.config.image_dir)
        info_print(f"[INFO] Found {len(image_paths)} images in {self.config.image_dir}")

        outputs = self._run_inference(image_paths)
        self._save_predictions(outputs)
        safe_empty_cache()


def _parse_layers(value: str) -> Tuple[int, ...]:
    value = value.strip()
    if not value:
        return ()
    parts = [p.strip() for p in value.split(",") if p.strip()]
    layers: List[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid layer index: {p}")
        layers.append(int(p))
    return tuple(layers)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 on a directory of images.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to process.",
    )
    parser.add_argument("--exp-name", type=str, default="demo_result_da3", help="Experiment name for output directory.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="",
        help="Optional DA3 export format string (e.g., 'mini_npz', 'npz', 'ply', 'feat_vis').",
    )
    parser.add_argument(
        "--export-feat-layers",
        type=str,
        default="",
        help="Comma-separated list of feature layers to export (e.g., '0,5,10').",
    )
    parser.add_argument("--process-res", type=int, default=504, help="Processing resolution (short side).")
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        help="Resolution processing method (e.g., upper_bound_resize, lower_bound_resize, pad).",
    )
    parser.add_argument("--use-ray-pose", action="store_true", help="Enable ray-based pose estimation.")
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        help="Reference view selection strategy for multi-view inputs.",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        image_dir=args.image_dir,
        exp_name=args.exp_name,
        model_name=args.model_name,
        export_format=args.export_format or None,
        export_feat_layers=_parse_layers(args.export_feat_layers),
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
    )


def main() -> None:
    config = parse_args()
    experiment = RobustDA3Experiment(config)
    experiment.run_demo()


if __name__ == "__main__":
    main()
