#!/usr/bin/env python3
"""
run_vggt.py — Run VGGT inference on a folder of images and save the
reconstructed 3D scene as a coloured PLY point cloud.

Two point sources are supported (controlled via --use_point_map):
  • depth branch  (default): depth maps are unprojected using the decoded
    camera extrinsics / intrinsics  →  more geometry detail.
  • point-map branch (--use_point_map): world_points predicted directly by
    the point head  →  slightly faster, no camera math needed.

Usage:
    python run_vggt.py --image_folder path/to/images
    python run_vggt.py --image_folder path/to/images --output scene.ply \
                       --conf_threshold 30 --use_point_map
"""

import argparse
import glob
import os
import struct

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary PLY point cloud (XYZ + RGB)."""
    assert points.shape[0] == colors.shape[0], \
        "points and colors must have the same number of rows"
    num_pts = points.shape[0]
    print(f"[INFO] Writing {num_pts:,} points → {path}")

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_pts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        pts_f32 = points.astype(np.float32)
        col_u8 = colors.astype(np.uint8)
        for i in range(num_pts):
            f.write(struct.pack("<fff", *pts_f32[i]))
            f.write(struct.pack("BBB", *col_u8[i]))

    print(f"[INFO] PLY saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VGGT on a folder of images and export a 3D PLY point cloud."
    )
    parser.add_argument(
        "--image_folder", type=str, required=True,
        help="Directory containing the input images.",
    )
    parser.add_argument(
        "--output", type=str, default="output.ply",
        help="Output PLY file path (default: output.ply).",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=50.0,
        help="Discard the lowest N%% of points by confidence score (0–100, default 50).",
    )
    parser.add_argument(
        "--use_point_map", action="store_true",
        help="Use the direct point-map branch instead of unprojecting depth maps.",
    )
    args = parser.parse_args()

    # ── Device / dtype ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"[INFO] device={device}, dtype={dtype}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("[INFO] Loading VGGT-1B …")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    model.requires_grad_(False)

    # ── Discover images ───────────────────────────────────────────────────────
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_names = []
    for ext in image_exts:
        image_names.extend(glob.glob(os.path.join(args.image_folder, ext)))
    image_names = sorted(image_names)
    if not image_names:
        raise RuntimeError(f"No images found in {args.image_folder}")
    print(f"[INFO] Found {len(image_names)} images")

    # ── Preprocess ────────────────────────────────────────────────────────────
    images = load_and_preprocess_images(image_names).to(device)
    print(f"[INFO] Preprocessed tensor shape: {tuple(images.shape)}")  # (1, S, 3, H, W)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("[INFO] Running VGGT inference …")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=(device == "cuda")):
            predictions, _ = model(images)

    # ── Decode camera poses ───────────────────────────────────────────────────
    print("[INFO] Decoding camera poses …")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic   # (1, S, 3, 4)
    predictions["intrinsic"] = intrinsic   # (1, S, 3, 3)

    # ── Move to CPU numpy, remove batch dim ───────────────────────────────────
    print("[INFO] Converting predictions to numpy …")
    np_preds = {}
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            np_preds[key] = val.float().cpu().numpy().squeeze(0)
        else:
            np_preds[key] = val
    # After squeeze(0):
    #   extrinsic         (S, 3, 4)
    #   intrinsic         (S, 3, 3)
    #   depth             (S, H, W, 1)
    #   depth_conf        (S, H, W)
    #   world_points      (S, H, W, 3)
    #   world_points_conf (S, H, W)
    #   images            (S, 3, H, W)

    # ── Choose point source ───────────────────────────────────────────────────
    if args.use_point_map and "world_points" in np_preds:
        print("[INFO] Using point-map branch (world_points)")
        world_points = np_preds["world_points"]                       # (S, H, W, 3)
        conf = np_preds.get(
            "world_points_conf",
            np.ones(world_points.shape[:3], dtype=np.float32),
        )
    else:
        print("[INFO] Unprojecting depth maps with decoded camera parameters")
        world_points = unproject_depth_map_to_point_map(
            np_preds["depth"],       # (S, H, W, 1)
            np_preds["extrinsic"],   # (S, 3, 4)
            np_preds["intrinsic"],   # (S, 3, 3)
        )                            # → (S, H, W, 3)
        conf = np_preds.get(
            "depth_conf",
            np.ones(world_points.shape[:3], dtype=np.float32),
        )

    # ── Derive colours from images  (S, 3, H, W) → (S, H, W, 3) ─────────────
    images_np = np_preds["images"]                      # (S, 3, H, W)
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # (S, H, W, 3)

    # ── Flatten ───────────────────────────────────────────────────────────────
    points_flat = world_points.reshape(-1, 3)                             # (N, 3)
    colors_flat = (images_np.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)  # (N, 3)
    conf_flat   = conf.reshape(-1)                                        # (N,)

    # ── Confidence filtering ──────────────────────────────────────────────────
    if args.conf_threshold > 0.0:
        threshold_val = np.percentile(conf_flat, args.conf_threshold)
    else:
        threshold_val = 0.0
    mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
    print(
        f"[INFO] Confidence threshold = {threshold_val:.5f} "
        f"(bottom {args.conf_threshold:.1f}% discarded)"
    )
    print(f"[INFO] Points after filtering: {mask.sum():,} / {len(mask):,}")

    points_out = points_flat[mask]
    colors_out = colors_flat[mask]

    # ── Save PLY ──────────────────────────────────────────────────────────────
    save_ply(args.output, points_out, colors_out)


if __name__ == "__main__":
    main()
