#!/usr/bin/env python3
"""
Export point cloud from VGGT predictions.

Usage:
    python export_pointcloud_from_npz.py --exp_dir demo_result/ --out_ply demo_result/out.ply
    python export_pointcloud_from_npz.py --exp_dir demo_result/ --out_ply demo_result/out.ply --image_dir /path/to/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_predictions(npz_path: Path) -> dict:
    """Load predictions from npz file."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """
    Write point cloud to PLY file.
    
    Args:
        path: Output PLY file path
        points: (N, 3) array of XYZ coordinates
        colors: Optional (N, 3) array of RGB colors (0-255 uint8 or 0-1 float)
    """
    n_points = points.shape[0]
    
    has_colors = colors is not None and len(colors) == n_points
    
    # Convert colors to uint8 if provided as float
    if has_colors:
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    with open(path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(n_points):
            x, y, z = points[i]
            if has_colors:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"Saved {n_points} points to {path}")


def load_images_for_colors(image_dir: Path, num_images: int) -> np.ndarray | None:
    """Load images to extract colors for point cloud."""
    if image_dir is None or not image_dir.exists():
        return None
    
    from PIL import Image
    
    valid_suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix in valid_suffixes],
        key=lambda p: p.name,
    )[:num_images]
    
    if not image_paths:
        return None
    
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        images.append(img_np)
    
    return images


def export_pointcloud(
    exp_dir: Path,
    out_ply: Path,
    image_dir: Path | None = None,
    conf_threshold: float = 0.5,
    use_survived: bool = True,
    max_depth: float = 100.0,
    subsample: int = 1,
) -> None:
    """
    Export point cloud from VGGT predictions.
    
    Args:
        exp_dir: Directory containing predictions npz files
        out_ply: Output PLY file path
        image_dir: Optional directory with source images for colors
        conf_threshold: Minimum confidence threshold for depth values
        use_survived: Use predictions_survived.npz if available
        max_depth: Maximum depth value to include
        subsample: Subsample factor for points (1 = all points, 2 = every other, etc.)
    """
    # Load predictions
    survived_path = exp_dir / "predictions_survived.npz"
    first_path = exp_dir / "predictions_first_forward.npz"
    
    if use_survived and survived_path.exists():
        npz_path = survived_path
        print(f"Loading predictions from {npz_path}")
    elif first_path.exists():
        npz_path = first_path
        print(f"Loading predictions from {npz_path}")
    else:
        raise FileNotFoundError(f"No predictions found in {exp_dir}")
    
    predictions = load_predictions(npz_path)
    
    # Print available keys
    print(f"Available keys: {list(predictions.keys())}")
    
    # Check if world_points is directly available (preferred)
    if "world_points" in predictions:
        world_points = predictions["world_points"]  # (B, N, H, W, 3) or (N, H, W, 3)
        depth_conf = predictions.get("world_points_conf", predictions.get("depth_conf", None))
        depth = predictions.get("depth", None)
        
        # Print shapes for debugging
        print(f"world_points shape: {world_points.shape}")
        if depth_conf is not None:
            print(f"depth_conf shape: {depth_conf.shape}")
        if depth is not None:
            print(f"depth shape: {depth.shape}")
        
        # Handle batch dimension
        if world_points.ndim == 5:  # (B, N, H, W, 3)
            world_points = world_points[0]  # (N, H, W, 3)
        if depth_conf is not None and depth_conf.ndim == 4:  # (B, N, H, W)
            depth_conf = depth_conf[0]  # (N, H, W)
        if depth is not None:
            if depth.ndim == 4:  # (B, N, H, W)
                depth = depth[0]  # (N, H, W)
            elif depth.ndim == 5:  # (B, N, H, W, 1)
                depth = depth[0].squeeze(-1)
            
        N, H, W, _ = world_points.shape
        print(f"Processing {N} images of size {H}x{W}")
        
    else:
        # Fall back to unprojecting from depth maps
        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        pose_enc = predictions["pose_enc"]
        
        print(f"depth shape: {depth.shape}")
        
        # Handle various shapes
        if depth.ndim == 5:  # (B, N, H, W, 1) or similar
            depth = depth.squeeze()
        if depth.ndim == 3:
            depth = depth[np.newaxis, ...]
        
        if depth.ndim == 4:
            B, N, H, W = depth.shape
        else:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")
            
        if depth_conf.ndim == 5:
            depth_conf = depth_conf.squeeze()
        if depth_conf.ndim == 3:
            depth_conf = depth_conf[np.newaxis, ...]
            
        if pose_enc.ndim == 2:
            pose_enc = pose_enc[np.newaxis, ...]
            
        print(f"Processing {N} images of size {H}x{W}")
        
        # Decode pose encoding to get extrinsics and intrinsics
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pose_enc_tensor = torch.from_numpy(pose_enc).float()
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc_tensor, (H, W))
        
        # Convert to numpy
        extrinsics = extrinsics.numpy()
        intrinsics = intrinsics.numpy()
        
        # Remove batch dimension if B=1
        if B == 1:
            depth = depth[0]
            depth_conf = depth_conf[0]
            extrinsics = extrinsics[0]
            intrinsics = intrinsics[0]
        
        # Unproject depth maps to world coordinates
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        depth_expanded = depth[..., np.newaxis]
        world_points = unproject_depth_map_to_point_map(depth_expanded, extrinsics, intrinsics)
    
    # Load images for colors if available
    # First check if images are in the predictions
    if "images" in predictions:
        images_from_pred = predictions["images"]  # likely (B, N, C, H, W) or (N, C, H, W)
        print(f"images shape from predictions: {images_from_pred.shape}")
        if images_from_pred.ndim == 5:
            images_from_pred = images_from_pred[0]  # (N, C, H, W)
        if images_from_pred.ndim == 4:
            # Convert from (N, C, H, W) to list of (H, W, C)
            images = [images_from_pred[i].transpose(1, 2, 0) for i in range(images_from_pred.shape[0])]
        else:
            images = None
    else:
        images = load_images_for_colors(image_dir, N) if image_dir else None
    
    # Collect all points with filtering
    all_points = []
    all_colors = []
    
    for i in range(N):
        pts = world_points[i]  # (H, W, 3)
        
        # Get confidence if available
        if depth_conf is not None:
            conf = depth_conf[i]  # (H, W)
        else:
            conf = np.ones((H, W), dtype=np.float32)
        
        # Get depth for filtering if available
        if depth is not None:
            d = depth[i]  # (H, W)
            depth_mask = (d > 0) & (d < max_depth)
        else:
            # Use z-coordinate from world points as proxy for depth filtering
            z_vals = pts[..., 2]
            depth_mask = np.isfinite(z_vals)
        
        # Create mask for valid points
        mask = (conf >= conf_threshold) & depth_mask & np.isfinite(pts).all(axis=-1)
        
        # Apply subsampling
        if subsample > 1:
            subsample_mask = np.zeros_like(mask, dtype=bool)
            subsample_mask[::subsample, ::subsample] = True
            mask = mask & subsample_mask
        
        valid_pts = pts[mask]  # (M, 3)
        all_points.append(valid_pts)
        
        # Get colors if available
        if images is not None and i < len(images):
            img = images[i]
            # Resize image to match depth map size if needed
            if img.shape[:2] != (H, W):
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
                img_pil = img_pil.resize((W, H), PILImage.BILINEAR)
                img = np.array(img_pil).astype(np.float32) / 255.0
            
            valid_colors = img[mask]  # (M, 3)
            all_colors.append(valid_colors)
    
    # Concatenate all points
    all_points = np.concatenate(all_points, axis=0)
    
    if all_colors:
        all_colors = np.concatenate(all_colors, axis=0)
    else:
        all_colors = None
    
    print(f"Total points: {all_points.shape[0]}")
    
    # Write PLY
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out_ply, all_points, all_colors)


def main():
    parser = argparse.ArgumentParser(description="Export point cloud from VGGT predictions")
    parser.add_argument("--exp_dir", type=Path, required=True, help="Directory containing predictions npz")
    parser.add_argument("--out_ply", type=Path, required=True, help="Output PLY file path")
    parser.add_argument("--image_dir", type=Path, default=None, help="Directory with source images for colors")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--max_depth", type=float, default=100.0, help="Maximum depth value")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample factor (1=all points)")
    parser.add_argument("--use_first", action="store_true", help="Use first forward predictions instead of survived")
    
    args = parser.parse_args()
    
    export_pointcloud(
        exp_dir=args.exp_dir,
        out_ply=args.out_ply,
        image_dir=args.image_dir,
        conf_threshold=args.conf_threshold,
        max_depth=args.max_depth,
        subsample=args.subsample,
        use_survived=not args.use_first,
    )


if __name__ == "__main__":
    main()
