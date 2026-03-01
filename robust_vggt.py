#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import einops
import numpy as np
import torch
import torchvision
import gc
from torch import Tensor

from tqdm import tqdm
import os
from PIL import Image

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[2]

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert batched SE3 matrices."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    t_inv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    Tin = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape)
    Tin = Tin.clone()
    Tin[..., :3, :3] = Rt
    Tin[..., :3, 3] = t_inv
    return Tin

@dataclass(frozen=True)
class ExperimentConfig:
    image_dir: Path
    preprocess_mode: str = "crop"
    exp_name: str = "demo_result"
    attn_a: float = 0.5
    cos_a: float = 0.5
    rej_thresh: float = 0.4
    use_point_map: bool = True
    conf_threshold_pct: float = 30.0


def safe_empty_cache() -> None:
    """Aggressively free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def free_cuda(*args):
    for arg in args:
        del arg
    safe_empty_cache()


def extrinsics_to_matrix(extrinsics: Tensor) -> Tensor:
    """Convert [N, 3, 4] extrinsics to homogenous [N, 4, 4] matrices."""
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics of shape [N,3,4], got {tuple(extrinsics.shape)}")
    n = extrinsics.shape[0]
    device = extrinsics.device
    dtype = extrinsics.dtype
    mats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(n, 4, 4).clone()
    mats[:, :3, :3] = extrinsics[:, :3, :3]
    mats[:, :3, 3] = extrinsics[:, :3, 3]
    return mats


def convert_world_to_cam_to_cam_to_world(extrinsics: Tensor) -> Tensor:
    """Invert world-to-camera extrinsics to obtain camera-to-world transforms."""
    T_w2c = extrinsics_to_matrix(extrinsics)
    return invert_se3(T_w2c)


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


def serialize_paths(value: Any) -> Any:
    """Recursively convert Path objects within nested structures to strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: serialize_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_paths(v) for v in value]
    return value


def select_device_and_dtype() -> tuple[str, torch.dtype]:
    """Choose the compute device and mixed-precision dtype."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


# ANSI color helpers for terminal output
ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"
def info_print(msg: str) -> None:
    print(f"{ANSI_GREEN}{msg}{ANSI_RESET}")


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian PLY point cloud (XYZ + RGB)."""
    assert points.shape[0] == colors.shape[0], "points and colors must have the same length"
    num_pts = points.shape[0]
    info_print(f"[INFO] Writing {num_pts:,} points → {path}")
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
        col_u8  = colors.astype(np.uint8)
        for i in range(num_pts):
            f.write(struct.pack("<fff", *pts_f32[i]))
            f.write(struct.pack("BBB",  *col_u8[i]))
    info_print(f"[INFO] PLY saved to {path}")


class RobustVGGTExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device, self.amp_dtype = select_device_and_dtype()
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

    def _forward_once(self, images: Tensor, image_hw: tuple[int, int], device: torch.device):
        import math
        import numpy as np
        import torch
        import torch.nn.functional as F
        from typing import List, Optional

        non_blocking = device.type == "cuda"
        images_device = images.to(device=device, non_blocking=non_blocking)
        if device.type == "cuda":
            images_device = images_device.to(device=device, dtype=self.amp_dtype, non_blocking=non_blocking)

        image_level_attn_mask: Optional[Tensor] = None
        rejected_image_indices: List[int] = []
        image_attention_means: List[float] = []
        global_norm_means: List[float] = []

        attn_layers: List[int] = []
        q_out: Dict[int, Tensor] = {}
        k_out: Dict[int, Tensor] = {}
        handles: List[Any] = []

        attn_layers = [23]
        info_print(f"Setting up hooks for layers: {attn_layers}")

        def _make_hook(store_dict, idx):
            def _hook(_module, _inp, out):
                store_dict[idx] = out.detach()
            return _hook

        for i in attn_layers:
            blk = self.model.aggregator.global_blocks[i].attn
            handles.append(blk.q_norm.register_forward_hook(_make_hook(q_out, i)))
            handles.append(blk.k_norm.register_forward_hook(_make_hook(k_out, i)))

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    predictions, aggregated_tokens_list = self.model(images_device)
            else:
                predictions, aggregated_tokens_list = self.model(images_device)

        prediction_save_path = self.pair_out_dir / f"predictions_first_forward.npz"
        save_predictions = {k: v.float().cpu() for k, v in predictions.items() if torch.is_tensor(v)}
        np.savez(prediction_save_path, **save_predictions)

        '''
        aggregated_tokens_list: features from all layers (list of tensors). 
        '''
        target_layers = [23]
        aggregated_tokens_selected = [aggregated_tokens_list[idx] for idx in target_layers]
        global_tokens = [tokens[..., 1024:] for tokens in aggregated_tokens_selected]

        aggregator = self.model.aggregator
        patch_size = aggregator.patch_size
        patch_start_idx = aggregator.patch_start_idx
        H, W = image_hw
        h_patches = H // patch_size 
        w_patches = W // patch_size
        num_patch_tokens = h_patches * w_patches
        tokens_per_image = patch_start_idx + num_patch_tokens
            
        cosine_similarities = []
        num_images_total = images.shape[0]
        reject_flags = [False] * num_images_total  
        layer_feat = ref_feat = ref_feat_norm = layer_feat_norm = cos_sim = cos_sim_mean = None
        for layer_idx, feature in enumerate(global_tokens):
            if feature.ndim != 4:
                continue
            B, N, T, C = feature.shape 
            if T == 0 or C == 0:
                continue
            
            feature = feature[:, :, patch_start_idx:, :]
            B, N, T, C = feature.shape
            layer_feat = feature.detach().to(dtype=torch.float32)
            num_samples = B * N
            layer_feat = layer_feat.reshape(num_samples, T, C)
            
            ref_feat = layer_feat[0:1, :, :]  # (1, T, C)
            ref_feat_norm = F.normalize(ref_feat, p=2, dim=-1)  # (1, T, C)
            layer_feat_norm = F.normalize(layer_feat, p=2, dim=-1)  # (B*N, T, C)
            cos_sim = torch.einsum("bic,bjc->bij", layer_feat_norm, ref_feat_norm)  # (B*N, T, T)
            cos_sim_mean = cos_sim.mean(-1).mean(-1)  # (B*N,)
            cosine_similarities.append(cos_sim_mean)  # List of (N,)
        
        global_tokens = None
        aggregated_tokens_selected = None
        safe_empty_cache()
        
        predictions_full = predictions
        try:
            if isinstance(predictions_full, dict):
                for _k, _v in list(predictions_full.items()):
                    if torch.is_tensor(_v) and _v.device.type == "cuda":
                        predictions_full[_k] = _v.detach().cpu()
        except Exception:
            pass
        safe_empty_cache()
        torch.cuda.empty_cache()

        # Extract first-round predictions for PLY (before filtering)
        first_pose_enc = predictions_full["pose_enc"]
        first_ext_raw, first_int_raw = pose_encoding_to_extri_intri(first_pose_enc, image_hw)
        first_ext_raw = first_ext_raw.squeeze(0) if first_ext_raw.ndim == 4 else first_ext_raw
        first_int_raw = first_int_raw.squeeze(0) if first_int_raw.ndim == 4 else first_int_raw
        first_depth = predictions_full["depth"].detach().cpu().float()
        first_conf = predictions_full["depth_conf"].detach().cpu().float()
        first_intrinsics_cpu = first_int_raw.detach().cpu().float()
        first_extrinsics_cpu = first_ext_raw.detach().cpu().float()
        first_world_points_cpu: Optional[Tensor] = None
        first_world_points_conf_cpu: Optional[Tensor] = None
        if "world_points" in predictions_full and torch.is_tensor(predictions_full["world_points"]):
            wp = predictions_full["world_points"]
            first_world_points_cpu = (wp.squeeze(0) if wp.ndim == 5 else wp).detach().cpu().float()
        if "world_points_conf" in predictions_full and torch.is_tensor(predictions_full["world_points_conf"]):
            wpc = predictions_full["world_points_conf"]
            first_world_points_conf_cpu = (wpc.squeeze(0) if wpc.ndim == 4 else wpc).detach().cpu().float()

        global_mean_vals = []
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        try:
            handles.clear()
        except Exception:
            pass

        def _num_input_images(x: Tensor) -> int:
            if x.ndim == 5:   # B, N, C, H, W
                return int(x.shape[1])
            if x.ndim == 4:   # N, C, H, W
                return int(x.shape[0])
            if x.ndim == 3:   # C, H, W
                return 1
            raise ValueError(f"Unsupported images shape: {x.shape}")

        total_imgs = _num_input_images(images)
        num_vis = total_imgs  # Visualize all images
        info_print(f"[INFO] Visualizing self-attention maps for the first {num_vis} input images.")
        batch_size = images.shape[0] if images.ndim == 5 else 1

        if images.ndim == 5:
            base_images = images[0, :num_vis].detach().cpu()         # First num_vis images of the first batch
        elif images.ndim == 4:
            base_images = images[:num_vis].detach().cpu()
        elif images.ndim == 3:
            base_images = images.unsqueeze(0).detach().cpu()
        else:
            raise ValueError(f"Unsupported images shape: {images.shape}")

        aggregator = self.model.aggregator
        patch_size = aggregator.patch_size
        patch_start_idx = aggregator.patch_start_idx
        H, W = image_hw
        h_patches = H // patch_size  
        w_patches = W // patch_size
        num_patch_tokens = h_patches * w_patches
        tokens_per_image = patch_start_idx + num_patch_tokens
        
        global_norms_list=[]

        avg_maps2d_sum = [torch.zeros((h_patches, w_patches), dtype=torch.float32) for _ in range(num_vis)]
        avg_counts = [0 for _ in range(num_vis)]
        reject_flags = [False] * num_vis
        image_attention_means = [float("nan")] * num_vis
        global_norm_means = [float("nan")] * num_vis
        rejection_threshold = self.config.rej_thresh

        first_image_patch_start = patch_start_idx
        first_image_patch_end = first_image_patch_start + num_patch_tokens

        for i in tqdm(attn_layers):
            global_norms_list=[]
            if i not in q_out or i not in k_out:
                continue

            Q = q_out[i]
            K = k_out[i]

            T = int(K.shape[-2])
            num_images_in_seq = T // tokens_per_image
            if num_images_in_seq <= 0:
                continue
            q_first_image = Q[:, :, first_image_patch_start:first_image_patch_end, :]  # (B, H, Nq, D)
            Tk = int(min(num_vis, num_images_in_seq) * tokens_per_image)              # (Tk)
            K_slice = K[:, :, :Tk, :]                                                # (B, H, Tk, D)
            scale = 1.0 / math.sqrt(float(q_first_image.shape[-1]))
            logits = torch.einsum("bhqd,bhtd->bhqt", q_first_image, K_slice) * scale  # (B, H, Nq, Tk)
            probs = torch.softmax(logits, dim=-1)                                      # (B, H, Nq, Tk)
            attn_first_image = probs.mean(dim=1).mean(dim=1)[0]                        # (Tk,)

            import matplotlib.pyplot as plt

            maps_up = []        # Torch tensors upsampled to (H,W) via bilinear interpolation
            maps_2d = []        # Original patch grid attention (h_patches, w_patches)
            valid_indices = []  # Image indices that were actually drawable

            global_min = None
            global_max = None

            for img_idx in range(num_vis):
                start = img_idx * tokens_per_image + patch_start_idx
                end = start + num_patch_tokens
                if start >= attn_first_image.shape[-1]:
                    break
                end = min(end, attn_first_image.shape[-1])

                patch_attn = attn_first_image[start:end]  
                if patch_attn.numel() != num_patch_tokens:
                    continue

                attn2d = patch_attn.view(h_patches, w_patches)  # (h, w)

                attn2d_up = F.interpolate(
                    attn2d.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )[0, 0]  # (H, W)

                maps_2d.append(attn2d)       # (h, w)
                maps_up.append(attn2d_up)    # (H, W)
                valid_indices.append(img_idx)

                vmin = torch.min(attn2d)
                vmax = torch.max(attn2d)
                global_min = vmin if (global_min is None) else torch.minimum(global_min, vmin)
                global_max = vmax if (global_max is None) else torch.maximum(global_max, vmax)

            if len(maps_up) == 0:
                free_cuda(Q, K, q_first_image, K_slice, logits, probs, attn_first_image)
                continue
            
            for idx, img_idx in enumerate(valid_indices):
                avg_maps2d_sum[img_idx] += maps_2d[idx].detach().cpu()
                avg_counts[img_idx] += 1

            eps = 1e-12
            gmin = float(global_min.item())
            gmax = float(global_max.item())
            gden = (gmax - gmin) if (gmax > gmin) else 1.0

            for col_idx, img_idx in enumerate(valid_indices):
                role = "anchor" if img_idx == 0 else "support"

                img_tensor = base_images[img_idx].float()  # (C, H, W)
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0.0, 1.0)

                m_up = maps_up[col_idx]
                m_up_np = m_up.detach().cpu().numpy()

                lmin = float(m_up.min().item())
                lmax = float(m_up.max().item())
                lden = (lmax - lmin) if (lmax > lmin) else 1.0
                local_norm = (m_up_np - lmin) / lden

                global_norm = (m_up_np - gmin) / gden
                global_mean_val = float(global_norm.mean())
                global_mean_vals.append(global_mean_val)
                global_norm_means[img_idx] = global_mean_val
                global_norms_list.append(global_norm)
       
            free_cuda(Q, K, q_first_image, K_slice, logits, probs, attn_first_image)
            del Q, K, q_first_image, K_slice, logits, probs, attn_first_image
            try:
                del maps_up, maps_2d, m_up
            except Exception:
                pass
            torch.cuda.empty_cache()
                
        if global_mean_vals and cosine_similarities:
            cos_sim = cosine_similarities[0]  # (N,)
            attn_val = torch.tensor(global_mean_vals, device=cos_sim.device, dtype=cos_sim.dtype)  # (N,)

            cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-6)
            attn_val = (attn_val - attn_val.min()) / (attn_val.max() - attn_val.min() + 1e-6)
            
            combined_score = self.config.attn_a * attn_val + self.config.cos_a * cos_sim

            # Print per-view scores
            info_print("[INFO] Per-view scores:")
            for idx in range(len(combined_score)):
                role = "anchor" if idx == 0 else f"view {idx}"
                info_print(
                    f"  [{role}] cos={cos_sim[idx].item():.4f}  "
                    f"attn={attn_val[idx].item():.4f}  "
                    f"combined={combined_score[idx].item():.4f}"
                )

            # Reject
            for idx in range(len(combined_score)):
                if idx == 0:
                    continue  # Never reject the reference image
                if combined_score[idx] < self.config.rej_thresh:
                    reject_flags[idx] = True
            info_print(f"[INFO] Rejection threshold: {self.config.rej_thresh:.4f}")
        
        rejected_image_indices = [idx for idx, flag in enumerate(reject_flags) if flag and idx != 0]
        info_print(f"[INFO] Integrated rejection: {rejected_image_indices}")

        del avg_maps2d_sum, avg_counts
        try:
            q_out.clear(); k_out.clear()
        except Exception:
            pass
        del q_out, k_out
        try:
            del base_images
        except Exception:
            pass
        safe_empty_cache()
            
        aggregated_tokens_list = None
        aggregated_tokens_selected = None
        global_tokens = None
        cosine_similarities = []
        safe_empty_cache()
            
        if len(rejected_image_indices) > 0:
            if images.ndim == 5:  # (B, N, C, H, W)
                total_N = int(images.shape[1])
                B_dim = int(images.shape[0])
            elif images.ndim == 4:  # (N, C, H, W)
                total_N = int(images.shape[0])
                B_dim = 1
            elif images.ndim == 3:  # (C, H, W)
                total_N = 1
                B_dim = 1
            else:
                raise ValueError(f"Unsupported images shape: {images.shape}")

            survivors = [i for i in range(total_N) if i not in rejected_image_indices]

            if len(survivors) == 0:
                info_print("[INFO] All images rejected by attention-vis; skipping second forward.")
            elif len(survivors) == total_N:
                pass
            else:
                try:
                    del image_level_attn_mask
                except Exception:
                    pass
                try:
                    del images_device
                except Exception:
                    pass
                safe_empty_cache()

                if images.ndim == 5:
                    images_subset_cpu = images[:, survivors, ...]
                elif images.ndim == 4:
                    images_subset_cpu = images[survivors, ...]
                elif images.ndim == 3:
                    images_subset_cpu = images.unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported images shape: {images.shape}")

                # Save the survivor images used for the second forward pass
                try:
                    survived_dir = self.pair_out_dir / "survived_images"
                    survived_dir.mkdir(parents=True, exist_ok=True)
                    survivors_vis = images_subset_cpu.detach().cpu().float()
                    if survivors_vis.ndim == 5:
                        survivors_vis = survivors_vis[0]  # (N, C, H, W)
                    for save_idx, orig_idx in enumerate(survivors):
                        img_np = survivors_vis[save_idx].permute(1, 2, 0).numpy()
                        img_np = np.clip(img_np, 0.0, 1.0)
                        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                        img_pil.save(survived_dir / f"survived_{orig_idx:04d}.png")
                    info_print(f"[INFO] Saved {len(survivors)} second-round input images to {survived_dir}")
                except Exception as _e:
                    print(f"[WARN] Failed to save second-round images: {_e}")

                images_subset = images_subset_cpu.to(device=device, dtype=self.amp_dtype, non_blocking=non_blocking)
        
                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                            predictions_survive, _ = self.model(images_subset)
                    else:
                        predictions_survive, _ = self.model(images_subset)
                
                prediction_save_path = self.pair_out_dir / f"predictions_survived.npz"
                save_predictions = {k: v.float().cpu() for k, v in predictions_survive.items() if torch.is_tensor(v)}
                np.savez(prediction_save_path, **save_predictions)

                try:
                    if isinstance(predictions_survive, dict):
                        for _k, _v in list(predictions_survive.items()):
                            if torch.is_tensor(_v) and _v.device.type == "cuda":
                                predictions_survive[_k] = _v.detach().cpu()
                    
                        predictions = predictions_survive
                except Exception:
                    print("[WARN] Failed to move survivor predictions to CPU")
                    pass
                try:
                    del images_subset, predictions_survive
                except Exception:
                    pass
                safe_empty_cache()
        else:
            prediction_save_path = self.pair_out_dir / f"predictions_survived.npz"
            save_predictions = {k: v.float().cpu() for k, v in predictions_full.items() if torch.is_tensor(v)}
            np.savez(prediction_save_path, **save_predictions)
            predictions = predictions_full

        pose_enc = predictions["pose_enc"]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_hw)
        extrinsics = extrinsics.squeeze(0) if extrinsics.ndim == 4 else extrinsics
        intrinsics = intrinsics.squeeze(0) if intrinsics.ndim == 4 else intrinsics
        Twc = convert_world_to_cam_to_cam_to_world(extrinsics)
        result = Twc.detach().cpu().float()
        intrinsics_cpu = intrinsics.detach().cpu().float()
        extrinsics_cpu = extrinsics.detach().cpu().float()  # world-to-cam (S, 3, 4)
        conf = predictions["depth_conf"].detach().cpu().float()
        depth = predictions["depth"].detach().cpu().float()

        world_points_cpu: Optional[Tensor] = None
        world_points_conf_cpu: Optional[Tensor] = None
        if "world_points" in predictions and torch.is_tensor(predictions["world_points"]):
            wp = predictions["world_points"]
            world_points_cpu = (wp.squeeze(0) if wp.ndim == 5 else wp).detach().cpu().float()
        if "world_points_conf" in predictions and torch.is_tensor(predictions["world_points_conf"]):
            wpc = predictions["world_points_conf"]
            world_points_conf_cpu = (wpc.squeeze(0) if wpc.ndim == 4 else wpc).detach().cpu().float()

        del predictions
        try:
            del images_device
        except Exception:
            pass
        safe_empty_cache()

        return (
            result, depth, conf, intrinsics_cpu, extrinsics_cpu,
            world_points_cpu, world_points_conf_cpu, rejected_image_indices,
            first_depth, first_conf, first_intrinsics_cpu, first_extrinsics_cpu,
            first_world_points_cpu, first_world_points_conf_cpu,
        )

    def _save_ply(
        self,
        ply_path: Path,
        images: Tensor,
        depth: Tensor,
        conf: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        world_points: Optional[Tensor],
        world_points_conf: Optional[Tensor],
        conf_threshold_pct: float = 50.0,
    ) -> None:
        """Build a coloured point cloud and write it as a binary PLY file.

        Supports two point sources (controlled by self.config.use_point_map):
          • point-map branch (default, use_point_map=True): uses world_points
            predicted directly by the point head — highest quality, no pose error.
          • depth branch (use_point_map=False): unprojects depth maps using the
            decoded camera extrinsics / intrinsics via unproject_depth_map_to_point_map.
        Confidence filtering is percentile-based so it is scale-independent.
        All inputs must be pre-aligned to the same set of survivor frames.
        """
        # ── Convert tensors to numpy, normalise batch/channel dims ───────────
        # depth: (B, S, H, W, 1) or (S, H, W, 1) → numpy (S, H, W, 1)
        depth_np = depth.float().cpu().numpy()
        if depth_np.ndim == 5:
            depth_np = depth_np.squeeze(0)

        # conf: (B, S, H, W) or (S, H, W) → numpy (S, H, W)
        conf_np = conf.float().cpu().numpy()
        if conf_np.ndim == 4:
            conf_np = conf_np.squeeze(0)

        # extrinsics: world-to-cam (S, 3, 4)
        ext_np = extrinsics.float().cpu().numpy()
        # intrinsics: (S, 3, 3)
        int_np = intrinsics.float().cpu().numpy()

        # images: (1, S, C, H, W) or (S, C, H, W) → (S, H, W, 3)
        images_np = images.float().cpu()
        if images_np.ndim == 5:
            images_np = images_np.squeeze(0)
        images_np = images_np.permute(0, 2, 3, 1).numpy()  # (S, H, W, 3)

        # ── Choose point source ───────────────────────────────────────────────
        if self.config.use_point_map and world_points is not None:
            info_print("[INFO] PLY: using point-map branch (world_points)")
            pts_np = world_points.float().cpu().numpy()     # (S, H, W, 3)
            if pts_np.ndim == 5:
                pts_np = pts_np.squeeze(0)
            pts_conf_np = (
                world_points_conf.float().cpu().numpy()
                if world_points_conf is not None
                else np.ones(pts_np.shape[:3], dtype=np.float32)
            )
            if pts_conf_np.ndim == 4:
                pts_conf_np = pts_conf_np.squeeze(0)
        else:
            info_print("[INFO] PLY: unprojecting depth maps with decoded camera parameters")
            pts_np      = unproject_depth_map_to_point_map(depth_np, ext_np, int_np)  # (S, H, W, 3)
            pts_conf_np = conf_np   # (S, H, W)

        # ── Flatten ───────────────────────────────────────────────────────────
        points_flat = pts_np.reshape(-1, 3)                                            # (N, 3)
        colors_flat = (images_np.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)  # (N, 3)
        conf_flat   = pts_conf_np.reshape(-1)                                          # (N,)

        # ── Percentile-based confidence filtering ─────────────────────────────
        threshold_val = np.percentile(conf_flat, conf_threshold_pct) if conf_threshold_pct > 0.0 else 0.0
        mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        info_print(
            f"[INFO] Confidence threshold = {threshold_val:.5f} "
            f"(bottom {conf_threshold_pct:.1f}% discarded)"
        )
        info_print(f"[INFO] Points after filtering: {mask.sum():,} / {len(mask):,}")

        if not mask.any():
            info_print("[WARN] No points survived confidence filtering; PLY not written.")
            return

        save_ply(str(ply_path), points_flat[mask], colors_flat[mask])

    def run_demo(self) -> None:
        self.pair_out_dir = Path(self.config.exp_name)
        self.pair_out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_image_paths(self.config.image_dir)
        info_print(f"[INFO] Found {len(image_paths)} images in {self.config.image_dir}")

        images_tensor = load_and_preprocess_images(
            [str(p) for p in image_paths],
            mode=self.config.preprocess_mode,
        )
        
        try:
            import matplotlib.pyplot as plt
            from torchvision.utils import make_grid
            images_cpu = images_tensor.detach().cpu()
            grid = make_grid(images_cpu, nrow=8, padding=2, pad_value=1.0)
            grid_np = grid.permute(1, 2, 0).numpy()
            plt.imsave(self.pair_out_dir / "preprocessed_grid.png", np.clip(grid_np, 0.0, 1.0))
        except Exception:
            pass

        # Save anchor image (first image)
        anchor_img = images_tensor[0].detach().cpu().float()
        anchor_np = anchor_img.permute(1, 2, 0).numpy()
        anchor_np = np.clip(anchor_np, 0.0, 1.0)
        anchor_pil = Image.fromarray((anchor_np * 255).astype(np.uint8))
        anchor_pil.save(self.pair_out_dir / "anchor.png")
        info_print(f"[INFO] Saved anchor image to {self.pair_out_dir / 'anchor.png'}")

        if torch.device(self.device).type == "cuda":
            images_tensor = images_tensor.pin_memory()

        image_hw = (int(images_tensor.shape[-2]), int(images_tensor.shape[-1]))
        
        (
            pred_twcs, depth, conf, intrinsics, extrinsics,
            world_points, world_points_conf, rejected_indices,
            first_depth, first_conf, first_intrinsics, first_extrinsics,
            first_world_points, first_world_points_conf,
        ) = self._forward_once(images_tensor, image_hw, torch.device(self.device))

        # Save first-round PLY (before filtering, all images)
        ply_before = self.pair_out_dir / "before_filtering.ply"
        self._save_ply(
            ply_before, images_tensor, first_depth, first_conf,
            first_intrinsics, first_extrinsics,
            first_world_points, first_world_points_conf,
            self.config.conf_threshold_pct,
        )
        info_print(f"[INFO] Saved PLY to {ply_before}")

        # Subset images to match survivor-only predictions
        # (depth / extrinsics / world_points are already subsetted by _forward_once)
        if rejected_indices:
            N_total = images_tensor.shape[1] if images_tensor.ndim == 5 else images_tensor.shape[0]
            survivor_ids = [i for i in range(N_total) if i not in rejected_indices]
            if images_tensor.ndim == 5:
                images_for_ply = images_tensor[:, survivor_ids, ...]
            else:
                images_for_ply = images_tensor[survivor_ids, ...]
        else:
            images_for_ply = images_tensor

        # Save second-round PLY (after filtering, survivor images only)
        ply_after = self.pair_out_dir / "after_filtering.ply"
        self._save_ply(
            ply_after, images_for_ply, depth, conf, intrinsics, extrinsics,
            world_points, world_points_conf, self.config.conf_threshold_pct,
        )
        info_print(f"[INFO] Saved PLY to {ply_after}")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run VGGT demo on a directory of images.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to process.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["crop", "pad"],
        default="crop",
        help="Image preprocessing mode.",
    )
    parser.add_argument("--exp-name", type=str, default="demo_result", help="Experiment name for output directory.")
    parser.add_argument("--attn_a", type=float, default=0.5, help="Attention weight.")
    parser.add_argument("--cos_a", type=float, default=0.5, help="Cosine similarity weight.")
    parser.add_argument("--rej-thresh", type=float, default=0.4, help="Rejection threshold.")
    parser.add_argument(
        "--no-point-map", dest="use_point_map", action="store_false",
        help="Use depth unprojection instead of the point-map branch for PLY generation.",
    )
    parser.set_defaults(use_point_map=True)
    parser.add_argument(
        "--conf-threshold-pct", type=float, default=30.0,
        help="Discard the bottom N%% of points by confidence score (0–100, default 30).",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        image_dir=args.image_dir,
        preprocess_mode=args.preprocess_mode,
        exp_name=args.exp_name,
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        use_point_map=args.use_point_map,
        conf_threshold_pct=args.conf_threshold_pct,
    )


def main() -> None:
    config = parse_args()
    experiment = RobustVGGTExperiment(config)
    experiment.run_demo()


if __name__ == "__main__":
    main()
