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
import shutil
from PIL import Image

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]


def _resolve_external_repo(env_var: str, candidates: List[Path]) -> Path:
    env_val = os.environ.get(env_var)
    if env_val:
        p = Path(env_val).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"{env_var}={p} does not exist")
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError(
        f"Could not locate external repo via {env_var}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


# Add Pi3 to path — override with PI3_ROOT env var if needed
PI3_ROOT = _resolve_external_repo(
    "PI3_ROOT",
    [
        REPO_ROOT / "Pi3",
        REPO_ROOT / "Robust-X" / "Pi3",
        REPO_ROOT.parent / "Pi3",
        REPO_ROOT.parent / "Robust-X" / "Pi3",
    ],
)
sys.path.insert(0, str(PI3_ROOT))

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge


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
    exp_name: str = "demo_result"
    attn_a: float = 0.5
    cos_a: float = 0.5
    rej_thresh: float = 0.4
    max_images: int = 400


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


class RobustPi3Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device, self.amp_dtype = select_device_and_dtype()
        
        info_print("[INFO] Loading Pi3 model...")
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

    def _forward_once(self, images: Tensor, image_hw: tuple[int, int], device: torch.device, image_paths: Optional[List[Path]] = None) -> Tensor:
        import math
        import numpy as np
        import torch
        import torch.nn.functional as F
        from typing import List, Optional

        non_blocking = device.type == "cuda"
        
        # Pi3 expects (B, N, C, H, W) input
        if images.ndim == 4:
            images = images.unsqueeze(0)  # Add batch dimension
        
        images_device = images.to(device=device, non_blocking=non_blocking)
        if device.type == "cuda":
            images_device = images_device.to(device=device, dtype=self.amp_dtype, non_blocking=non_blocking)

        rejected_image_indices: List[int] = []
        image_attention_means: List[float] = []
        global_norm_means: List[float] = []

        # Pi3 model has 36 decoder layers for 'large' size
        # Odd layers (1, 3, 5, ...) perform inter-frame attention
        # Layer 17 shows the largest gap between clean and distractor views (per paper)
        attn_layers: List[int] = [33]
        info_print(f"Setting up hooks for layers: {attn_layers}")
        
        q_out: Dict[int, Tensor] = {}
        k_out: Dict[int, Tensor] = {}
        handles: List[Any] = []

        def _make_hook(store_dict, idx):
            def _hook(_module, _inp, out):
                store_dict[idx] = out.detach()
            return _hook

        # Hook into the decoder blocks' attention q_norm and k_norm
        for i in attn_layers:
            blk = self.model.decoder[i].attn
            handles.append(blk.q_norm.register_forward_hook(_make_hook(q_out, i)))
            handles.append(blk.k_norm.register_forward_hook(_make_hook(k_out, i)))

        # Store hidden states for cosine similarity computation
        hidden_states: Dict[int, Tensor] = {}
        
        def _make_hidden_hook(store_dict, idx):
            def _hook(_module, _inp, out):
                store_dict[idx] = out.detach()
            return _hook

        # Hook into last decoder layer output
        for i in attn_layers:
            handles.append(self.model.decoder[i].register_forward_hook(_make_hidden_hook(hidden_states, i)))

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    predictions = self.model(images_device)
            else:
                predictions = self.model(images_device)

        # Save initial predictions
        prediction_save_path = self.pair_out_dir / f"predictions_first_forward.npz"
        save_predictions = {k: v.float().cpu().numpy() for k, v in predictions.items() if torch.is_tensor(v)}
        # Extract depth from local_points (z coordinate)
        save_predictions["depth"] = predictions["local_points"][..., 2].float().cpu().numpy()
        # Add aliases for compatibility with visualization scripts
        save_predictions["world_points"] = save_predictions["points"]
        save_predictions["depth_conf"] = save_predictions["conf"].squeeze(-1)  # (B, N, H, W)
        np.savez(prediction_save_path, **save_predictions)

        # Compute cosine similarities using hidden states
        cosine_similarities = []
        num_images_total = images.shape[1]  # B, N, C, H, W
        reject_flags = [False] * num_images_total
        
        H, W = image_hw
        patch_size = self.model.patch_size  # 14
        h_patches = H // patch_size 
        w_patches = W // patch_size
        num_patch_tokens = h_patches * w_patches
        patch_start_idx = self.model.patch_start_idx  # 5 (register tokens)
        tokens_per_image = patch_start_idx + num_patch_tokens

        # Process hidden states for cosine similarity
        for layer_idx, feature in hidden_states.items():
            # The layer processes with shape (B, N*hw, C) for odd layers
            if feature.ndim < 2:
                continue
            
            B_times_hw_or_N_times_hw, T, C = feature.shape
            
            # For inter-frame layers (odd), hidden is shaped (B, N*tokens_per_image, C)
            # We need to extract per-image features
            layer_feat = feature.detach().to(dtype=torch.float32)
            
            # Reshape to (B, N, tokens_per_image, C) if possible
            B_dim = images.shape[0]
            N_dim = images.shape[1]
            
            # Check if this is an inter-frame layer output
            if T == N_dim * tokens_per_image:
                layer_feat = layer_feat.reshape(B_dim, N_dim, tokens_per_image, C)
                # Extract patch tokens only (skip register tokens)
                layer_feat = layer_feat[:, :, patch_start_idx:, :]  # (B, N, num_patch_tokens, C)
                
                # Average over batch dimension
                layer_feat = layer_feat.mean(0)  # (N, num_patch_tokens, C)
                
                ref_feat = layer_feat[0:1, :, :]  # (1, num_patch_tokens, C)
                ref_feat_norm = F.normalize(ref_feat, p=2, dim=-1)
                layer_feat_norm = F.normalize(layer_feat, p=2, dim=-1)
                cos_sim = torch.einsum("bic,bjc->bij", layer_feat_norm, ref_feat_norm)  # (N, T, T)
                cos_sim_mean = cos_sim.mean(-1).mean(-1)  # (N,)
                cosine_similarities.append(cos_sim_mean)

        safe_empty_cache()

        # Remove hooks
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

        # Visualize attention
        num_vis = num_images_total
        info_print(f"[INFO] Visualizing self-attention maps for {num_vis} input images.")
        
        global_mean_vals = []
        global_norms_list = []

        # Process attention from Q and K outputs
        avg_maps2d_sum = [torch.zeros((h_patches, w_patches), dtype=torch.float32) for _ in range(num_vis)]
        avg_counts = [0 for _ in range(num_vis)]
        image_attention_means = [float("nan")] * num_vis
        global_norm_means = [float("nan")] * num_vis

        first_image_patch_start = patch_start_idx
        first_image_patch_end = first_image_patch_start + num_patch_tokens

        for i in tqdm(attn_layers, desc="Processing attention layers"):
            if i not in q_out or i not in k_out:
                continue

            Q = q_out[i]  # (B, num_heads, seq_len, head_dim)
            K = k_out[i]  # (B, num_heads, seq_len, head_dim)

            # For odd layers (inter-frame), Q and K have shape (B, num_heads, N*tokens_per_image, head_dim)
            if Q.ndim != 4:
                continue
                
            B_dim, num_heads, T, head_dim = Q.shape
            
            # Check if this is inter-frame attention
            if T != num_images_total * tokens_per_image:
                continue
                
            # Extract attention from first image patches to all other image patches
            q_first_image = Q[:, :, first_image_patch_start:first_image_patch_end, :]  # (B, H, num_patch_tokens, D)
            
            scale = 1.0 / math.sqrt(float(head_dim))
            
            # Compute attention to all images
            Tk = min(num_vis, num_images_total) * tokens_per_image
            K_slice = K[:, :, :Tk, :]  # (B, H, Tk, D)
            
            # Compute attention per head to avoid OOM from large (B, H, Nq, Tk) tensors
            num_heads_l = q_first_image.shape[1]
            attn_accum = torch.zeros(Tk, device=q_first_image.device, dtype=torch.float32)
            for h_idx in range(num_heads_l):
                head_logits = torch.einsum("bqd,btd->bqt", q_first_image[:, h_idx], K_slice[:, h_idx]) * scale
                head_probs = torch.softmax(head_logits, dim=-1)
                del head_logits
                attn_accum += head_probs.mean(dim=1)[0]
                del head_probs
            attn_first_image = attn_accum / num_heads_l  # (Tk,)

            maps_up = []
            maps_2d = []
            valid_indices = []
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

                attn2d = patch_attn.view(h_patches, w_patches)

                attn2d_up = F.interpolate(
                    attn2d.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )[0, 0]

                maps_2d.append(attn2d)
                maps_up.append(attn2d_up)
                valid_indices.append(img_idx)

                vmin = torch.min(attn2d)
                vmax = torch.max(attn2d)
                global_min = vmin if (global_min is None) else torch.minimum(global_min, vmin)
                global_max = vmax if (global_max is None) else torch.maximum(global_max, vmax)

            if len(maps_up) == 0:
                free_cuda(Q, K, q_first_image, K_slice, attn_first_image)
                continue

            for idx, img_idx in enumerate(valid_indices):
                avg_maps2d_sum[img_idx] += maps_2d[idx].detach().cpu()
                avg_counts[img_idx] += 1

            eps = 1e-12
            gmin = float(global_min.item())
            gmax = float(global_max.item())
            gden = (gmax - gmin) if (gmax > gmin) else 1.0

            for col_idx, img_idx in enumerate(valid_indices):
                m_up = maps_up[col_idx]
                m_up_np = m_up.detach().cpu().numpy()

                global_norm = (m_up_np - gmin) / gden
                global_mean_val = float(global_norm.mean())
                global_mean_vals.append(global_mean_val)
                global_norm_means[img_idx] = global_mean_val
                global_norms_list.append(global_norm)

            free_cuda(Q, K, q_first_image, K_slice, attn_first_image)
            del Q, K, q_first_image, K_slice, attn_first_image
            try:
                del maps_up, maps_2d, m_up
            except Exception:
                pass
            torch.cuda.empty_cache()

        # Compute combined score and rejection
        per_image_scores = torch.zeros(num_images_total)
        per_image_cos = torch.zeros(num_images_total)
        per_image_attn = torch.zeros(num_images_total)
        if global_mean_vals and cosine_similarities:
            cos_sim = cosine_similarities[0]  # (N,)
            attn_val = torch.tensor(global_mean_vals[:num_vis], device=cos_sim.device, dtype=cos_sim.dtype)
            info_print(f"[INFO] Raw cosine similarity values: {[f'{v:.4f}' for v in cos_sim.tolist()]}")
            info_print(f"[INFO] Raw attention values: {[f'{v:.4f}' for v in attn_val.tolist()]}")

            # Normalize
            cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-6)
            attn_val = (attn_val - attn_val.min()) / (attn_val.max() - attn_val.min() + 1e-6)

            combined_score = self.config.attn_a * attn_val + self.config.cos_a * cos_sim
            info_print(f"[INFO] Normalized cosine similarity: {[f'{v:.4f}' for v in cos_sim.tolist()]}")

            n = len(combined_score)
            per_image_scores[:n] = combined_score.cpu()
            per_image_cos[:n] = cos_sim.cpu()
            per_image_attn[:n] = attn_val.cpu()

            # Reject
            for idx in range(len(combined_score)):
                if idx == 0:
                    continue  # Never reject the reference image
                if combined_score[idx] < self.config.rej_thresh:
                    reject_flags[idx] = True
            info_print(f"[INFO] Rejection threshold: {self.config.rej_thresh:.4f}")

        rejected_image_indices = [idx for idx, flag in enumerate(reject_flags) if flag and idx != 0]
        info_print(f"[INFO] Integrated rejection: {rejected_image_indices}")

        # Write single image_list.txt with per-image scores
        if image_paths is not None:
            n_kept = num_images_total - len(rejected_image_indices)
            layer_num = attn_layers[0] if attn_layers else -1
            out_path = self.pair_out_dir / "image_list.txt"
            with open(out_path, "w") as f:
                f.write(f"tau: {self.config.rej_thresh}  layer: {layer_num}  attn_weight: {self.config.attn_a}\n")
                f.write(f"kept: {n_kept}/{num_images_total}\n\n")
                for idx in range(num_images_total):
                    status = "[PRUNE]" if reject_flags[idx] else "[KEEP]"
                    img_path = str(image_paths[idx]) if idx < len(image_paths) else "unknown"
                    f.write(
                        f"  view {idx:3d} (local {idx:3d}): "
                        f"score={per_image_scores[idx].item():.4f}  "
                        f"cos={per_image_cos[idx].item():.4f}  "
                        f"attn={per_image_attn[idx].item():.4f}  "
                        f"{status}  {img_path}\n"
                    )
            info_print(f"[INFO] Image list written: {out_path}")

        # Cleanup
        del avg_maps2d_sum, avg_counts
        try:
            q_out.clear()
            k_out.clear()
            hidden_states.clear()
        except Exception:
            pass
        del q_out, k_out, hidden_states
        safe_empty_cache()

        # Second forward pass with surviving images
        if len(rejected_image_indices) > 0:
            total_N = num_images_total
            survivors = [i for i in range(total_N) if i not in rejected_image_indices]

            if len(survivors) == 0:
                info_print("[INFO] All images rejected; skipping second forward.")
            elif len(survivors) == total_N:
                pass
            else:
                try:
                    del images_device
                except Exception:
                    pass
                safe_empty_cache()

                # Subset images (B, N, C, H, W)
                images_subset_cpu = images[:, survivors, ...]
                images_subset = images_subset_cpu.to(device=device, dtype=self.amp_dtype, non_blocking=non_blocking)

                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                            predictions_survive = self.model(images_subset)
                    else:
                        predictions_survive = self.model(images_subset)

                prediction_save_path = self.pair_out_dir / f"predictions_survived.npz"
                save_predictions = {k: v.float().cpu().numpy() for k, v in predictions_survive.items() if torch.is_tensor(v)}
                # Extract depth from local_points (z coordinate)
                save_predictions["depth"] = predictions_survive["local_points"][..., 2].float().cpu().numpy()
                # Add aliases for compatibility with visualization scripts
                save_predictions["world_points"] = save_predictions["points"]
                save_predictions["depth_conf"] = save_predictions["conf"].squeeze(-1)  # (B, N, H, W)
                np.savez(prediction_save_path, **save_predictions)

                predictions = predictions_survive
                try:
                    del images_subset, predictions_survive
                except Exception:
                    pass
                safe_empty_cache()
        else:
            prediction_save_path = self.pair_out_dir / f"predictions_survived.npz"
            save_predictions = {k: v.float().cpu().numpy() for k, v in predictions.items() if torch.is_tensor(v)}
            # Extract depth from local_points (z coordinate)
            save_predictions["depth"] = predictions["local_points"][..., 2].float().cpu().numpy()
            # Add aliases for compatibility with visualization scripts
            save_predictions["world_points"] = save_predictions["points"]
            save_predictions["depth_conf"] = save_predictions["conf"].squeeze(-1)  # (B, N, H, W)
            np.savez(prediction_save_path, **save_predictions)

        # Extract results
        camera_poses = predictions["camera_poses"].detach().cpu().float()  # (B, N, 4, 4)
        camera_poses = camera_poses.squeeze(0)  # (N, 4, 4)
        
        conf = predictions["conf"].detach().cpu().float()
        points = predictions["points"].detach().cpu().float()
        local_points = predictions["local_points"].detach().cpu().float()

        del predictions
        try:
            del images_device
        except Exception:
            pass
        safe_empty_cache()

        return camera_poses, points, conf, local_points, rejected_image_indices

    def run_demo(self) -> None:
        self.pair_out_dir = Path(self.config.exp_name)
        self.pair_out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_image_paths(self.config.image_dir)
        info_print(f"[INFO] Found {len(image_paths)} images in {self.config.image_dir}")

        # Load and preprocess images using Pi3's utility
        images_tensor = load_images_as_tensor(
            str(self.config.image_dir),
            interval=1,
            verbose=True
        )  # (N, C, H, W) - keep on CPU initially

        if len(image_paths) > self.config.max_images:
            info_print(f"[INFO] Randomly sampling {self.config.max_images} images from {len(image_paths)}")
            sampled_indices = sorted(random.sample(range(len(image_paths)), self.config.max_images))
            image_paths = [image_paths[i] for i in sampled_indices]
            images_tensor = images_tensor[sampled_indices]

        # Save preprocessed grid
        try:
            import matplotlib.pyplot as plt
            from torchvision.utils import make_grid
            images_cpu = images_tensor.detach().cpu()
            grid = make_grid(images_cpu, nrow=8, padding=2, pad_value=1.0)
            grid_np = grid.permute(1, 2, 0).numpy()
            plt.imsave(self.pair_out_dir / "preprocessed_grid.png", np.clip(grid_np, 0.0, 1.0))
        except Exception:
            pass

        if torch.device(self.device).type == "cuda" and images_tensor.device.type == "cpu":
            images_tensor = images_tensor.pin_memory()

        image_hw = tuple(int(dim) for dim in images_tensor.shape[-2:])

        camera_poses, points, conf, local_points, rejected_indices = self._forward_once(
            images_tensor, image_hw, torch.device(self.device), image_paths
        )

        # Save surviving images to a subdirectory for debugging
        survived_dir = self.pair_out_dir / "clean_images"
        survived_dir.mkdir(parents=True, exist_ok=True)
        
        total_images = len(image_paths)
        survivors = [i for i in range(total_images) if i not in rejected_indices]
        
        info_print(f"[INFO] Saving {len(survivors)} survived images to {survived_dir}")
        for idx in survivors:
            src_path = image_paths[idx]
            dst_path = survived_dir / src_path.name
            # Copy the original image
            shutil.copy2(src_path, dst_path)
        
        info_print(f"[INFO] Results saved to {self.pair_out_dir}")
        info_print(f"[INFO] Camera poses shape: {camera_poses.shape}")
        info_print(f"[INFO] Points shape: {points.shape}")
        info_print(f"[INFO] Rejected image indices: {rejected_indices}")

        # Generate point cloud similar to Pi3's example.py
        # Re-load survived images for color extraction
        survived_imgs = load_images_as_tensor(
            str(survived_dir),
            interval=1,
            verbose=False
        ).to(self.device)  # (N_survived, 3, H, W)
        
        # Run Pi3 on survived images only for clean reconstruction
        info_print(f"[INFO] Running Pi3 on {len(survivors)} survived images for reconstruction...")
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                res = self.model(survived_imgs[None])  # Add batch dimension
        
        # Process mask following Pi3's approach
        masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]  # (N, H, W)
        
        # Save point cloud
        ply_path = self.pair_out_dir / "reconstruction.ply"
        info_print(f"[INFO] Saving point cloud to {ply_path}")
        write_ply(
            res['points'][0][masks].cpu(),
            survived_imgs.permute(0, 2, 3, 1)[masks],  # (N, H, W, 3) colors
            str(ply_path)
        )
        info_print(f"[INFO] Point cloud saved successfully!")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run robust Pi3 on a directory of images.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to process.",
    )
    parser.add_argument("--exp-name", type=str, default="demo_result", help="Experiment name for output directory.")
    parser.add_argument("--attn_a", type=float, default=0.5, help="Attention weight.")
    parser.add_argument("--cos_a", type=float, default=0.5, help="Cosine similarity weight.")
    parser.add_argument("--rej-thresh", type=float, default=0.4, help="Rejection threshold.")
    parser.add_argument(
        "--max-images", type=int, default=400,
        help="Maximum number of images per sequence. If more are found, randomly sample this many.",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        image_dir=args.image_dir,
        exp_name=args.exp_name,
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        max_images=args.max_images,
    )


def main() -> None:
    config = parse_args()
    experiment = RobustPi3Experiment(config)
    experiment.run_demo()


if __name__ == "__main__":
    main()
