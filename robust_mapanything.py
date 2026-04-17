#!/usr/bin/env python3
"""
Robust MapAnything: Adapting the RobustVGGT outlier view rejection methodology
to the MapAnything feed-forward 3D reconstruction model.

This implementation mirrors the mathematical logic found in RobustVGGT's _forward_once():
  1. Hook into the last global attention layer to capture Q and K after normalization.
  2. Hook into the block output to extract per-view features for cosine similarity.
  3. Compute combined attention + cosine similarity scores per view.
  4. Reject views whose combined score falls below a threshold.
  5. Re-run the model with only the surviving views.

Architecture mapping:
  VGGT (24 global_blocks)            → MapAnything (16 alternating self_attention_blocks)
  global_blocks[23]                  → self_attention_blocks[14] (last even = last global layer)
  blk.attn.q_norm / blk.attn.k_norm → blk.attn.q_norm / blk.attn.k_norm
  patch_start_idx = 5                → patch_start_idx = 0 (no camera/register tokens in info_sharing)
  aggregated_tokens_list[23]         → block output after layer 14
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]

# Also expose the project root so local modules like export_pointcloud_from_npz
# (sibling of this file) are importable when this script is imported from
# elsewhere (e.g. the view_pruning batch runner).
sys.path.insert(0, str(FILE_PATH.parent))


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


# Add MapAnything to path — override with MAPANYTHING_ROOT env var if needed
MAPANYTHING_ROOT = _resolve_external_repo(
    "MAPANYTHING_ROOT",
    [
        REPO_ROOT / "map-anything",
        REPO_ROOT / "Robust-X" / "map-anything",
        REPO_ROOT.parent / "map-anything",
        REPO_ROOT.parent / "Robust-X" / "map-anything",
    ],
)
sys.path.insert(0, str(MAPANYTHING_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    image_dir: Path
    exp_name: str = "demo_result_mapanything"
    attn_a: float = 0.5
    cos_a: float = 0.5
    rej_thresh: float = 0.4
    # The last global attention layer index in MapAnything's 16-layer alternating transformer.
    # Even indices (0,2,4,...,14) are global attention; odd indices are frame-level.
    attn_layer: int = 14
    max_images: int = 400


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def info_print(msg: str) -> None:
    print(f"{ANSI_GREEN}{msg}{ANSI_RESET}")


def safe_empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def free_cuda(*args):
    for arg in args:
        del arg
    safe_empty_cache()


def select_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


def list_image_paths(images_dir: Path) -> List[Path]:
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


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class RobustMapAnythingExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device, self.amp_dtype = select_device_and_dtype()

        info_print("[INFO] Loading MapAnything model...")

        from mapanything.models.mapanything import MapAnything
        self.model = MapAnything.from_pretrained("facebook/map-anything")
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Cache architecture constants
        self.patch_size = self.model.encoder.patch_size  # 14
        self.data_norm_type = self.model.encoder.data_norm_type

        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Save predictions in exporter-compatible format
    # ------------------------------------------------------------------

    @staticmethod
    def _save_predictions_npz(
        predictions: List[dict],
        save_path: Path,
        images: Optional[np.ndarray] = None,
    ) -> None:
        """Save MapAnything predictions in the same npz format as VGGT/Pi3.

        The exporter expects stacked arrays with keys ``world_points``
        (N, H, W, 3), ``depth`` (N, H, W), ``depth_conf`` (N, H, W),
        and ``images`` (N, 3, H, W) for RGB colors.
        MapAnything returns a list of per-view dicts with ``pts3d`` and
        ``conf`` tensors, so we stack them here.
        """
        pts3d_list: List[np.ndarray] = []
        conf_list: List[np.ndarray] = []

        for pred in predictions:
            if "pts3d" in pred and torch.is_tensor(pred["pts3d"]):
                pts = pred["pts3d"].float().cpu().numpy()
                # Handle (B, H, W, 3) → (H, W, 3)
                if pts.ndim == 4:
                    pts = pts[0]
                pts3d_list.append(pts)

            if "conf" in pred and torch.is_tensor(pred["conf"]):
                c = pred["conf"].float().cpu().numpy()
                # Handle (B, H, W) → (H, W)  or  (B, H, W, 1) → (H, W)
                if c.ndim == 4:
                    c = c[0]
                if c.ndim == 3 and c.shape[-1] == 1:
                    c = c[..., 0]
                # If still (B, H, W) with B=1, squeeze
                if c.ndim == 3 and c.shape[0] == 1:
                    c = c[0]
                conf_list.append(c)

        save_dict: Dict[str, np.ndarray] = {}

        if pts3d_list:
            world_points = np.stack(pts3d_list, axis=0)  # (N, H, W, 3)
            save_dict["world_points"] = world_points
            save_dict["depth"] = world_points[..., 2]     # (N, H, W)

        if conf_list:
            save_dict["depth_conf"] = np.stack(conf_list, axis=0)  # (N, H, W)

        if images is not None:
            save_dict["images"] = images  # (N, 3, H, W), float32 in [0, 1]

        np.savez(save_path, **save_dict)

    # ------------------------------------------------------------------
    # Raw image loading for RGB colors
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw_images(
        image_paths: List[Path], target_hw: Tuple[int, int]
    ) -> np.ndarray:
        """Load raw RGB images from disk and resize to target resolution.

        Returns:
            images: (N, 3, H, W) float32 array in [0, 1] range,
                    matching the format the exporter expects.
        """
        H, W = target_hw
        imgs: List[np.ndarray] = []
        for p in image_paths:
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            imgs.append(img_np.transpose(2, 0, 1))  # (3, H, W)
        return np.stack(imgs, axis=0)  # (N, 3, H, W)

    # ------------------------------------------------------------------
    # Image loading & view preparation
    # ------------------------------------------------------------------

    def _load_and_prepare_views(
        self, image_paths: List[Path]
    ) -> Tuple[List[dict], Tuple[int, int]]:
        """Load images and create MapAnything-compatible view dicts.

        Uses mapanything's load_images() with fixed_mapping resize (max ~518px)
        to match demo_colmap.py behaviour and avoid OOM on high-resolution inputs.

        Returns:
            views: List of view dictionaries ready for model.forward().
            image_hw: (H, W) of the patch-aligned images.
        """
        from mapanything.utils.image import load_images

        views = load_images(
            [str(p) for p in image_paths],
            norm_type=self.data_norm_type,
            patch_size=self.patch_size,
        )
        for v in views:
            v["img"] = v["img"].to(self.device)
            v["true_shape"] = torch.from_numpy(v["true_shape"]).view(1, 2).to(self.device)

        image_hw = (views[0]["img"].shape[2], views[0]["img"].shape[3])
        return views, image_hw

    # ------------------------------------------------------------------
    # Core: forward pass with hooks for robust scoring
    # ------------------------------------------------------------------

    def _forward_once(
        self,
        views: List[dict],
        image_hw: Tuple[int, int],
        raw_images: Optional[np.ndarray] = None,
        image_paths: Optional[List[Path]] = None,
    ) -> Tuple[List[dict], List[int]]:
        """Run a single forward pass with hooks to compute RobustVGGT-style metrics.

        This mirrors the logic of robust_vggt.py::_forward_once():
          - Hooks on q_norm / k_norm capture Q and K for attention weight computation.
          - A block-output hook captures features for cosine similarity.
          - Scores are combined and thresholded for view rejection.

        Args:
            views: List of MapAnything view dicts.
            image_hw: (H, W) tuple for the input images.
            raw_images: Optional (N, 3, H, W) float32 array of RGB images in [0, 1].

        Returns:
            predictions: Model output (list of dicts per view).
            rejected_image_indices: Indices of rejected views.
        """
        H, W = image_hw
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patch_tokens = h_patches * w_patches
        num_images_total = len(views)

        # ----- Determine the target layer -----
        # Even-indexed layers in the alternating transformer are global attention.
        target_layer = self.config.attn_layer
        assert target_layer % 2 == 0, (
            f"attn_layer must be an even index (global attention layer), got {target_layer}"
        )
        info_print(f"[INFO] Setting up hooks on info_sharing.self_attention_blocks[{target_layer}]")

        # ----- Register hooks -----
        q_out: Dict[int, Tensor] = {}
        k_out: Dict[int, Tensor] = {}
        hidden_states: Dict[int, Tensor] = {}
        handles: List[Any] = []

        def _make_hook(store_dict, idx):
            def _hook(_module, _inp, out):
                store_dict[idx] = out.detach()
            return _hook

        blk = self.model.info_sharing.self_attention_blocks[target_layer]
        # Hook Q and K after normalization (even if q_norm/k_norm are Identity,
        # the output still equals the normalized Q/K tensors)
        handles.append(blk.attn.q_norm.register_forward_hook(_make_hook(q_out, target_layer)))
        handles.append(blk.attn.k_norm.register_forward_hook(_make_hook(k_out, target_layer)))
        # Hook the full block output for cosine similarity features
        handles.append(blk.register_forward_hook(_make_hook(hidden_states, target_layer)))

        # ----- First forward pass -----
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=self.amp_dtype):
                predictions = self.model(views)

        # Save first-pass predictions in exporter-compatible format
        self._save_predictions_npz(
            predictions, self.pair_out_dir / "predictions_first_forward.npz",
            images=raw_images,
        )

        # ----- Compute cosine similarity from hidden states -----
        # The block output at the global layer has shape (B, V*H*W + T_additional, C).
        # V*H*W are the view patch tokens; T_additional = 1 for the scale token.
        cosine_similarities = []
        reject_flags = [False] * num_images_total

        if target_layer in hidden_states:
            feature = hidden_states[target_layer]  # (B, V*num_patch_tokens + 1, C)
            B_dim = feature.shape[0]
            C_dim = feature.shape[-1]

            # Trim off the scale token (last token)
            view_tokens = feature[:, :num_images_total * num_patch_tokens, :]

            # Reshape to (B, V, num_patch_tokens, C)
            view_tokens = view_tokens.reshape(B_dim, num_images_total, num_patch_tokens, C_dim)

            # Average over batch dimension
            layer_feat = view_tokens.mean(0).to(dtype=torch.float32)  # (V, num_patch_tokens, C)

            ref_feat = layer_feat[0:1, :, :]  # (1, num_patch_tokens, C)
            ref_feat_norm = F.normalize(ref_feat, p=2, dim=-1)
            layer_feat_norm = F.normalize(layer_feat, p=2, dim=-1)

            # Pixel-wise cosine similarity averaged over all spatial positions (Eq. 2-3 in paper)
            cos_sim = torch.einsum("bic,bjc->bij", layer_feat_norm, ref_feat_norm)  # (V, T, T)
            cos_sim_mean = cos_sim.mean(-1).mean(-1)  # (V,)
            cosine_similarities.append(cos_sim_mean)

        safe_empty_cache()

        # Remove hooks
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

        # ----- Compute attention weight scores from Q and K -----
        # Mirrors robust_vggt.py lines 305-318:
        #   q_first_image = Q[:, :, first_patch_start:first_patch_end, :]
        #   logits = einsum("bhqd,bhtd->bhqt", q_first_image, K_slice) * scale
        #   probs = softmax(logits, dim=-1)
        #   attn_first_image = probs.mean(dim=1).mean(dim=1)[0]  → (Tk,)
        global_mean_vals = []
        global_norms_list = []

        if target_layer in q_out and target_layer in k_out:
            Q = q_out[target_layer]  # (B, num_heads, seq_len, head_dim)
            K = k_out[target_layer]

            if Q.ndim == 4:
                B_dim, num_heads, seq_len, head_dim = Q.shape

                # tokens_per_image = num_patch_tokens (no register/camera tokens in info_sharing)
                tokens_per_image = num_patch_tokens
                num_images_in_seq = (seq_len - 1) // tokens_per_image  # -1 for scale token
                # Fallback if no scale token was appended
                if num_images_in_seq <= 0:
                    num_images_in_seq = seq_len // tokens_per_image

                if num_images_in_seq > 0:
                    # Extract Q for reference view (first image) patches
                    first_image_patch_start = 0
                    first_image_patch_end = num_patch_tokens
                    q_first_image = Q[:, :, first_image_patch_start:first_image_patch_end, :]

                    # K slice covering all view tokens
                    Tk = min(num_images_total, num_images_in_seq) * tokens_per_image
                    K_slice = K[:, :, :Tk, :]

                    scale = 1.0 / math.sqrt(float(head_dim))
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

                    # Extract per-image attention maps and compute global-normalized means
                    global_min = None
                    global_max = None
                    maps_2d = []
                    valid_indices = []

                    for img_idx in range(min(num_images_total, num_images_in_seq)):
                        start = img_idx * tokens_per_image
                        end = start + num_patch_tokens
                        if start >= attn_first_image.shape[-1]:
                            break
                        end = min(end, attn_first_image.shape[-1])

                        patch_attn = attn_first_image[start:end]
                        if patch_attn.numel() != num_patch_tokens:
                            continue

                        attn2d = patch_attn.view(h_patches, w_patches)
                        maps_2d.append(attn2d)
                        valid_indices.append(img_idx)

                        vmin = torch.min(attn2d)
                        vmax = torch.max(attn2d)
                        global_min = vmin if global_min is None else torch.minimum(global_min, vmin)
                        global_max = vmax if global_max is None else torch.maximum(global_max, vmax)

                    if len(maps_2d) > 0:
                        gmin = float(global_min.item())
                        gmax = float(global_max.item())
                        gden = (gmax - gmin) if gmax > gmin else 1.0

                        for col_idx, img_idx in enumerate(valid_indices):
                            attn2d_up = F.interpolate(
                                maps_2d[col_idx].unsqueeze(0).unsqueeze(0),
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            )[0, 0]
                            m_up_np = attn2d_up.detach().cpu().float().numpy()
                            global_norm = (m_up_np - gmin) / gden
                            global_mean_val = float(global_norm.mean())
                            global_mean_vals.append(global_mean_val)
                            global_norms_list.append(global_norm)

                    free_cuda(Q, K, q_first_image, K_slice, attn_first_image)
                    del Q, K

        # ----- Combine scores and reject (mirrors robust_vggt.py lines 400-417) -----
        per_image_scores = torch.zeros(num_images_total)
        per_image_cos = torch.zeros(num_images_total)
        per_image_attn = torch.zeros(num_images_total)
        if global_mean_vals and cosine_similarities:
            cos_sim = cosine_similarities[0]  # (V,)
            attn_val = torch.tensor(
                global_mean_vals[:num_images_total], device=cos_sim.device, dtype=cos_sim.dtype
            )

            info_print(f"[INFO] Raw cosine similarity values: {[f'{v:.4f}' for v in cos_sim.tolist()]}")
            info_print(f"[INFO] Raw attention values: {[f'{v:.4f}' for v in attn_val.tolist()]}")

            # Min-max normalize to [0, 1]
            cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-6)
            attn_val = (attn_val - attn_val.min()) / (attn_val.max() - attn_val.min() + 1e-6)

            # Weighted combination (Eq. 5 in paper)
            combined_score = self.config.attn_a * attn_val + self.config.cos_a * cos_sim

            info_print(f"[INFO] Normalized cosine similarity: {[f'{v:.4f}' for v in cos_sim.tolist()]}")
            info_print(f"[INFO] Normalized attention values: {[f'{v:.4f}' for v in attn_val.tolist()]}")
            info_print(f"[INFO] Combined scores: {[f'{v:.4f}' for v in combined_score.tolist()]}")

            n = len(combined_score)
            per_image_scores[:n] = combined_score.cpu()
            per_image_cos[:n] = cos_sim.cpu()
            per_image_attn[:n] = attn_val.cpu()

            # Reject views below threshold (Eq. 4/6 in paper); never reject reference (index 0)
            for idx in range(len(combined_score)):
                if idx == 0:
                    continue
                if combined_score[idx] < self.config.rej_thresh:
                    reject_flags[idx] = True

            info_print(f"[INFO] Rejection threshold: {self.config.rej_thresh:.4f}")

        rejected_image_indices = [idx for idx, flag in enumerate(reject_flags) if flag and idx != 0]
        info_print(f"[INFO] Integrated rejection: {rejected_image_indices}")

        if image_paths is not None:
            n_kept = num_images_total - len(rejected_image_indices)
            out_path = self.pair_out_dir / "image_list.txt"
            with open(out_path, "w") as f:
                f.write(f"tau: {self.config.rej_thresh}  layer: {target_layer}  attn_weight: {self.config.attn_a}\n")
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

        # Cleanup hook data
        q_out.clear()
        k_out.clear()
        hidden_states.clear()
        safe_empty_cache()

        # ----- Second forward pass with surviving views -----
        if len(rejected_image_indices) > 0:
            survivors = [i for i in range(num_images_total) if i not in rejected_image_indices]

            if len(survivors) == 0:
                info_print("[INFO] All images rejected; skipping second forward.")
            elif len(survivors) == num_images_total:
                pass
            else:
                safe_empty_cache()
                surviving_views = [views[i] for i in survivors]

                with torch.inference_mode():
                    with torch.autocast("cuda", enabled=True, dtype=self.amp_dtype):
                        predictions_survive = self.model(surviving_views)

                # Save surviving predictions in exporter-compatible format
                survived_images = raw_images[survivors] if raw_images is not None else None
                self._save_predictions_npz(
                    predictions_survive, self.pair_out_dir / "predictions_survived.npz",
                    images=survived_images,
                )

                predictions = predictions_survive
                safe_empty_cache()
        else:
            # All views survived; save again under survived name
            self._save_predictions_npz(
                predictions, self.pair_out_dir / "predictions_survived.npz",
                images=raw_images,
            )

        return predictions, rejected_image_indices

    # ------------------------------------------------------------------
    # Demo entry point
    # ------------------------------------------------------------------

    def run_demo(self) -> None:
        self.pair_out_dir = Path(self.config.exp_name)
        self.pair_out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_image_paths(self.config.image_dir)
        info_print(f"[INFO] Found {len(image_paths)} images in {self.config.image_dir}")

        if len(image_paths) > self.config.max_images:
            info_print(f"[INFO] Randomly sampling {self.config.max_images} images from {len(image_paths)}")
            image_paths = sorted(random.sample(image_paths, self.config.max_images), key=lambda p: p.name)

        views, image_hw = self._load_and_prepare_views(image_paths)
        info_print(f"[INFO] Image resolution (patch-aligned): {image_hw}")

        # Load raw RGB images for color extraction (matching prediction resolution)
        raw_images = self._load_raw_images(image_paths, image_hw)
        info_print(f"[INFO] Loaded raw images for colors: {raw_images.shape}")

        predictions, rejected_indices = self._forward_once(views, image_hw, raw_images=raw_images, image_paths=image_paths)

        # Save surviving / rejected image lists
        total_images = len(image_paths)
        survivors = [i for i in range(total_images) if i not in rejected_indices]

        survived_dir = self.pair_out_dir / "clean_images"
        survived_dir.mkdir(parents=True, exist_ok=True)
        info_print(f"[INFO] Saving {len(survivors)} survived images to {survived_dir}")
        for idx in survivors:
            shutil.copy2(image_paths[idx], survived_dir / image_paths[idx].name)

        info_print(f"[INFO] Results saved to {self.pair_out_dir}")
        info_print(f"[INFO] Rejected image indices: {rejected_indices}")

        # Extract per-view point clouds from surviving predictions and generate PLY
        if predictions:
            survived_images = raw_images[survivors]  # (N_survived, 3, H, W)
            H, W = image_hw
            all_points: List[np.ndarray] = []
            all_colors: List[np.ndarray] = []

            for view_idx, pred in enumerate(predictions):
                if "pts3d" not in pred:
                    continue
                pts = pred["pts3d"].float().cpu().numpy()
                if pts.ndim == 4:
                    pts = pts[0]  # (H, W, 3)
                info_print(f"[INFO] View {view_idx} pts3d shape: {pts.shape}")

                # Confidence mask
                if "conf" in pred and torch.is_tensor(pred["conf"]):
                    conf = pred["conf"].float().cpu().numpy()
                    if conf.ndim == 4:
                        conf = conf[0]
                    if conf.ndim == 3 and conf.shape[-1] == 1:
                        conf = conf[..., 0]
                    if conf.ndim == 3 and conf.shape[0] == 1:
                        conf = conf[0]
                    mask = conf > 0.5
                else:
                    mask = np.ones(pts.shape[:2], dtype=bool)

                # Depth / finite filter
                mask = mask & np.isfinite(pts).all(axis=-1)

                valid_pts = pts[mask]  # (M, 3)
                all_points.append(valid_pts)

                # RGB colors from raw images: (3, H, W) → (H, W, 3)
                if view_idx < len(survived_images):
                    img_hwc = survived_images[view_idx].transpose(1, 2, 0)  # (H, W, 3)
                    all_colors.append(img_hwc[mask])  # (M, 3)

            if all_points:
                all_points_arr = np.concatenate(all_points, axis=0)
                all_colors_arr = np.concatenate(all_colors, axis=0) if all_colors else None

                ply_path = self.pair_out_dir / "reconstruction.ply"
                from export_pointcloud_from_npz import write_ply as write_ply_func
                write_ply_func(ply_path, all_points_arr, all_colors_arr)
                info_print(f"[INFO] Point cloud saved to {ply_path} ({all_points_arr.shape[0]} points)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run Robust MapAnything on a directory of images."
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True,
        help="Directory containing images to process.",
    )
    parser.add_argument("--exp-name", type=str, default="demo_result_mapanything")
    parser.add_argument("--attn_a", type=float, default=0.5, help="Attention weight.")
    parser.add_argument("--cos_a", type=float, default=0.5, help="Cosine similarity weight.")
    parser.add_argument("--rej-thresh", type=float, default=0.4, help="Rejection threshold.")
    parser.add_argument(
        "--attn-layer", type=int, default=14,
        help="Index of the global attention layer to probe (must be even).",
    )
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
        attn_layer=args.attn_layer,
        max_images=args.max_images,
    )


def main() -> None:
    config = parse_args()
    experiment = RobustMapAnythingExperiment(config)
    experiment.run_demo()


if __name__ == "__main__":
    main()
