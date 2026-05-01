"""Profile robust_{vggt,pi3,mapanything} runtime on scannet sequences.

Sweeps every sequence under
    <data_root>/<noise_level>/scannet_30/<seq>/images/
for the requested backends and reports wall-time + a model.forward
breakdown isolated from data loading and disk I/O.

To keep timing focused on the model:
  * The backbone is loaded exactly ONCE per backend (re-used across
    all sequences via ``dataclasses.replace`` on ExperimentConfig).
  * Per-sequence image preprocessing is run *before* the timed region
    and cached; the loader inside the robust_* module is monkey-patched
    to return the cached tensor while ``run_demo`` is timed. Pi3's
    second loader call (on the survived_dir) still runs normally so
    the second forward sees the correct subset.
  * All disk writes in the timed region are suppressed:
      - Heavy: np.savez / *.ply (no-op'd entirely).
      - Lightweight: plt.imsave (preprocessed_grid.png), PIL Image.save
        (anchor.png, survived_*.png), open("image_list.txt", "w"),
        shutil.copy2 (survivor image copies). For Pi3 the recording
        copy2 stub also eliminates the second disk-load of survived
        images: the survivor subset is sliced directly from the
        pre-cached tensor.
  * ``model.forward`` is wrapped with a CUDA-synchronized timer that
    routes each call into one of two buckets:
      - ``pruning``       — forwards invoked inside ``_forward_once``
                            (the first pass + optional survivor pass).
      - ``reconstruction``— forwards invoked elsewhere inside
                            ``run_demo``. Pi3 does an extra final
                            forward on the surviving images to build
                            the output point cloud (robust_pi3.py:608),
                            so this bucket is non-empty for Pi3 and
                            empty for VGGT / MapAnything.

Per-sequence record fields:
    wall_seconds                end-to-end ``run_demo`` wall (sync'd)
    forward_total               sum of every ``model.forward`` call
    forward_calls               total number of forward passes
    pruning_forward_total       sum of pruning-only forwards
    pruning_forward_calls       1 (no rejection) or 2 (with rejection)
    recon_forward_total         sum of reconstruction-only forwards
                                (Pi3-only; 0.0 for VGGT/MapAnything)
    recon_forward_calls         0 for VGGT/MapAnything; 1 for Pi3
    forward_per_call            list of (label, seconds) tuples, in order
    overhead                    wall_seconds - forward_total (CPU/GPU
                                work outside the backbone: hooks,
                                attn-map probe, softmax/normalize,
                                cleanup, etc.)
    precache_seconds            time spent pre-loading images outside
                                the timed region (NOT in wall_seconds)
    stages                      dict of cProfile-derived cumulative
                                seconds per stage label, matching
                                ``STAGE_LOOKUPS[model]``. CPU/wall as
                                seen by cProfile (no CUDA sync), so for
                                forward time prefer ``forward_total``.
                                Pass ``--no-cprofile`` to skip.

Usage:
    python view_pruning/profile_efficiency.py \\
        --models vggt pi3 mapanything \\
        --noise-levels clean low_noisy mid_noisy high_noisy \\
        --gpu 0
"""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import gc
import json
import logging
import os
import pstats
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_ROOT = Path(
    "/home/zhijiewu/Documents/Share/Datasets/MACV/Final_Benchmarks"
)
DATASET_SUBDIR = "scannet_30"
NOISE_LEVELS = ("clean", "low_noisy", "mid_noisy", "high_noisy")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MODELS = ("vggt", "pi3", "mapanything")


# ---------------------------------------------------------------------------
# cProfile stage aggregation (mirrors profile_macv_efficiency.py's approach).
#
# Each entry is ``(label, file_suffix, funcname)`` — pstats matches by
# filename suffix + exact funcname so unrelated overrides on every nn.Module
# don't get aggregated. Cumulative time (``ct``) is summed across all
# matching call sites. Numbers are CPU/wall as recorded by cProfile and do
# NOT include CUDA-side waits unless the function explicitly synchronizes;
# treat ``forward`` here as approximate and rely on ``forward_total`` from
# the direct CUDA-synced wrapper for authoritative GPU time.
# ---------------------------------------------------------------------------

STAGE_LOOKUPS: Dict[str, List[Tuple[str, str, str]]] = {
    "vggt": [
        ("_forward_once",                  "robust_vggt.py",                 "_forward_once"),
        ("  ↳ backbone.forward",           "vggt/models/vggt.py",            "forward"),
        ("  ↳ aggregator.forward",         "vggt/models/aggregator.py",      "forward"),
        ("  ↳ pose_encoding_to_extri_intri","vggt/utils/pose_enc.py",        "pose_encoding_to_extri_intri"),
        ("_save_ply",                      "robust_vggt.py",                 "_save_ply"),
        ("  ↳ unproject_depth_map_to_point_map", "vggt/utils/geometry.py",   "unproject_depth_map_to_point_map"),
        ("load_and_preprocess_images",     "vggt/utils/load_fn.py",          "load_and_preprocess_images"),
    ],
    "pi3": [
        ("_forward_once",                  "robust_pi3.py",                  "_forward_once"),
        ("  ↳ backbone.forward",           "pi3/models/pi3.py",              "forward"),
        ("load_images_as_tensor",          "pi3/utils/basic.py",             "load_images_as_tensor"),
        ("write_ply",                      "pi3/utils/basic.py",             "write_ply"),
        ("depth_edge",                     "pi3/utils/geometry.py",          "depth_edge"),
    ],
    "mapanything": [
        ("_forward_once",                  "robust_mapanything.py",          "_forward_once"),
        ("  ↳ backbone.forward",           "mapanything/models/mapanything/model.py", "forward"),
        ("_load_and_prepare_views",        "robust_mapanything.py",          "_load_and_prepare_views"),
        ("_load_raw_images",               "robust_mapanything.py",          "_load_raw_images"),
        ("_save_predictions_npz",          "robust_mapanything.py",          "_save_predictions_npz"),
        ("load_images",                    "mapanything/utils/image.py",     "load_images"),
    ],
}


def _extract_stages(stats: pstats.Stats,
                    lookups: List[Tuple[str, str, str]]) -> Dict[str, float]:
    """Sum cumulative time for each ``(file_suffix, funcname)`` pair."""
    out: Dict[str, float] = {}
    raw = stats.stats  # type: ignore[attr-defined]
    for label, file_suffix, funcname in lookups:
        cum = 0.0
        for (filename, _line, fname), (_cc, _nc, _tt, ct, _) in raw.items():
            if fname == funcname and filename.endswith(file_suffix):
                cum += ct
        out[label] = cum
    return out


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def _list_sequences(data_root: Path, noise_level: str) -> List[Path]:
    base = data_root / noise_level / DATASET_SUBDIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def _image_dir_of(seq_dir: Path) -> Path:
    return seq_dir / "images" if (seq_dir / "images").is_dir() else seq_dir


def _list_images(seq_dir: Path) -> List[Path]:
    return sorted(
        p for p in _image_dir_of(seq_dir).iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class _ForwardTimer:
    """Wrap ``module.forward`` to accumulate CUDA-synchronized wall time,
    routing each call into a label bucket selected via ``label()``.

    Default bucket is ``"reconstruction"`` so that anything we don't
    explicitly mark counts as reconstruction-side. Inside
    ``_forward_once`` we open a ``label("pruning")`` scope so the first
    + survivor passes go into the pruning bucket.
    """

    def __init__(self, module, default_label: str = "reconstruction"):
        import torch
        self.torch = torch
        self.module = module
        self._orig: Optional[Callable] = None
        self._current_label = default_label
        self.calls: List[Tuple[str, float]] = []  # (label, seconds), in order

    def __enter__(self) -> "_ForwardTimer":
        torch = self.torch
        orig = self.module.forward
        self._orig = orig
        self_ref = self

        def wrapped(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                self_ref.calls.append((self_ref._current_label, dt))

        self.module.forward = wrapped
        return self

    def __exit__(self, *_):
        self.module.forward = self._orig

    @contextlib.contextmanager
    def label(self, name: str):
        prev = self._current_label
        self._current_label = name
        try:
            yield
        finally:
            self._current_label = prev

    def totals(self) -> Dict[str, Tuple[float, int]]:
        """Return {label: (total_seconds, n_calls)} for all observed labels."""
        out: Dict[str, Tuple[float, int]] = {}
        for label, dt in self.calls:
            tot, n = out.get(label, (0.0, 0))
            out[label] = (tot + dt, n + 1)
        return out


@contextlib.contextmanager
def _patch_attrs(patches: List[Tuple[Any, str, Any]]):
    saved: List[Tuple[Any, str, Any]] = []
    try:
        for obj, attr, replacement in patches:
            saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, replacement)
        yield
    finally:
        for obj, attr, orig in saved:
            setattr(obj, attr, orig)


def _no_op(*_a, **_kw):
    return None


def _silence_module_prints(mod) -> None:
    """Suppress verbose per-image/per-view terminal output from the robust_*
    modules and vggt's load_fn, following profile_macv_efficiency.py's approach."""
    try:
        mod.info_print = lambda _: None
    except Exception:
        pass
    try:
        import vggt.utils.load_fn as _load_fn
        _load_fn.info_print = lambda _: None
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Per-backend factories
# ---------------------------------------------------------------------------

def _build_vggt(args):
    import robust_vggt
    cfg = robust_vggt.ExperimentConfig(
        image_dir=Path("."),
        preprocess_mode=args.preprocess_mode,
        exp_name="placeholder",
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        use_point_map=args.use_point_map,
        conf_threshold_pct=args.conf_threshold_pct,
    )
    exp = robust_vggt.RobustVGGTExperiment(cfg)
    return robust_vggt, exp


def _build_pi3(args):
    import robust_pi3
    cfg = robust_pi3.ExperimentConfig(
        image_dir=Path("."),
        exp_name="placeholder",
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        max_images=args.max_images,
    )
    exp = robust_pi3.RobustPi3Experiment(cfg)
    return robust_pi3, exp


def _build_mapanything(args):
    import robust_mapanything
    cfg = robust_mapanything.ExperimentConfig(
        image_dir=Path("."),
        exp_name="placeholder",
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        attn_layer=args.attn_layer,
        max_images=args.max_images,
    )
    exp = robust_mapanything.RobustMapAnythingExperiment(cfg)
    return robust_mapanything, exp


BUILDERS: Dict[str, Callable] = {
    "vggt": _build_vggt,
    "pi3": _build_pi3,
    "mapanything": _build_mapanything,
}


# ---------------------------------------------------------------------------
# Patch construction (per sequence): pre-cache loaders + stub heavy writes
# ---------------------------------------------------------------------------

def _make_patches(
    model: str,
    mod,
    exp,
    image_paths: List[Path],
    image_dir: Path,
) -> Tuple[List[Tuple[Any, str, Any]], float]:
    """Return (patch_list, precache_seconds).

    ``patch_list`` is meant to be applied as a context via ``_patch_attrs``
    around the timed ``run_demo`` call. Pre-caching is performed eagerly here.
    """
    import builtins as _builtins
    import numpy as np
    import shutil as _shutil

    patches: List[Tuple[Any, str, Any]] = []
    precache_t = 0.0

    # Stub the heavy disk writes. ``np.savez`` lives on the numpy module
    # object; patching there suffices because ``robust_*`` import ``numpy as np``
    # (sharing the same module object).
    patches.append((np, "savez", _no_op))
    patches.append((np, "savez_compressed", _no_op))

    # Suppress lightweight PNG saves (preprocessed_grid.png, anchor.png,
    # survived_*.png) that add measurable I/O inside the timed region.
    try:
        import matplotlib.pyplot as _plt
        patches.append((_plt, "imsave", _no_op))
    except ImportError:
        pass
    try:
        import PIL.Image as _pil
        patches.append((_pil.Image, "save", _no_op))
    except ImportError:
        pass

    # Suppress image_list.txt writes: intercept builtins.open so that any
    # write-mode open of a path containing "image_list.txt" goes to /dev/null.
    # All modules that do not locally rebind ``open`` share this builtin.
    _orig_open = _builtins.open

    def _filtered_open(file, mode="r", *a, **kw):
        try:
            if "image_list.txt" in str(file) and "w" in str(mode):
                return _orig_open(os.devnull, mode, *a, **kw)
        except Exception:
            pass
        return _orig_open(file, mode, *a, **kw)

    patches.append((_builtins, "open", _filtered_open))

    if model == "vggt":
        patches.append((_shutil, "copy2", _no_op))
        patches.append((mod, "save_ply", _no_op))
        t0 = time.perf_counter()
        cached = mod.load_and_preprocess_images(
            [str(p) for p in image_paths], mode=exp.config.preprocess_mode
        )
        precache_t = time.perf_counter() - t0

        def fake_loader(_paths, mode="crop"):
            return cached

        patches.append((mod, "load_and_preprocess_images", fake_loader))

    elif model == "pi3":
        patches.append((mod, "write_ply", _no_op))
        orig_loader = mod.load_images_as_tensor
        t0 = time.perf_counter()
        cached = orig_loader(str(image_dir), interval=1, verbose=False)
        precache_t = time.perf_counter() - t0

        target_dir = Path(image_dir).resolve()

        # Recording copy2 stub: capture which source images survive so we can
        # reconstruct the survived-subset tensor from ``cached`` without a
        # second disk read. Pi3 copies survivors to survived_dir *before*
        # calling load_images_as_tensor(survived_dir) at robust_pi3.py:598.
        copied_srcs: List[Path] = []

        def _recording_copy2(src, *_):
            copied_srcs.append(Path(src))

        patches.append((_shutil, "copy2", _recording_copy2))

        _src_to_idx: Dict[Path, int] = {
            Path(p).resolve(): i for i, p in enumerate(image_paths)
        }

        def fake_loader(path, interval=1, PIXEL_LIMIT=255000, verbose=True):
            try:
                resolved = Path(path).resolve()
            except OSError:
                resolved = None
            if resolved == target_dir:
                return cached
            # Survived dir: slice the pre-cached tensor using the sources
            # captured by the recording copy2 stub instead of reading disk.
            if copied_srcs:
                import torch as _torch
                survivor_indices = sorted(
                    _src_to_idx[p.resolve()] for p in copied_srcs
                    if p.resolve() in _src_to_idx
                )
                if survivor_indices:
                    return cached[_torch.tensor(survivor_indices)]
            # Fallback if recording failed (should not occur normally)
            return orig_loader(path, interval=interval,
                               PIXEL_LIMIT=PIXEL_LIMIT, verbose=False)

        patches.append((mod, "load_images_as_tensor", fake_loader))

    elif model == "mapanything":
        patches.append((_shutil, "copy2", _no_op))
        from mapanything.utils import image as _ma_image
        from mapanything.utils.image import load_images as _orig_load_images

        t0 = time.perf_counter()
        cached_views = _orig_load_images(
            [str(p) for p in image_paths],
            norm_type=exp.data_norm_type,
            patch_size=exp.patch_size,
        )
        # ``_load_raw_images`` opens each file with PIL, resizes to the
        # patch-aligned (H, W) and stacks — also pure data I/O. Pre-cache
        # it now so the timed wall doesn't include it. ``image_hw`` is
        # derived from cached views ((C, H, W) layout, indices 2 and 3).
        cached_image_hw = (
            cached_views[0]["img"].shape[2],
            cached_views[0]["img"].shape[3],
        )
        cached_raw_images = exp._load_raw_images(image_paths, cached_image_hw)
        precache_t = time.perf_counter() - t0

        def fake_load_images(*_a, **_kw):
            return cached_views

        def fake_load_raw_images(*_a, **_kw):
            return cached_raw_images

        patches.append((_ma_image, "load_images", fake_load_images))
        # ``_load_raw_images`` is a staticmethod on the class; patch the
        # instance so we don't perturb other experiments that share the
        # class. Instance attr shadows the class attr during the timed run.
        patches.append((exp, "_load_raw_images", fake_load_raw_images))

        # ``write_ply`` is imported lazily inside ``run_demo`` from
        # ``export_pointcloud_from_npz``; stub the source module symbol.
        try:
            import export_pointcloud_from_npz as _exp_ply
            patches.append((_exp_ply, "write_ply", _no_op))
        except ImportError:
            pass

    else:
        raise ValueError(model)

    return patches, precache_t


# ---------------------------------------------------------------------------
# Per-sequence runner
# ---------------------------------------------------------------------------

def _run_sequence(model: str, mod, exp, seq_dir: Path, out_root: Path,
                  noise_level: str, args) -> Optional[dict]:
    import torch

    image_paths = _list_images(seq_dir)
    if not image_paths:
        return None
    image_dir = _image_dir_of(seq_dir)

    # scannet_30 has 30 images per sequence — well below max_images defaults,
    # so the ``random.sample`` path inside Pi3/MapAnything won't trigger.

    out_dir = out_root / model / noise_level / seq_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    exp.config = replace(exp.config, image_dir=image_dir, exp_name=str(out_dir))

    patches, precache_t = _make_patches(model, mod, exp, image_paths, image_dir)

    timer = _ForwardTimer(exp.model, default_label="reconstruction")

    # Wrap ``_forward_once`` so every backbone call inside it lands in the
    # "pruning" bucket. Anything else inside ``run_demo`` (Pi3's
    # reconstruction forward at robust_pi3.py:608) stays "reconstruction".
    orig_forward_once = exp._forward_once

    def _forward_once_pruning_scope(*a, **kw):
        with timer.label("pruning"):
            return orig_forward_once(*a, **kw)

    exp._forward_once = _forward_once_pruning_scope

    pr: Optional[cProfile.Profile] = None
    if not args.no_cprofile:
        pr = cProfile.Profile()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    try:
        with timer, _patch_attrs(patches):
            if pr is not None:
                pr.enable()
            try:
                exp.run_demo()
            finally:
                if pr is not None:
                    pr.disable()
    except Exception as exc:
        traceback.print_exc()
        exp._forward_once = orig_forward_once
        return {
            "sequence": seq_dir.name,
            "n_images": len(image_paths),
            "error": repr(exc),
            "wall_seconds": float("nan"),
            "precache_seconds": precache_t,
            "forward_total": float("nan"),
            "forward_calls": 0,
            "pruning_forward_total": float("nan"),
            "pruning_forward_calls": 0,
            "recon_forward_total": float("nan"),
            "recon_forward_calls": 0,
            "forward_per_call": [],
            "overhead": float("nan"),
            "stages": {},
        }
    finally:
        exp._forward_once = orig_forward_once

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t_start

    stages: Dict[str, float] = {}
    if pr is not None:
        stats_path = out_dir / "run.pstats"
        pr.dump_stats(str(stats_path))
        try:
            stats = pstats.Stats(str(stats_path))
            stages = _extract_stages(stats, STAGE_LOOKUPS[model])
        except Exception as e:
            print(f"  [WARN] pstats parse failed: {e!r}")

    totals = timer.totals()
    pruning_t, pruning_n = totals.get("pruning", (0.0, 0))
    recon_t, recon_n = totals.get("reconstruction", (0.0, 0))
    forward_total = sum(t for t, _ in totals.values())
    forward_calls = sum(n for _, n in totals.values())

    return {
        "sequence": seq_dir.name,
        "n_images": len(image_paths),
        "wall_seconds": wall,
        "precache_seconds": precache_t,
        "forward_total": forward_total,
        "forward_calls": forward_calls,
        "pruning_forward_total": pruning_t,
        "pruning_forward_calls": pruning_n,
        "recon_forward_total": recon_t,
        "recon_forward_calls": recon_n,
        "forward_per_call": [
            {"label": label, "seconds": dt} for label, dt in timer.calls
        ],
        "overhead": wall - forward_total,
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Aggregation / pretty-printing
# ---------------------------------------------------------------------------

def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _print_noise_summary(model: str, noise_level: str,
                         records: List[dict]) -> None:
    ok = [r for r in records if "error" not in r]
    if not ok:
        print(f"\n[{model}/{noise_level}] no valid runs ({len(records)} attempts)")
        return

    s_wall = _summarize([r["wall_seconds"] for r in ok])
    s_fwd = _summarize([r["forward_total"] for r in ok])
    s_prune = _summarize([r["pruning_forward_total"] for r in ok])
    s_recon = _summarize([r["recon_forward_total"] for r in ok])
    s_over = _summarize([r["overhead"] for r in ok])
    s_pre = _summarize([r["precache_seconds"] for r in ok])

    mean_prune_calls = statistics.fmean(r["pruning_forward_calls"] for r in ok)
    mean_recon_calls = statistics.fmean(r["recon_forward_calls"] for r in ok)

    mean_wall = s_wall["mean"] or 1e-9
    print(f"\n[{model}/{noise_level}] {len(ok)} sequences (mean #imgs="
          f"{statistics.fmean(r['n_images'] for r in ok):.1f})")
    print(f"  wall                mean={s_wall['mean']:7.3f}s  med={s_wall['median']:7.3f}s  "
          f"min={s_wall['min']:7.3f}s  max={s_wall['max']:7.3f}s")
    print(f"  forward (all)       mean={s_fwd['mean']:7.3f}s  med={s_fwd['median']:7.3f}s  "
          f"min={s_fwd['min']:7.3f}s  max={s_fwd['max']:7.3f}s  "
          f"({100 * s_fwd['mean'] / mean_wall:5.1f}% of wall)")
    print(f"  ↳ pruning           mean={s_prune['mean']:7.3f}s  med={s_prune['median']:7.3f}s  "
          f"(avg {mean_prune_calls:.2f} calls/seq)")
    print(f"  ↳ reconstruction    mean={s_recon['mean']:7.3f}s  med={s_recon['median']:7.3f}s  "
          f"(avg {mean_recon_calls:.2f} calls/seq)")
    print(f"  overhead            mean={s_over['mean']:7.3f}s  med={s_over['median']:7.3f}s  "
          f"min={s_over['min']:7.3f}s  max={s_over['max']:7.3f}s")
    print(f"  precache(*)         mean={s_pre['mean']:7.3f}s  med={s_pre['median']:7.3f}s  "
          f"(NOT in wall)")

    # cProfile stage breakdown — average each labelled stage across sequences.
    stage_dicts = [r.get("stages") or {} for r in ok]
    all_labels: List[str] = []
    for d in stage_dicts:
        for k in d:
            if k not in all_labels:
                all_labels.append(k)
    if all_labels:
        label_w = max(len(s) for s in all_labels)
        print(f"  cProfile stages (cumulative, CPU/wall — GPU work not"
              f" sync'd, may undercount):")
        print(f"  {'stage'.ljust(label_w)}      mean       %wall")
        for lbl in all_labels:
            cums = [d.get(lbl, 0.0) for d in stage_dicts]
            m = statistics.fmean(cums) if cums else 0.0
            print(f"  {lbl.ljust(label_w)}  {m:7.3f}s   {100 * m / mean_wall:5.1f}%")


# ---------------------------------------------------------------------------
# Backend driver
# ---------------------------------------------------------------------------

def _process_backend(model: str, args, out_root: Path) -> dict:
    import torch

    print(f"\n{'=' * 72}")
    print(f"=== Backend: robust_{model} ===")
    print(f"{'=' * 72}")

    t0 = time.perf_counter()
    mod, exp = BUILDERS[model](args)
    print(f"[{model}] loaded model in {time.perf_counter() - t0:.2f}s")
    _silence_module_prints(mod)

    backend_results: dict = {"model": model, "noise_levels": {}}
    for noise_level in args.noise_levels:
        seqs = _list_sequences(args.data_root, noise_level)
        if args.max_sequences is not None:
            seqs = seqs[: args.max_sequences]
        if not seqs:
            print(f"[{model}/{noise_level}] no sequences in "
                  f"{args.data_root / noise_level / DATASET_SUBDIR}")
            backend_results["noise_levels"][noise_level] = {"sequences": []}
            continue

        records: List[dict] = []
        for i, seq_dir in enumerate(seqs):
            print(f"[{model}/{noise_level}] ({i + 1}/{len(seqs)}) {seq_dir.name} ...",
                  flush=True)
            rec = _run_sequence(model, mod, exp, seq_dir, out_root, noise_level, args)
            if rec is None:
                print("  (no images, skipped)")
                continue
            records.append(rec)
            if "error" in rec:
                print(f"  FAILED: {rec['error']}")
            else:
                print(f"  wall={rec['wall_seconds']:.3f}s  "
                      f"forward={rec['forward_total']:.3f}s "
                      f"[prune={rec['pruning_forward_total']:.3f}s × {rec['pruning_forward_calls']}, "
                      f"recon={rec['recon_forward_total']:.3f}s × {rec['recon_forward_calls']}]  "
                      f"overhead={rec['overhead']:.3f}s")

            # Free intermediate caches so memory doesn't accumulate.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _print_noise_summary(model, noise_level, records)
        ok_recs = [r for r in records if "error" not in r]
        backend_results["noise_levels"][noise_level] = {
            "sequences": records,
            "summary_wall_seconds": _summarize(
                [r["wall_seconds"] for r in ok_recs]
            ),
            "summary_forward_seconds": _summarize(
                [r["forward_total"] for r in ok_recs]
            ),
            "summary_pruning_forward_seconds": _summarize(
                [r["pruning_forward_total"] for r in ok_recs]
            ),
            "summary_recon_forward_seconds": _summarize(
                [r["recon_forward_total"] for r in ok_recs]
            ),
            "summary_overhead_seconds": _summarize(
                [r["overhead"] for r in ok_recs]
            ),
        }

    out_dir_model = out_root / model
    out_dir_model.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir_model / "summary.json"
    with summary_path.open("w") as f:
        json.dump(backend_results, f, indent=2)
    print(f"\n[{model}] summary written to {summary_path}")

    del exp, mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return backend_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--noise-levels", nargs="+", default=list(NOISE_LEVELS),
                    choices=list(NOISE_LEVELS))
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--gpu", type=str, default="0",
                    help="GPU index visible to HIP/CUDA. 'cpu' to run on CPU.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where per-backend summary JSONs go. Default: a fresh temp dir.")
    ap.add_argument("--max-sequences", type=int, default=None,
                    help="Optional cap per (model, noise_level) for quick smoke tests.")
    ap.add_argument("--no-cprofile", action="store_true",
                    help="Disable cProfile stage aggregation. Direct CUDA-synced "
                         "forward timing is unaffected; this only skips the "
                         "per-sequence pstats dump and stage breakdown.")

    # Shared scoring args (must match defaults used in run_noisy.py).
    ap.add_argument("--attn-a", type=float, default=0.5)
    ap.add_argument("--cos-a", type=float, default=0.5)
    ap.add_argument("--rej-thresh", type=float, default=0.4)

    # VGGT-only
    ap.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop")
    ap.add_argument("--no-point-map", dest="use_point_map", action="store_false")
    ap.set_defaults(use_point_map=True)
    ap.add_argument("--conf-threshold-pct", type=float, default=30.0)

    # Pi3/MapAnything
    ap.add_argument("--max-images", type=int, default=400)
    # MapAnything-only
    ap.add_argument("--attn-layer", type=int, default=14)

    args = ap.parse_args()

    # GPU pinning must happen before torch is imported anywhere.
    if args.gpu.lower() == "cpu":
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(var, None)
    else:
        os.environ["HIP_VISIBLE_DEVICES"] = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    if args.out_dir is None:
        args.out_dir = Path(tempfile.mkdtemp(prefix="robust_efficiency_"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    print(f"[harness] data_root:    {args.data_root}")
    print(f"[harness] dataset:      {DATASET_SUBDIR}")
    print(f"[harness] models:       {args.models}")
    print(f"[harness] noise_levels: {args.noise_levels}")
    print(f"[harness] device:       {'cpu' if args.gpu.lower() == 'cpu' else 'cuda:' + args.gpu}")
    print(f"[harness] out_dir:      {args.out_dir}")

    overall: dict = {"runs": {}}
    for model in args.models:
        overall["runs"][model] = _process_backend(model, args, args.out_dir)

    overall_path = args.out_dir / "summary_all.json"
    with overall_path.open("w") as f:
        json.dump(overall, f, indent=2)
    print(f"\n[harness] combined summary: {overall_path}")

    # Cross-backend comparison: per-noise-level mean forward (s).
    print(f"\n{'=' * 72}")
    print("=== Cross-backend mean per-sequence times (s) ===")
    print(f"  metric       noise        " + "  ".join(f"{m:>14s}" for m in args.models))
    for metric, key in (
        ("wall      ", "summary_wall_seconds"),
        ("forward   ", "summary_forward_seconds"),
        ("  pruning ", "summary_pruning_forward_seconds"),
        ("  recon   ", "summary_recon_forward_seconds"),
        ("overhead  ", "summary_overhead_seconds"),
    ):
        for nl in args.noise_levels:
            row = f"  {metric}  {nl:<10s} "
            for m in args.models:
                stats = (overall["runs"]
                         .get(m, {})
                         .get("noise_levels", {})
                         .get(nl, {})
                         .get(key, {}))
                mean = stats.get("mean", float("nan"))
                n = stats.get("n", 0)
                row += f"  {mean:8.3f} (n={n:>2d})"
            print(row)


if __name__ == "__main__":
    main()
