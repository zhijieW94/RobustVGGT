"""Benchmark — Peak GPU memory and wall-clock runtime vs number of frames.

Evaluates the robust view-pruning pipelines (robust_vggt / robust_pi3 /
robust_mapanything) on a single 7scenes scene at increasing collection
sizes (N50 → N100 → … → N350). For each (model, N-level) the following
are recorded:

  * Wall-clock runtime (seconds) of ``experiment.run_demo`` end-to-end
    (CUDA-synchronized at start and end).
  * Peak GPU memory allocated (GiB) during the same call, measured via
    ``torch.cuda.max_memory_allocated`` after resetting the peak counter.

Mirrors ``MACV_X/scripts/run_ablation_A5_memory_runtime.py`` but targets
the robust_X RobustVGGT/Pi3/MapAnything experiment classes. The backbone
is loaded exactly once per backend and reused across N-levels by mutating
``experiment.config`` via ``dataclasses.replace``.

Heavy disk writes inside ``run_demo`` (PLY exports, .npz dumps, anchor /
preprocessed PNGs, image_list.txt) are stubbed during the timed region so
the measurement is dominated by GPU work, not I/O. Outputs still go into
a fresh temp directory each run for safety.

Dataset layout assumed::

    <bench_root>/<N>/<scene>/images/

where <N> ∈ {N25, N50, N100, N150, N200, N250, N300, N350} and the default
scene is ``chess``.

Outputs
-------
``<out_root>/memory_runtime/<model>/``::

    per_run.json           Raw per-N measurements for this backend.
    summary.csv            n_level | scene | n_images | wall_s | peak_gpu_gib | error
    summary_by_N.csv       n_level | n_images | wall_s | peak_gpu_gib

``<out_root>/memory_runtime/``::

    summary_all.json       Combined machine-readable results for all backends.

A cross-backend table is printed at the end.

Usage
-----
    # All backends, all N-levels, chess scene (defaults)
    python view_pruning/run_ablation_memory_runtime.py

    # Restrict models or N-levels
    python view_pruning/run_ablation_memory_runtime.py \\
        --models vggt pi3 \\
        --n_levels N50 N100 N150 N200 N250

    # Change scene or benchmark root
    python view_pruning/run_ablation_memory_runtime.py \\
        --scene fire \\
        --bench_root /data/7scenes
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import csv
import gc
import json
import logging
import os
import shutil as _shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCH_ROOT = Path(
    "/nvmepool/zhijiewu/Datasets/Final_Benchmarks/7scenes"
)
DEFAULT_OUT_ROOT = Path(
    "/nvmepool/zhijiewu/results/MACV/7scenes/robust_x/ablation"
)

N_LEVELS = ("N25", "N50", "N100", "N150", "N200", "N250", "N300", "N350")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MODELS = ("vggt", "pi3", "mapanything")

logger = logging.getLogger("run_ablation_memory_runtime")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    model: str
    n_level: str
    scene: str
    n_images: int
    wall_seconds: float
    peak_gpu_gib: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-backend factories — load backbone once, return (mod, experiment)
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


BUILDERS: dict = {
    "vggt": _build_vggt,
    "pi3": _build_pi3,
    "mapanything": _build_mapanything,
}


def _silence_module_prints(mod) -> None:
    try:
        mod.info_print = lambda _msg: None
    except Exception:
        pass
    try:
        import vggt.utils.load_fn as _load_fn
        _load_fn.info_print = lambda _msg: None
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Heavy-write stubs (applied only during the timed region)
# ---------------------------------------------------------------------------

def _no_op(*_a, **_kw):
    return None


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


def _build_io_stubs(model: str, mod) -> List[Tuple[Any, str, Any]]:
    """Stub the heavy disk writes inside ``run_demo`` to keep the
    measurement focused on GPU work. Pre-cached loaders are NOT installed
    here — image loading is part of the pipeline we want to time, the same
    way the MACV A5 reference times ``run_macv_long_context`` end-to-end.
    """
    import numpy as np

    patches: List[Tuple[Any, str, Any]] = [
        (np, "savez", _no_op),
        (np, "savez_compressed", _no_op),
    ]

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

    # image_list.txt writes — redirect write-mode opens to /dev/null.
    _orig_open = builtins.open

    def _filtered_open(file, mode="r", *a, **kw):
        try:
            if "image_list.txt" in str(file) and "w" in str(mode):
                return _orig_open(os.devnull, mode, *a, **kw)
        except Exception:
            pass
        return _orig_open(file, mode, *a, **kw)

    patches.append((builtins, "open", _filtered_open))

    # PLY writers + survivor copies vary by backend.
    if model == "vggt":
        patches.append((_shutil, "copy2", _no_op))
        patches.append((mod, "save_ply", _no_op))
    elif model == "pi3":
        patches.append((_shutil, "copy2", _no_op))
        if hasattr(mod, "write_ply"):
            patches.append((mod, "write_ply", _no_op))
    elif model == "mapanything":
        patches.append((_shutil, "copy2", _no_op))
        try:
            import export_pointcloud_from_npz as _exp_ply
            patches.append((_exp_ply, "write_ply", _no_op))
        except ImportError:
            pass

    return patches


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _list_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTS
    )


def _discover(bench_root: Path, n_levels: List[str],
              scene: str) -> List[Tuple[str, Path, List[Path]]]:
    """Return list of (n_level, image_dir, image_paths) for the given scene."""
    entries: List[Tuple[str, Path, List[Path]]] = []
    for n_level in n_levels:
        images_dir = bench_root / n_level / scene / "images"
        if not images_dir.is_dir():
            logger.warning("Images dir not found: %s", images_dir)
            continue
        paths = _list_images(images_dir)
        if not paths:
            logger.warning("No images in %s", images_dir)
            continue
        entries.append((n_level, images_dir, paths))
    return entries


# ---------------------------------------------------------------------------
# Memory / timing
# ---------------------------------------------------------------------------

def _reset_peak_memory(device: str) -> None:
    if device == "cuda":
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _peak_memory_gib(device: str) -> float:
    if device == "cuda":
        try:
            import torch
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
    return float("nan")


def _sync(device: str) -> None:
    if device == "cuda":
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass


def _run_one(mod, exp, image_dir: Path, exp_name: str, model: str,
             device: str) -> Tuple[float, float]:
    """Return (wall_seconds, peak_gpu_gib) for one ``run_demo`` call."""
    exp.config = replace(exp.config, image_dir=image_dir, exp_name=exp_name)

    patches = _build_io_stubs(model, mod)

    _sync(device)
    _reset_peak_memory(device)

    t0 = time.perf_counter()
    with _patch_attrs(patches):
        exp.run_demo()
    _sync(device)
    wall = time.perf_counter() - t0

    return wall, _peak_memory_gib(device)


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------

def _write_per_run_csv(records: List[RunRecord], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_level", "scene", "n_images", "wall_s", "peak_gpu_gib", "error"])
        for r in records:
            w.writerow([
                r.n_level, r.scene, r.n_images,
                f"{r.wall_seconds:.3f}", f"{r.peak_gpu_gib:.3f}",
                r.error or "",
            ])


def _write_by_n_csv(records: List[RunRecord], path: Path,
                    n_levels: List[str]) -> None:
    by_n = {r.n_level: r for r in records if r.error is None}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_level", "n_images", "wall_s", "peak_gpu_gib"])
        for n_level in n_levels:
            if n_level not in by_n:
                continue
            r = by_n[n_level]
            w.writerow([n_level, r.n_images,
                        f"{r.wall_seconds:.3f}", f"{r.peak_gpu_gib:.3f}"])


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_model_summary(model: str, records: List[RunRecord]) -> None:
    print(f"\n{'=' * 65}")
    print(f"=== robust_{model}: Runtime & Peak GPU Memory ===")
    print(f"{'=' * 65}")
    print(f"  {'N-level':<8}  {'N_imgs':>6}  {'wall (s)':>10}  {'peak GPU (GiB)':>14}")
    print("  " + "-" * 44)
    for r in sorted(records, key=lambda x: x.n_level):
        if r.error:
            print(f"  {r.n_level:<8}  {r.n_images:>6}  {'ERROR':>10}  {r.error}")
        else:
            print(
                f"  {r.n_level:<8}  {r.n_images:>6}"
                f"  {r.wall_seconds:>10.2f}  {r.peak_gpu_gib:>14.3f}"
            )


def _print_cross_backend_table(all_records: dict, models: List[str],
                                n_levels: List[str]) -> None:
    print(f"\n{'=' * 80}")
    print("=== Cross-backend: wall time (s) / peak GPU (GiB) per N-level ===")
    print(f"{'=' * 80}")
    col_w = 22
    print(f"  {'N-level':<8}" + "".join(f"  {m:^{col_w}}" for m in models))
    print(f"  {'':8}" + "".join(f"  {'wall(s) / peak(GiB)':^{col_w}}" for _ in models))
    print("  " + "-" * (8 + len(models) * (col_w + 2)))
    for n_level in n_levels:
        row = f"  {n_level:<8}"
        for m in models:
            by_n = {r.n_level: r for r in all_records.get(m, [])}
            if n_level in by_n and by_n[n_level].error is None:
                r = by_n[n_level]
                cell = f"{r.wall_seconds:6.1f}s / {r.peak_gpu_gib:5.2f} GiB"
            else:
                cell = "        —        "
            row += f"  {cell:^{col_w}}"
        print(row)


# ---------------------------------------------------------------------------
# Per-backend run loop
# ---------------------------------------------------------------------------

def _run_backend(model: str, args, out_root: Path,
                 device: str) -> List[RunRecord]:
    print(f"\n{'=' * 65}")
    print(f"=== Backend: robust_{model} ===")
    print(f"{'=' * 65}")

    t_load0 = time.perf_counter()
    mod, exp = BUILDERS[model](args)
    print(f"[{model}] loaded model in {time.perf_counter() - t_load0:.2f}s")
    _silence_module_prints(mod)

    out_dir = out_root / model
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_json = out_dir / "per_run.json"

    existing: dict = {}
    if args.skip_existing and per_run_json.exists():
        with per_run_json.open() as f:
            raw = json.load(f)
        for rd in raw:
            existing[rd["n_level"]] = RunRecord(**rd)
        print(f"[{model}] loaded {len(existing)} existing records")

    entries = _discover(args.bench_root, args.n_levels, args.scene)
    print(f"[{model}] {len(entries)} N-levels to evaluate  (scene={args.scene})")

    records: List[RunRecord] = list(existing.values())

    for n_level, image_dir, image_paths in entries:
        if args.skip_existing and n_level in existing:
            print(f"  skip {n_level} (existing)")
            continue

        n_imgs = len(image_paths)
        print(f"  {n_level}  ({n_imgs} images) ...", flush=True)

        with tempfile.TemporaryDirectory(prefix=f"robust_{model}_mem_") as td:
            try:
                wall, peak_gib = _run_one(
                    mod, exp, image_dir, td, model, device,
                )
                rec = RunRecord(
                    model=model, n_level=n_level, scene=args.scene,
                    n_images=n_imgs, wall_seconds=wall, peak_gpu_gib=peak_gib,
                )
                print(f"    wall={wall:.2f}s  peak_gpu={peak_gib:.3f} GiB")
            except Exception as exc:
                logger.exception("Run failed: %s/%s", model, n_level)
                rec = RunRecord(
                    model=model, n_level=n_level, scene=args.scene,
                    n_images=n_imgs, wall_seconds=float("nan"),
                    peak_gpu_gib=float("nan"), error=repr(exc),
                )
                print(f"    FAILED: {exc!r}")

        records.append(rec)

        with per_run_json.open("w") as f:
            json.dump([asdict(r) for r in records], f, indent=2)

        gc.collect()
        try:
            import torch
            if device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass

    _write_per_run_csv(records, out_dir / "summary.csv")
    _write_by_n_csv(records, out_dir / "summary_by_N.csv", args.n_levels)

    _print_model_summary(model, records)

    del exp, mod
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT,
                    help="Root of 7scenes benchmark: <bench_root>/<N>/<scene>/images/")
    ap.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="Output root. Results go under <out_root>/memory_runtime/.")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS),
                    help="Backends to evaluate (default: all three).")
    ap.add_argument("--n_levels", nargs="+", default=list(N_LEVELS),
                    choices=list(N_LEVELS),
                    help="Frame-count levels to evaluate (default: N25–N350).")
    ap.add_argument("--scene", type=str, default="chess",
                    help="7scenes scene to use (default: chess).")
    ap.add_argument("--gpu", type=str, default="0",
                    help="GPU index (CUDA/HIP). Pass 'cpu' for CPU-only.")
    ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Skip an N-level if its record already exists.")
    ap.add_argument("--log_level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Shared scoring args (match run_7scenes.py defaults).
    ap.add_argument("--attn_a", type=float, default=0.5)
    ap.add_argument("--cos_a", type=float, default=0.5)
    ap.add_argument("--rej_thresh", type=float, default=0.4)

    # VGGT-only
    ap.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    ap.add_argument("--no_point_map", dest="use_point_map", action="store_false",
                    help="Use depth unprojection instead of the point-map branch.")
    ap.set_defaults(use_point_map=True)
    ap.add_argument("--conf_threshold_pct", type=float, default=30.0)

    # Pi3 / MapAnything shared
    ap.add_argument("--max_images", type=int, default=400,
                    help="Max images per sequence (sampled if more are present).")

    # MapAnything-only
    ap.add_argument("--attn_layer", type=int, default=14,
                    help="Index of the global attention layer to probe (must be even).")

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # GPU pinning must happen before torch is imported anywhere downstream.
    if args.gpu.lower() == "cpu":
        device = "cpu"
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(var, None)
    else:
        device = "cuda"
        os.environ["HIP_VISIBLE_DEVICES"] = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    out_dir = args.out_root / "memory_runtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench] bench_root : {args.bench_root}")
    print(f"[bench] scene      : {args.scene}")
    print(f"[bench] N-levels   : {args.n_levels}")
    print(f"[bench] models     : {args.models}")
    print(f"[bench] device     : {device}" + (f" (gpu={args.gpu})" if device == "cuda" else ""))
    print(f"[bench] out_dir    : {out_dir}")

    all_records: dict = {}
    for model in args.models:
        all_records[model] = _run_backend(model, args, out_dir, device)

    with (out_dir / "summary_all.json").open("w") as f:
        json.dump(
            {m: [asdict(r) for r in recs] for m, recs in all_records.items()},
            f, indent=2,
        )

    _print_cross_backend_table(all_records, args.models, args.n_levels)

    print(f"\n[bench] all outputs written to {out_dir}")


if __name__ == "__main__":
    main()
