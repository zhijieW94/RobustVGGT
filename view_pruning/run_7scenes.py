#!/usr/bin/env python3
"""Batch-run robust view-pruning methods over the 7scenes long-sequence benchmark.

Dataset layout::

    <dataset_root>/<n_level>/<scene>/images/*.jpg

where
    n_level ∈ {N50, N100, N150, N200, N250, N300, N350}
    scene   ∈ {chess, fire, heads, office, pumpkin, redkitchen, stairs}

For each selected (method, n_level, scene), the model is invoked on
``<scene_dir>/images`` and results are written to::

    <output_root>/<method>/<n_level>/<scene>/

GPU detection, method factories (VGGT / Pi3 / MapAnything), persistent-worker
spawning, and logging utilities are reused from ``view_pruning.run_noisy``.
Methods run sequentially (outer loop); inside each method, sequences are
round-robin distributed across the resolved GPUs.

Usage example::

    python view_pruning/run_7scenes.py \\
        --methods vggt pi3 mapanything \\
        --n-levels N50 N100 N150 N200 N250 N300 N350 \\
        --gpus 0 1 2 3
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from view_pruning.run_noisy import (  # noqa: E402
    METHOD_BUILDERS,
    _color_status,
    _info_print,
    _locked_print,
    _pin_to_gpu,
    _redirect_fd_to_file,
    _resolve_gpus,
    _round_robin_split,
    _sentinel_output_filename,
    _warn_print,
    discover_sequences,
)


DEFAULT_DATASET_ROOT = Path("/nvmepool/zhijiewu/Datasets/Final_Benchmarks/7scenes")
DEFAULT_OUTPUT_ROOT = Path("/nvmepool/zhijiewu/results/MACV/7scenes/robust_x/first_stage")

METHODS = ("vggt", "pi3", "mapanything")
N_LEVELS = ("N50", "N100", "N150", "N200", "N250", "N300", "N350")
SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")


Job = Tuple[str, str, str]                # (n_level, scene, image_dir_str)
IndexedJob = Tuple[int, str, str, str]    # (global_idx, n_level, scene, image_dir_str)


# ---------------------------------------------------------------------------
# Worker (runs inside a spawned subprocess, pinned to one GPU)
# ---------------------------------------------------------------------------

def _index_jobs(jobs: List[Job], method_offset: int) -> List[IndexedJob]:
    return [
        (method_offset + i + 1, n_level, scene, img)
        for i, (n_level, scene, img) in enumerate(jobs)
    ]


def _gpu_worker_entry(
    gpu_id: int,
    method: str,
    args_dict: dict,
    jobs: List[IndexedJob],
    output_root_str: str,
    total_runs: int,
    print_lock,
    result_queue: mp.Queue,
) -> None:
    """Per-GPU worker: load model once, run all assigned jobs on this GPU."""
    _pin_to_gpu(gpu_id)

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = argparse.Namespace(**args_dict)
    output_root = Path(output_root_str)
    sentinel_name = _sentinel_output_filename(method)

    try:
        ExperimentCls, base_config, reconfigure, cleanup = METHOD_BUILDERS[method](args)
        experiment = ExperimentCls(base_config)
    except Exception as e:
        tb = traceback.format_exc()
        _locked_print(
            print_lock,
            f"\033[93m[FATAL] {method} model load on gpu={gpu_id} failed: {e}\n{tb}\033[0m",
        )
        result_queue.put(
            [(n_level, scene, f"model-load failure: {e}")
             for _, n_level, scene, _ in jobs]
        )
        return

    failures: List[Tuple[str, str, str]] = []
    for global_idx, n_level, scene, image_dir_str in jobs:
        image_dir = Path(image_dir_str)
        out_dir = output_root / method / n_level / scene
        tag = f"{method}/{n_level}/{scene}"
        log_path = out_dir / "run.log"

        if not args.force and (out_dir / "clean_images").is_dir():
            _locked_print(
                print_lock,
                f"[{global_idx}/{total_runs}] SKIP  {tag}  (gpu={gpu_id}, clean_images/ exists)",
            )
            continue

        if args.skip_existing and (out_dir / sentinel_name).exists():
            _locked_print(
                print_lock,
                f"[{global_idx}/{total_runs}] SKIP  {tag}  (gpu={gpu_id}, already done)",
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        _locked_print(
            print_lock,
            f"[{global_idx}/{total_runs}] START {tag}  (gpu={gpu_id})  log={log_path}",
        )

        reconfigure(experiment, image_dir, str(out_dir))
        start = time.time()
        try:
            with _redirect_fd_to_file(log_path):
                experiment.run_demo()
            dur = time.time() - start
            _locked_print(
                print_lock,
                f"[{global_idx}/{total_runs}] {_color_status('DONE')}  {tag}  "
                f"(gpu={gpu_id}, {dur:.1f}s)",
            )
        except Exception as e:
            dur = time.time() - start
            tb = traceback.format_exc()
            try:
                with open(log_path, "a") as lf:
                    lf.write(f"\n# run_demo raised: {e}\n{tb}\n")
            except Exception:
                pass
            _locked_print(
                print_lock,
                f"[{global_idx}/{total_runs}] {_color_status('FAIL')}  {tag}  "
                f"(gpu={gpu_id}, {dur:.1f}s)  see {log_path}",
            )
            failures.append((n_level, scene, str(e)))
        finally:
            cleanup()

    del experiment
    cleanup()
    result_queue.put(failures)


def _run_method_single_process(
    method: str,
    args: argparse.Namespace,
    jobs: List[Job],
    method_offset: int,
    total_runs: int,
) -> List[Tuple[str, str, str]]:
    """Single-process fallback when no GPUs are pinned."""
    sentinel_name = _sentinel_output_filename(method)
    ExperimentCls, base_config, reconfigure, cleanup = METHOD_BUILDERS[method](args)
    experiment = ExperimentCls(base_config)

    failures: List[Tuple[str, str, str]] = []
    for global_idx, n_level, scene, image_dir_str in _index_jobs(jobs, method_offset):
        image_dir = Path(image_dir_str)
        out_dir = args.output_root / method / n_level / scene
        tag = f"{method}/{n_level}/{scene}"
        log_path = out_dir / "run.log"

        if not args.force and (out_dir / "clean_images").is_dir():
            print(
                f"[{global_idx}/{total_runs}] SKIP  {tag}  (cpu, clean_images/ exists)",
                flush=True,
            )
            continue

        if args.skip_existing and (out_dir / sentinel_name).exists():
            print(f"[{global_idx}/{total_runs}] SKIP  {tag}  (cpu, already done)", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{global_idx}/{total_runs}] START {tag}  (cpu)  log={log_path}",
            flush=True,
        )
        reconfigure(experiment, image_dir, str(out_dir))
        start = time.time()
        try:
            with _redirect_fd_to_file(log_path):
                experiment.run_demo()
            dur = time.time() - start
            print(
                f"[{global_idx}/{total_runs}] {_color_status('DONE')}  {tag}  (cpu, {dur:.1f}s)",
                flush=True,
            )
        except Exception as e:
            dur = time.time() - start
            tb = traceback.format_exc()
            try:
                with open(log_path, "a") as lf:
                    lf.write(f"\n# run_demo raised: {e}\n{tb}\n")
            except Exception:
                pass
            print(
                f"[{global_idx}/{total_runs}] {_color_status('FAIL')}  {tag}  "
                f"(cpu, {dur:.1f}s)  see {log_path}",
                flush=True,
            )
            failures.append((n_level, scene, str(e)))
        finally:
            cleanup()

    del experiment
    cleanup()
    return failures


def _run_method_multi_gpu(
    method: str,
    args: argparse.Namespace,
    jobs: List[Job],
    gpus: List[int],
    ctx: mp.context.BaseContext,
    method_offset: int,
    total_runs: int,
    print_lock,
) -> List[Tuple[str, str, str]]:
    """Spawn one subprocess per GPU and round-robin the method's jobs across them."""
    indexed = _index_jobs(jobs, method_offset)
    n = min(len(gpus), len(indexed))
    active_gpus = gpus[:n]
    chunks = _round_robin_split(indexed, n)

    result_queue: mp.Queue = ctx.Queue()
    processes: List[mp.Process] = []
    args_dict = vars(args).copy()

    for gpu_id, chunk in zip(active_gpus, chunks):
        p = ctx.Process(
            target=_gpu_worker_entry,
            args=(gpu_id, method, args_dict, chunk, str(args.output_root),
                  total_runs, print_lock, result_queue),
            daemon=False,
        )
        p.start()
        processes.append(p)

    failures: List[Tuple[str, str, str]] = []
    for _ in processes:
        failures.extend(result_queue.get())

    for p in processes:
        p.join()
        if p.exitcode != 0:
            _locked_print(
                print_lock,
                f"\033[93m[WARN] Worker (pid {p.pid}) exited with code {p.exitcode}.\033[0m",
            )
    return failures


# ---------------------------------------------------------------------------
# Job collection
# ---------------------------------------------------------------------------

def _collect_jobs(
    dataset_root: Path,
    n_levels: List[str],
    scenes: List[str],
) -> List[Job]:
    """Return a flat list of (n_level, scene, image_dir_str)."""
    wanted_scenes = set(scenes)
    jobs: List[Job] = []

    for n_level in n_levels:
        n_dir = dataset_root / n_level
        if not n_dir.is_dir():
            _warn_print(f"[WARN] Missing n_level directory: {n_dir}; skipping.")
            continue
        sequences = discover_sequences(n_dir)
        present = {s for s, _ in sequences}
        missing = wanted_scenes - present
        if missing:
            _warn_print(
                f"[WARN] {n_level}: requested scenes not found: {sorted(missing)}"
            )
        for scene, image_dir in sequences:
            if scene in wanted_scenes:
                jobs.append((n_level, scene, str(image_dir)))
    return jobs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run robust view-pruning methods on every 7scenes sequence "
            "across (method × n_level × scene), parallelized over free GPUs."
        )
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS),
        help="Which robust methods to run (default: all).",
    )
    parser.add_argument(
        "--n-levels", nargs="+", choices=N_LEVELS, default=list(N_LEVELS),
        help="Sequence-length tiers to process (default: all).",
    )
    parser.add_argument(
        "--scenes", nargs="+", choices=SCENES, default=list(SCENES),
        help="Which 7scenes scenes to process (default: all).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help="Root directory of the 7scenes benchmark (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for results. Per-run output goes to "
             "<output_root>/<method>/<n_level>/<scene>. Default: %(default)s.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs whose output directory already contains the method's main "
             "PLY file (after_filtering.ply for vggt, reconstruction.ply otherwise).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run scenes even if their out_dir already has a 'clean_images/' folder.",
    )

    gpu_grp = parser.add_argument_group("GPU / parallelism")
    gpu_grp.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="Explicit GPU indices to use (e.g. '--gpus 0 2 3'). Pass '-1' to "
             "disable multi-GPU. If omitted, free GPUs are auto-detected.",
    )
    gpu_grp.add_argument(
        "--gpu-mem-threshold-mb", type=float, default=500.0,
        help="A GPU is considered FREE when its VRAM usage is below this many MB "
             "(default: %(default)s).",
    )
    gpu_grp.add_argument(
        "--max-workers", type=int, default=None,
        help="Cap the number of parallel workers (default: one per selected GPU).",
    )

    # Shared scoring args
    parser.add_argument("--attn-a", type=float, default=0.5)
    parser.add_argument("--cos-a", type=float, default=0.5)
    parser.add_argument("--rej-thresh", type=float, default=0.4)

    # VGGT-only
    vggt_grp = parser.add_argument_group("VGGT-only arguments")
    vggt_grp.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop")
    vggt_grp.add_argument(
        "--no-point-map", dest="use_point_map", action="store_false",
        help="Use depth unprojection instead of the point-map branch for PLY generation.",
    )
    vggt_grp.set_defaults(use_point_map=True)
    vggt_grp.add_argument("--conf-threshold-pct", type=float, default=30.0)

    # Pi3 / MapAnything shared
    pi3_ma_grp = parser.add_argument_group("Pi3 / MapAnything arguments")
    pi3_ma_grp.add_argument(
        "--max-images", type=int, default=400,
        help="Max images per sequence (randomly sampled if more are present).",
    )

    # MapAnything-only
    ma_grp = parser.add_argument_group("MapAnything-only arguments")
    ma_grp.add_argument(
        "--attn-layer", type=int, default=14,
        help="Index of the global attention layer to probe (must be even).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    jobs = _collect_jobs(args.dataset_root, args.n_levels, args.scenes)
    if not jobs:
        _warn_print("[ERROR] No sequences to process.")
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    gpus = _resolve_gpus(args)

    total_runs = len(jobs) * len(args.methods)
    _info_print(
        f"[INFO] methods={args.methods}  n_levels={args.n_levels}  "
        f"scenes={args.scenes}  jobs/method={len(jobs)}  total_runs={total_runs}  "
        f"gpus={gpus if gpus else '[single-process]'}  "
        f"dataset_root={args.dataset_root}  output_root={args.output_root}"
    )

    ctx = mp.get_context("spawn") if gpus else None
    print_lock = ctx.Lock() if ctx is not None else None
    all_failures: List[Tuple[str, str, str, str]] = []

    for method_idx, method in enumerate(args.methods):
        _info_print(f"\n========== method={method} ==========")
        method_offset = method_idx * len(jobs)
        if gpus:
            method_failures = _run_method_multi_gpu(
                method, args, jobs, gpus, ctx,
                method_offset, total_runs, print_lock,
            )
        else:
            method_failures = _run_method_single_process(
                method, args, jobs, method_offset, total_runs,
            )
        for n_level, scene, msg in method_failures:
            all_failures.append((method, n_level, scene, msg))

    succeeded = total_runs - len(all_failures)
    _info_print(f"\n[DONE] Processed {succeeded}/{total_runs} runs successfully.")
    if all_failures:
        _warn_print("[DONE] Failures:")
        for method, n_level, scene, msg in all_failures:
            _warn_print(f"    - {method}/{n_level}/{scene}: {msg}")
        sys.exit(2)


if __name__ == "__main__":
    main()
