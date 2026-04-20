#!/usr/bin/env python3
"""Batch-run robust view-pruning methods over noisy testbeds (eth3d / hiroom / scannet).

Noisy dataset layout (unified across the three datasets):
    <dataset_root>/<noise_level>/<dataset>/<seq_name>/images/*.jpg

where
    noise_level ∈ {low_noisy, mid_noisy, high_noisy}
    dataset     ∈ {eth3d, hiroom, scannetpp_50}

For each selected (method, noise_level, dataset, seq), the model is invoked
on ``<seq_dir>/images`` and results are written to::

    <output_root>/<method>/<noise_level>/<dataset>/<seq_name>/

Parallelism
-----------
The script auto-detects *free* AMD GPUs via ``rocm-smi`` (a GPU is considered
free when its VRAM usage is below ``--gpu-mem-threshold-mb``). Sequences of a
given method are round-robin distributed across those GPUs and run in parallel
subprocesses. Each subprocess loads the method's model once and reuses it
across its assigned sequences. Methods run sequentially (outer loop).

Override auto-detection with ``--gpus 0 2 3`` or force single-process with
``--gpus -1``.

External repo roots for Pi3 / MapAnything can be overridden via the
``PI3_ROOT`` / ``MAPANYTHING_ROOT`` environment variables.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASET_ROOT = Path("/nvmepool/zhijiewu/Datasets/MACV_testbeds/noisy")
DEFAULT_OUTPUT_ROOT = Path("/nvmepool/zhijiewu/results/MACV/noisy/Robust_X")

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
METHODS = ("vggt", "pi3", "mapanything")
NOISE_LEVELS = ("low_noisy", "mid_noisy", "high_noisy")
DATASETS = ("eth3d", "hiroom", "scannetpp_50")


def _info_print(msg: str) -> None:
    print(f"\033[92m{msg}\033[0m", flush=True)


def _warn_print(msg: str) -> None:
    print(f"\033[93m{msg}\033[0m", flush=True)


_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"
_USE_ANSI_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _color_status(status: str) -> str:
    if not _USE_ANSI_COLOR:
        return status
    if status == "DONE":
        return f"{_ANSI_GREEN}{status}{_ANSI_RESET}"
    if status == "FAIL":
        return f"{_ANSI_RED}{status}{_ANSI_RESET}"
    return status


@contextlib.contextmanager
def _redirect_fd_to_file(log_path: Path):
    """Redirect fd 1 (stdout) and fd 2 (stderr) to ``log_path`` for the block.

    Uses ``os.dup2`` so the redirect also captures C-level output (e.g. HF
    Hub warnings, CUDA/HIP runtime messages) — not just Python ``print``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    f = open(log_path, "ab", buffering=0)
    try:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
    finally:
        f.close()
        os.close(saved_stdout)
        os.close(saved_stderr)


def find_image_dir(seq_dir: Path) -> Optional[Path]:
    """Return ``<seq_dir>/images`` if it contains at least one image file."""
    candidate = seq_dir / "images"
    if candidate.is_dir() and any(
        p.suffix in VALID_IMAGE_SUFFIXES for p in candidate.iterdir()
    ):
        return candidate
    return None


def discover_sequences(dataset_dir: Path) -> List[Tuple[str, Path]]:
    """Return (seq_name, image_dir) pairs sorted by sequence name."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    sequences: List[Tuple[str, Path]] = []
    for seq_dir in sorted(dataset_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        image_dir = find_image_dir(seq_dir)
        if image_dir is None:
            _warn_print(f"[WARN] No image folder found for sequence {seq_dir.name}; skipping.")
            continue
        sequences.append((seq_dir.name, image_dir))
    return sequences


# ---------------------------------------------------------------------------
# GPU detection (AMD / ROCm)
# ---------------------------------------------------------------------------

_ROCM_VRAM_USED_RE = re.compile(
    r"GPU\[(\d+)\].*VRAM Total Used Memory \(B\)\s*:\s*(\d+)",
    re.IGNORECASE,
)


def detect_free_amd_gpus(mem_threshold_mb: float) -> List[int]:
    """Return indices of AMD GPUs whose VRAM usage is below ``mem_threshold_mb``.

    Shells out to ``rocm-smi --showmeminfo vram``. If ``rocm-smi`` is missing or
    errors out, returns an empty list so the caller can fall back.
    """
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"],
            text=True,
            timeout=20,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        _warn_print("[WARN] rocm-smi not found on PATH; cannot auto-detect GPUs.")
        return []
    except subprocess.CalledProcessError as e:
        _warn_print(f"[WARN] rocm-smi failed (exit {e.returncode}).")
        return []
    except subprocess.TimeoutExpired:
        _warn_print("[WARN] rocm-smi timed out; cannot auto-detect GPUs.")
        return []

    used_mb_per_gpu: dict[int, float] = {}
    for line in out.splitlines():
        m = _ROCM_VRAM_USED_RE.search(line)
        if m:
            gpu_id = int(m.group(1))
            used_mb_per_gpu[gpu_id] = int(m.group(2)) / (1024 * 1024)

    free = sorted(g for g, mb in used_mb_per_gpu.items() if mb < mem_threshold_mb)
    for g, mb in sorted(used_mb_per_gpu.items()):
        status = "FREE" if mb < mem_threshold_mb else "BUSY"
        _info_print(f"    GPU[{g}]: {mb:8.1f} MB used  → {status}")
    return free


# ---------------------------------------------------------------------------
# Method-specific factories
#
# Each factory returns a tuple:
#   (ExperimentClass, base_config, reconfigure_fn, cleanup_fn)
# ---------------------------------------------------------------------------

def _build_vggt(args: argparse.Namespace) -> Tuple[Any, Any, Callable, Callable]:
    from robust_vggt import ExperimentConfig, RobustVGGTExperiment, safe_empty_cache

    base_config = ExperimentConfig(
        image_dir=Path("."),  # placeholder, overridden per sequence
        preprocess_mode=args.preprocess_mode,
        exp_name="placeholder",
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        use_point_map=args.use_point_map,
        conf_threshold_pct=args.conf_threshold_pct,
    )

    def reconfigure(exp, image_dir: Path, exp_name: str) -> None:
        exp.config = replace(base_config, image_dir=image_dir, exp_name=exp_name)

    return RobustVGGTExperiment, base_config, reconfigure, safe_empty_cache


def _build_pi3(args: argparse.Namespace) -> Tuple[Any, Any, Callable, Callable]:
    from robust_pi3 import ExperimentConfig, RobustPi3Experiment, safe_empty_cache

    base_config = ExperimentConfig(
        image_dir=Path("."),
        exp_name="placeholder",
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        max_images=args.max_images,
    )

    def reconfigure(exp, image_dir: Path, exp_name: str) -> None:
        exp.config = replace(base_config, image_dir=image_dir, exp_name=exp_name)

    return RobustPi3Experiment, base_config, reconfigure, safe_empty_cache


def _build_mapanything(args: argparse.Namespace) -> Tuple[Any, Any, Callable, Callable]:
    from robust_mapanything import (
        ExperimentConfig,
        RobustMapAnythingExperiment,
        safe_empty_cache,
    )

    base_config = ExperimentConfig(
        image_dir=Path("."),
        exp_name="placeholder",
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
        attn_layer=args.attn_layer,
        max_images=args.max_images,
    )

    def reconfigure(exp, image_dir: Path, exp_name: str) -> None:
        exp.config = replace(base_config, image_dir=image_dir, exp_name=exp_name)

    return RobustMapAnythingExperiment, base_config, reconfigure, safe_empty_cache


METHOD_BUILDERS: dict[str, Callable[[argparse.Namespace], Tuple[Any, Any, Callable, Callable]]] = {
    "vggt": _build_vggt,
    "pi3": _build_pi3,
    "mapanything": _build_mapanything,
}


def _sentinel_output_filename(method: str) -> str:
    """Filename used by ``--skip-existing`` to detect completed runs."""
    if method == "vggt":
        return "after_filtering.ply"
    # Both Pi3 and MapAnything write reconstruction.ply on success.
    return "reconstruction.ply"


# ---------------------------------------------------------------------------
# Worker (runs inside a spawned subprocess, pinned to one AMD GPU)
# ---------------------------------------------------------------------------

Job = Tuple[str, str, str, str]  # (noise_level, dataset, seq_name, image_dir_str)


def _pin_to_gpu(gpu_id: int) -> None:
    """Restrict the current process to a single AMD GPU.

    Must be called BEFORE importing torch. Sets HIP and CUDA visibility vars
    (PyTorch+ROCm respects both).

    NOTE: do NOT also set ROCR_VISIBLE_DEVICES. ROCR filters at the ROCr
    runtime level and renumbers devices before HIP sees them; stacking it
    with HIP_VISIBLE_DEVICES to the same physical index makes every worker
    except GPU0 see zero devices (the HIP index refers to a position that
    no longer exists after ROCr already filtered down to one GPU).
    """
    val = str(gpu_id)
    os.environ["HIP_VISIBLE_DEVICES"] = val
    os.environ["CUDA_VISIBLE_DEVICES"] = val
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)


IndexedJob = Tuple[int, str, str, str, str]  # (global_idx, noise, dataset, seq, image_dir)


def _locked_print(lock, msg: str) -> None:
    """Print ``msg`` while holding ``lock`` (no-op-safe if lock is None)."""
    if lock is None:
        print(msg, flush=True)
        return
    with lock:
        print(msg, flush=True)


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
    """Entry point for a per-GPU subprocess.

    Loads the method's model once, then runs every assigned job on that
    single GPU. Each job's stdout/stderr are redirected to ``<out_dir>/run.log``
    so the master terminal only shows concise START/DONE/FAIL status lines.
    """
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
            [(noise_level, dataset, seq_name, f"model-load failure: {e}")
             for _, noise_level, dataset, seq_name, _ in jobs]
        )
        return

    failures: List[Tuple[str, str, str, str]] = []
    for global_idx, noise_level, dataset, seq_name, image_dir_str in jobs:
        image_dir = Path(image_dir_str)
        out_dir = output_root / method / noise_level / dataset / seq_name
        tag = f"{method}/{noise_level}/{dataset}/{seq_name}"
        log_path = out_dir / "run.log"

        if (out_dir / "clean_images").is_dir():
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
            failures.append((noise_level, dataset, seq_name, str(e)))
        finally:
            cleanup()

    del experiment
    cleanup()
    result_queue.put(failures)


# ---------------------------------------------------------------------------
# Argument parsing & job collection
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run robust view-pruning methods on every noisy-testbed sequence "
            "across (method × noise_level × dataset), parallelized over free AMD GPUs."
        )
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS),
        help="Which robust methods to run (default: all).",
    )
    parser.add_argument(
        "--noise-levels", nargs="+", choices=NOISE_LEVELS, default=list(NOISE_LEVELS),
        help="Which noise levels to process (default: all).",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
        help="Which datasets to process (default: all).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help="Root directory of the noisy testbeds (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for results. Per-run output goes to "
             "<output_root>/<method>/<noise_level>/<dataset>/<seq_name>. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--sequences", nargs="*", default=None,
        help="Restrict to these sequence names (space-separated, applied to every "
             "(noise_level, dataset) combination). Default: all discovered sequences.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs whose output directory already contains the method's "
             "main PLY file (after_filtering.ply for vggt, reconstruction.ply otherwise).",
    )

    # GPU / parallelism
    gpu_grp = parser.add_argument_group("GPU / parallelism")
    gpu_grp.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="Explicit GPU indices to use (e.g. '--gpus 0 2 3'). Pass '-1' to "
             "disable multi-GPU and run a single in-process worker on the "
             "default device. If omitted, free AMD GPUs are auto-detected via rocm-smi.",
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


def _collect_jobs(
    dataset_root: Path,
    noise_levels: List[str],
    datasets: List[str],
    seq_filter: Optional[List[str]],
) -> List[Job]:
    """Return a flat list of (noise_level, dataset, seq_name, image_dir_str)."""
    wanted_seqs = set(seq_filter) if seq_filter else None
    jobs: List[Job] = []

    for noise_level in noise_levels:
        for dataset in datasets:
            dataset_dir = dataset_root / noise_level / dataset
            if not dataset_dir.is_dir():
                _warn_print(
                    f"[WARN] Missing dataset directory: {dataset_dir}; skipping."
                )
                continue
            sequences = discover_sequences(dataset_dir)
            if wanted_seqs is not None:
                filtered = [(s, d) for s, d in sequences if s in wanted_seqs]
                missing = wanted_seqs - {s for s, _ in sequences}
                if missing:
                    _warn_print(
                        f"[WARN] {noise_level}/{dataset}: requested sequences "
                        f"not found: {sorted(missing)}"
                    )
                sequences = filtered
            for seq_name, image_dir in sequences:
                jobs.append((noise_level, dataset, seq_name, str(image_dir)))
    return jobs


def _resolve_gpus(args: argparse.Namespace) -> List[int]:
    """Resolve which GPU indices to use.

    Returns ``[]`` to mean "single in-process worker on the default device
    (no GPU pinning, no subprocess spawning)".
    """
    if args.gpus is not None:
        if len(args.gpus) == 1 and args.gpus[0] < 0:
            _info_print("[INFO] --gpus -1 given; running a single in-process worker.")
            return []
        _info_print(f"[INFO] Using user-specified GPUs: {args.gpus}")
        return list(args.gpus)

    _info_print("[INFO] Auto-detecting free AMD GPUs via rocm-smi …")
    free = detect_free_amd_gpus(args.gpu_mem_threshold_mb)
    if not free:
        _warn_print(
            "[WARN] No free AMD GPUs detected; falling back to a single "
            "in-process worker on the default device."
        )
        return []
    _info_print(f"[INFO] Free AMD GPUs: {free}")
    if args.max_workers is not None and args.max_workers > 0:
        free = free[: args.max_workers]
        _info_print(f"[INFO] Capped to --max-workers: {free}")
    return free


def _round_robin_split(jobs: List[IndexedJob], n: int) -> List[List[IndexedJob]]:
    """Split ``jobs`` into ``n`` round-robin chunks to balance length bias."""
    chunks: List[List[IndexedJob]] = [[] for _ in range(n)]
    for i, job in enumerate(jobs):
        chunks[i % n].append(job)
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _index_jobs(jobs: List[Job], method_offset: int) -> List[IndexedJob]:
    """Attach a 1-based global index to each job, starting at ``method_offset + 1``."""
    return [
        (method_offset + i + 1, noise, dataset, seq, img)
        for i, (noise, dataset, seq, img) in enumerate(jobs)
    ]


def _run_method_single_process(
    method: str,
    args: argparse.Namespace,
    jobs: List[Job],
    method_offset: int,
    total_runs: int,
) -> List[Tuple[str, str, str, str]]:
    """Single-process path when no GPUs are pinned. Matches multi-GPU output format."""
    sentinel_name = _sentinel_output_filename(method)
    ExperimentCls, base_config, reconfigure, cleanup = METHOD_BUILDERS[method](args)
    experiment = ExperimentCls(base_config)

    failures: List[Tuple[str, str, str, str]] = []
    for global_idx, noise_level, dataset, seq_name, image_dir_str in _index_jobs(jobs, method_offset):
        image_dir = Path(image_dir_str)
        out_dir = args.output_root / method / noise_level / dataset / seq_name
        tag = f"{method}/{noise_level}/{dataset}/{seq_name}"
        log_path = out_dir / "run.log"

        if (out_dir / "clean_images").is_dir():
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
            failures.append((noise_level, dataset, seq_name, str(e)))
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
) -> List[Tuple[str, str, str, str]]:
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

    failures: List[Tuple[str, str, str, str]] = []
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


def main() -> None:
    args = parse_args()

    jobs = _collect_jobs(
        args.dataset_root, args.noise_levels, args.datasets, args.sequences
    )
    if not jobs:
        _warn_print("[ERROR] No sequences to process.")
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    gpus = _resolve_gpus(args)

    total_runs = len(jobs) * len(args.methods)
    _info_print(
        f"[INFO] methods={args.methods}  "
        f"noise_levels={args.noise_levels}  datasets={args.datasets}  "
        f"jobs/method={len(jobs)}  total_runs={total_runs}  "
        f"gpus={gpus if gpus else '[single-process]'}  "
        f"dataset_root={args.dataset_root}  output_root={args.output_root}"
    )

    ctx = mp.get_context("spawn") if gpus else None
    print_lock = ctx.Lock() if ctx is not None else None
    all_failures: List[Tuple[str, str, str, str, str]] = []

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
        for noise_level, dataset, seq_name, msg in method_failures:
            all_failures.append((method, noise_level, dataset, seq_name, msg))

    succeeded = total_runs - len(all_failures)
    _info_print(f"\n[DONE] Processed {succeeded}/{total_runs} runs successfully.")
    if all_failures:
        _warn_print("[DONE] Failures:")
        for method, noise_level, dataset, seq_name, msg in all_failures:
            _warn_print(f"    - {method}/{noise_level}/{dataset}/{seq_name}: {msg}")
        sys.exit(2)


if __name__ == "__main__":
    main()
