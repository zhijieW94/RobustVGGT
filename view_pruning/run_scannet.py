#!/usr/bin/env python3
"""Batch-run a robust view-pruning method over every sequence in the ScanNet++ dataset.

Dataset layout (example):
    <dataset_root>/<seq_name>/*.jpg

Each sequence directory contains the RGB frames directly (no nested
`image/` or `images/dslr_images/` subfolders). Only sequences with at least
one image file are processed.

The underlying method is selected with ``--method``:
    vggt          → robust_vggt.RobustVGGTExperiment           (default)
    pi3           → robust_pi3.RobustPi3Experiment
    mapanything   → robust_mapanything.RobustMapAnythingExperiment

External repo roots for Pi3 / MapAnything can be overridden via the
``PI3_ROOT`` / ``MAPANYTHING_ROOT`` environment variables.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASET_ROOT = Path(
    "/home/zhijiewu/Documents/Share/Datasets/BAT3R/scannetpp_short_50"
)
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
METHODS = ("vggt", "pi3", "mapanything")


def _info_print(msg: str) -> None:
    print(f"\033[92m{msg}\033[0m")


def find_image_dir(seq_dir: Path) -> Optional[Path]:
    """Return the sequence dir itself if it contains image files."""
    if seq_dir.is_dir() and any(
        p.suffix in VALID_IMAGE_SUFFIXES for p in seq_dir.iterdir()
    ):
        return seq_dir
    return None


def discover_sequences(dataset_root: Path) -> List[Tuple[str, Path]]:
    """Return (seq_name, image_dir) pairs sorted by sequence name."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    sequences: List[Tuple[str, Path]] = []
    for seq_dir in sorted(dataset_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        image_dir = find_image_dir(seq_dir)
        if image_dir is None:
            _info_print(f"[WARN] No image folder found for sequence {seq_dir.name}; skipping.")
            continue
        sequences.append((seq_dir.name, image_dir))
    return sequences


# ---------------------------------------------------------------------------
# Method-specific factories
#
# Each factory returns a tuple:
#   (ExperimentClass, base_config, reconfigure_fn, cleanup_fn)
#
# reconfigure_fn(experiment, image_dir, exp_name) → updates experiment.config
# cleanup_fn()                                    → optional per-iteration cleanup
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a robust view-pruning method on every ScanNet++ sequence."
    )
    parser.add_argument(
        "--method", choices=METHODS, default="vggt",
        help="Which robust method to run (default: vggt).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help="Root directory of the ScanNet++ dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Directory where per-sequence results are written. "
             "Default: <view_pruning>/results_<method>.",
    )
    parser.add_argument(
        "--sequences", nargs="*", default=None,
        help="Restrict to these sequence names (space-separated). Default: all sequences.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip sequences whose output directory already contains the method's "
             "main PLY file (after_filtering.ply for vggt, reconstruction.ply otherwise).",
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


def main() -> None:
    args = parse_args()
    method = args.method

    if args.output_root is None:
        args.output_root = FILE_PATH.parent / f"results_{method}"

    sequences = discover_sequences(args.dataset_root)
    if args.sequences:
        wanted = set(args.sequences)
        sequences = [(s, d) for s, d in sequences if s in wanted]
        missing = wanted - {s for s, _ in sequences}
        if missing:
            _info_print(f"[WARN] Requested sequences not found: {sorted(missing)}")

    if not sequences:
        _info_print("[ERROR] No sequences to process.")
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    _info_print(
        f"[INFO] method={method}  processing {len(sequences)} sequence(s) "
        f"from {args.dataset_root} → {args.output_root}"
    )

    ExperimentCls, base_config, reconfigure, cleanup = METHOD_BUILDERS[method](args)

    # Load the model once and reuse across sequences.
    experiment = ExperimentCls(base_config)

    sentinel_name = _sentinel_output_filename(method)
    failures: List[Tuple[str, str]] = []
    for idx, (seq_name, image_dir) in enumerate(sequences, start=1):
        out_dir = args.output_root / seq_name
        if args.skip_existing and (out_dir / sentinel_name).exists():
            _info_print(f"[{idx}/{len(sequences)}] Skipping {seq_name} (already done).")
            continue

        _info_print(
            f"[{idx}/{len(sequences)}] === Sequence {seq_name} ===\n"
            f"    images: {image_dir}\n"
            f"    output: {out_dir}"
        )

        reconfigure(experiment, image_dir, str(out_dir))

        try:
            experiment.run_demo()
        except Exception as e:
            tb = traceback.format_exc()
            _info_print(f"[ERROR] Sequence {seq_name} failed: {e}\n{tb}")
            failures.append((seq_name, str(e)))
        finally:
            cleanup()

    _info_print(
        f"[DONE] Processed {len(sequences) - len(failures)}/{len(sequences)} sequences successfully."
    )
    if failures:
        _info_print("[DONE] Failures:")
        for name, msg in failures:
            _info_print(f"    - {name}: {msg}")
        sys.exit(2)


if __name__ == "__main__":
    main()
