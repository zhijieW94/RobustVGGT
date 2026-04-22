"""Compute view-filtering classification metrics for Robust_X outputs.

Ground truth: /nvmepool/zhijiewu/Datasets/Final_Benchmarks/<level>/<dataset>/<seq>/image_list.txt
Predictions: <pred_root>/<method>/<level>/<dataset>/<seq>/clean_images/
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

GT_ROOT = Path("/nvmepool/zhijiewu/Datasets/Final_Benchmarks")

METHODS = ["vggt", "pi3", "mapanything"]
LEVELS = ["clean", "low", "mid", "high"]
DATASETS = ["eth3d", "scannetpp_50", "onthego", "phototourism"]

METRIC_NAMES = ["CleanKeepRate", "DistractorRejectionRate", "F1", "MCC"]
INT_COLS = {"TP", "FN", "FP", "TN", "N_clean", "N_noisy", "N_kept", "n_sequences"}


def parse_image_list(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag, _, rel = line.partition(" ")
            name = Path(rel).name
            if tag == "[Clean]":
                labels[name] = 1
            elif tag == "[Noisy]":
                labels[name] = 0
            else:
                raise ValueError(f"bad line in {path}: {line!r}")
    return labels


def list_kept(clean_dir: Path) -> set[str]:
    if not clean_dir.is_dir():
        return set()
    return {p.name for p in clean_dir.iterdir() if p.is_file()}


def confusion(labels: dict[str, int], kept: set[str]) -> tuple[int, int, int, int]:
    tp = fn = fp = tn = 0
    for name, lab in labels.items():
        in_kept = name in kept
        if lab == 1 and in_kept:
            tp += 1
        elif lab == 1 and not in_kept:
            fn += 1
        elif lab == 0 and in_kept:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def metrics_from_cm(tp: int, fn: int, fp: int, tn: int) -> dict[str, float]:
    clean_keep = _safe_div(tp, tp + fn)
    distractor_rej = _safe_div(tn, tn + fp)
    precision_kept = _safe_div(tp, tp + fp)
    recall = clean_keep
    f1 = _safe_div(2 * precision_kept * recall, precision_kept + recall) if not (
        math.isnan(precision_kept) or math.isnan(recall) or (precision_kept + recall) == 0
    ) else float("nan")
    balanced_acc = (
        0.5 * (clean_keep + distractor_rej)
        if not (math.isnan(clean_keep) or math.isnan(distractor_rej))
        else float("nan")
    )
    denom2 = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denom2) if denom2 > 0 else float("nan")
    false_clean_discard = _safe_div(fn, tp + fn)
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "N_clean": tp + fn, "N_noisy": fp + tn, "N_kept": tp + fp,
        "CleanKeepRate": clean_keep,
        "FalseCleanDiscardRate": false_clean_discard,
        "DistractorRejectionRate": distractor_rej,
        "PrecisionKept": precision_kept,
        "F1": f1,
        "BalancedAccuracy": balanced_acc,
        "MCC": mcc,
    }


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _fmt_cell(col: str, val) -> str:
    if col in INT_COLS:
        return str(int(val))
    if isinstance(val, float):
        return "nan" if math.isnan(val) else f"{val:.4f}"
    return str(val)


def write_markdown_table(path: Path, rows: list[dict], columns: list[str], title: str) -> None:
    cells = [[_fmt_cell(c, r[c]) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c)
              for i, c in enumerate(columns)]

    def line(vals):
        return "| " + " | ".join(v.rjust(w) if i > 0 and cells else v.ljust(w)
                                  for i, (v, w) in enumerate(zip(vals, widths))) + " |"

    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    sep_parts = []
    for i, w in enumerate(widths):
        if i == 0:
            sep_parts.append(":" + "-" * (w + 1))
        else:
            sep_parts.append("-" * (w + 1) + ":")
    sep = "| " + " | ".join(sep_parts) + " |"
    body = "\n".join(line(row) for row in cells)

    content = f"# {title}\n\n{header}\n{sep}\n{body}\n"
    path.write_text(content)


def print_table(rows: list[dict], columns: list[str], title: str) -> None:
    cells = [[_fmt_cell(c, r[c]) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c)
              for i, c in enumerate(columns)]
    print(f"\n=== {title} ===")
    header = "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                       for i, (c, w) in enumerate(zip(columns, widths)))
    print(header)
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) if i == 0 else v.rjust(w)
                        for i, (v, w) in enumerate(zip(row, widths))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-root",
        default="/nvmepool/zhijiewu/results/MACV/Robust_X",
        help="Root containing <method>/<level>/<dataset>/<seq>/clean_images/",
    )
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <pred-root>/_metrics)")
    args = parser.parse_args()
    PRED_ROOT = Path(args.pred_root)
    OUT_DIR = Path(args.out_dir) if args.out_dir else PRED_ROOT / "_metrics"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seq_rows: list[dict] = []
    missing: list[str] = []

    for method in METHODS:
        for level in LEVELS:
            for dataset in DATASETS:
                gt_ds_dir = GT_ROOT / level / dataset
                if not gt_ds_dir.is_dir():
                    continue
                for seq_dir in sorted(gt_ds_dir.iterdir()):
                    if not seq_dir.is_dir():
                        continue
                    seq = seq_dir.name
                    gt_file = seq_dir / "image_list.txt"
                    if not gt_file.is_file():
                        continue
                    pred_seq = PRED_ROOT / method / level / dataset / seq
                    clean_dir = pred_seq / "clean_images"
                    if not clean_dir.is_dir():
                        missing.append(f"{method}/{level}/{dataset}/{seq}")
                        continue
                    labels = parse_image_list(gt_file)
                    kept = list_kept(clean_dir)
                    unknown = kept - set(labels.keys())
                    if unknown:
                        print(
                            f"[warn] {method}/{level}/{dataset}/{seq}: "
                            f"{len(unknown)} kept images not in GT list (ignored): "
                            f"{sorted(unknown)[:3]}..."
                        )
                    tp, fn, fp, tn = confusion(labels, kept)
                    m = metrics_from_cm(tp, fn, fp, tn)
                    per_seq_rows.append({
                        "method": method, "noise_level": level,
                        "dataset": dataset, "sequence": seq, **m,
                    })

    if missing:
        print(f"[info] {len(missing)} sequences missing clean_images folder")
        for m in missing[:10]:
            print(f"  - {m}")

    seq_cols = ["method", "noise_level", "dataset", "sequence", *METRIC_NAMES]
    write_markdown_table(OUT_DIR / "per_sequence_metrics.md", per_seq_rows, seq_cols,
                         f"Per-sequence metrics ({len(per_seq_rows)} sequences)")
    print(f"[ok] wrote {OUT_DIR / 'per_sequence_metrics.md'} ({len(per_seq_rows)} rows)")

    def aggregate(group_keys: tuple[str, ...]) -> list[dict]:
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for r in per_seq_rows:
            buckets[tuple(r[k] for k in group_keys)].append(r)
        rows = []
        for key, rs in sorted(buckets.items()):
            row = dict(zip(group_keys, key))
            row["n_sequences"] = len(rs)
            for m in METRIC_NAMES:
                row[m] = _mean([r[m] for r in rs])
            rows.append(row)
        return rows

    per_scene = aggregate(("method", "noise_level", "dataset"))
    write_markdown_table(
        OUT_DIR / "per_scene_metrics.md", per_scene,
        ["method", "noise_level", "dataset", "n_sequences", *METRIC_NAMES],
        "Per-scene metrics (averaged over sequences)")
    print(f"[ok] wrote {OUT_DIR / 'per_scene_metrics.md'}")

    per_method = aggregate(("method",))
    write_markdown_table(
        OUT_DIR / "per_method_metrics.md", per_method,
        ["method", "n_sequences", *METRIC_NAMES],
        "Per-method metrics (averaged over all sequences)")
    print(f"[ok] wrote {OUT_DIR / 'per_method_metrics.md'}")

    per_method_level = aggregate(("method", "noise_level"))
    write_markdown_table(
        OUT_DIR / "per_method_per_level_metrics.md", per_method_level,
        ["method", "noise_level", "n_sequences", *METRIC_NAMES],
        "Per-method x noise-level metrics")
    print(f"[ok] wrote {OUT_DIR / 'per_method_per_level_metrics.md'}")

    short_cols = ["method", *METRIC_NAMES, "n_sequences"]
    print_table(per_method, short_cols, "Per-method (averaged over all sequences)")
    print_table(per_method_level,
                ["method", "noise_level", *METRIC_NAMES, "n_sequences"],
                "Per-method x noise-level")


if __name__ == "__main__":
    main()
