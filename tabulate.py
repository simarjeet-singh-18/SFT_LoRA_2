"""
Tabulate final validation & test accuracy across all training runs into CSV(s),
grouped by seed for easy comparison.

Each training run writes a `metrics_summary.json` into its own output folder
(./outputs/<run_name>/metrics_summary.json). This script walks that tree, reads
every summary it finds, derives a short human-readable config label for each run,
and writes:

  1. results_long.csv      -- one row per run (seed, config, val acc, test acc, ...),
                              sorted by (seed, config). Easiest to filter/pivot.
  2. results_by_seed.csv   -- a wide table: one row per config, one pair of
                              (val, test) columns per seed, plus a mean +/- std
                              across seeds. This is the "clear-cut comparison"
                              view: read across a row to see how one config does
                              on every seed, read down a column to compare configs
                              on a fixed seed.

It also prints a formatted table to stdout so you can eyeball results immediately.

Usage:
    python3 tabulate_results.py                     # scans ./outputs
    python3 tabulate_results.py --outputs-dir DIR   # scans a custom dir
    python3 tabulate_results.py --out-prefix runA   # -> runA_long.csv, runA_by_seed.csv
    python3 tabulate_results.py --dataset pets      # only include runs on this dataset

Robust to:
  - older metrics_summary.json formats (pruned_block_idx as int, missing fields)
  - runs that crashed before writing final_test_acc (shown as blank / NaN)
  - full_finetune runs and any --num-filter-blocks / adapter combos
It reads ONLY metrics_summary.json, not the SLURM .out/.err logs, so partial or
malformed log text can't corrupt the numbers.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict


def find_summary_files(outputs_dir: str):
    """Yield every metrics_summary.json path under outputs_dir (recursively)."""
    for root, _dirs, files in os.walk(outputs_dir):
        if "metrics_summary.json" in files:
            yield os.path.join(root, "metrics_summary.json")


def config_label(summary: dict) -> str:
    """
    Build a short, stable label describing the run's method/config, independent of
    seed. This is what groups runs together across seeds.

    Derived from the fields train_sfp_lora.py writes into the summary:
      - mode: one of full_finetune / sft_lora_ortho / sft_lora / sft
      - adapter_type: lora / dora  (None when no LoRA)
      - lora_rank: int or None
    Falls back gracefully if a field is missing in an older summary.
    """
    mode = summary.get("mode")
    adapter = summary.get("adapter_type")
    rank = summary.get("lora_rank")

    if mode == "full_finetune" or summary.get("full_finetune"):
        return "full_finetune"

    # No adapters -> paper's plain SFT.
    if not rank or rank <= 0:
        return "sft"

    adapter = (adapter or "lora").lower()
    ortho = (mode == "sft_lora_ortho")

    label = f"sft_{adapter}"                # sft_lora / sft_dora
    if ortho:
        label += "_ortho"                   # sft_lora_ortho / sft_dora_ortho
    return label


def _to_float(x):
    """Coerce to float, returning None for missing/None/non-numeric values."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(outputs_dir: str, dataset_filter: str = None):
    """
    Read every metrics_summary.json into a flat list of row dicts. Skips files that
    can't be parsed (warns), and optionally filters to a single dataset.
    """
    rows = []
    for path in sorted(find_summary_files(outputs_dir)):
        try:
            with open(path, "r") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] Skipping unreadable summary: {path} ({e})")
            continue

        dataset = summary.get("dataset")
        if dataset_filter is not None and dataset != dataset_filter:
            continue

        rows.append({
            "seed": summary.get("seed"),
            "dataset": dataset,
            "config": config_label(summary),
            "best_val_acc": _to_float(summary.get("best_val_acc")),
            "final_test_acc": _to_float(summary.get("final_test_acc")),
            "best_epoch": summary.get("best_epoch"),
            "epochs_trained": summary.get("epochs_trained"),
            "pruned_block_idx": summary.get("pruned_block_idx"),
            "run_dir": os.path.dirname(path),
        })
    return rows


def _mean_std(values):
    """Population-style mean and sample std (ddof=1) over non-None values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def write_long_csv(rows, path):
    """One row per run, sorted by (config, seed)."""
    fields = ["seed", "dataset", "config", "best_val_acc", "final_test_acc",
              "best_epoch", "epochs_trained", "pruned_block_idx", "run_dir"]
    ordered = sorted(
        rows,
        key=lambda r: (str(r["config"]), (r["seed"] if r["seed"] is not None else 1e9)),
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ordered:
            writer.writerow(r)


def write_by_seed_csv(rows, path):
    """
    Wide table: one row per config, columns for each seed's val & test acc, plus
    mean +/- std across seeds. This is the main comparison view.
    """
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    configs = sorted({r["config"] for r in rows})

    # (config, seed) -> row  (last one wins if duplicated; also collect dupes to warn)
    lookup = {}
    dupes = defaultdict(int)
    for r in rows:
        key = (r["config"], r["seed"])
        if key in lookup:
            dupes[key] += 1
        lookup[key] = r
    for (cfg, seed), n in dupes.items():
        print(f"[warn] {n+1} runs found for config='{cfg}' seed={seed}; "
              f"using the last one scanned. Check for duplicate run folders.")

    header = ["config"]
    for s in seeds:
        header += [f"val_seed{s}", f"test_seed{s}"]
    header += ["val_mean", "val_std", "test_mean", "test_std", "n_seeds"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cfg in configs:
            row = [cfg]
            val_list, test_list = [], []
            for s in seeds:
                r = lookup.get((cfg, s))
                v = r["best_val_acc"] if r else None
                t = r["final_test_acc"] if r else None
                val_list.append(v)
                test_list.append(t)
                row += [_fmt(v), _fmt(t)]
            v_mean, v_std = _mean_std(val_list)
            t_mean, t_std = _mean_std(test_list)
            n = sum(1 for x in test_list if x is not None)
            row += [_fmt(v_mean), _fmt(v_std), _fmt(t_mean), _fmt(t_std), n]
            writer.writerow(row)

    return seeds, configs, lookup


def _fmt(x, nd=2):
    """Format a float to nd decimals, or empty string for None."""
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def print_console_table(rows):
    """Human-readable grouped-by-seed summary to stdout."""
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    configs = sorted({r["config"] for r in rows})
    lookup = {(r["config"], r["seed"]): r for r in rows}

    if not rows:
        print("\n[tabulate] No metrics_summary.json files found. Nothing to report.")
        return

    col_w = max(18, max((len(c) for c in configs), default=18) + 2)

    print("\n" + "=" * 70)
    print("RESULTS BY SEED  (val% / test%)")
    print("=" * 70)
    for s in seeds:
        print(f"\nSeed {s}")
        print(f"  {'config'.ljust(col_w)} {'val_acc':>9}  {'test_acc':>9}  {'best_ep':>7}")
        print("  " + "-" * (col_w + 30))
        for cfg in configs:
            r = lookup.get((cfg, s))
            if r is None:
                print(f"  {cfg.ljust(col_w)} {'--':>9}  {'--':>9}  {'--':>7}   (missing)")
                continue
            v = _fmt(r["best_val_acc"])
            t = _fmt(r["final_test_acc"])
            be = r["best_epoch"] if r["best_epoch"] is not None else "--"
            print(f"  {cfg.ljust(col_w)} {v:>9}  {t:>9}  {str(be):>7}")

    # Mean +/- std across seeds per config.
    print("\n" + "=" * 70)
    print("MEAN +/- STD ACROSS SEEDS")
    print("=" * 70)
    print(f"  {'config'.ljust(col_w)} {'val (mean+/-std)':>20}  {'test (mean+/-std)':>20}  {'n':>3}")
    print("  " + "-" * (col_w + 48))
    for cfg in configs:
        vals = [lookup[(cfg, s)]["best_val_acc"] for s in seeds if (cfg, s) in lookup]
        tests = [lookup[(cfg, s)]["final_test_acc"] for s in seeds if (cfg, s) in lookup]
        v_mean, v_std = _mean_std(vals)
        t_mean, t_std = _mean_std(tests)
        n = sum(1 for x in tests if x is not None)
        v_str = f"{_fmt(v_mean)} +/- {_fmt(v_std)}" if v_mean is not None else "--"
        t_str = f"{_fmt(t_mean)} +/- {_fmt(t_std)}" if t_mean is not None else "--"
        print(f"  {cfg.ljust(col_w)} {v_str:>20}  {t_str:>20}  {n:>3}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", type=str, default="/export/home/achyut/Simarjeet/SFT_LoRA/outputs",
                        help="Root directory containing the per-run output folders "
                             "(each with a metrics_summary.json). Default: ./outputs")
    parser.add_argument("--out-prefix", type=str, default="results",
                        help="Prefix for the two output CSVs. Default: 'results' "
                             "-> results_long.csv, results_by_seed.csv")
    parser.add_argument("--dataset", type=str, default=None,
                        help="If set, only include runs whose dataset matches this "
                             "(e.g. 'pets'). Default: include all datasets found.")
    args = parser.parse_args()

    if not os.path.isdir(args.outputs_dir):
        parser.error(f"--outputs-dir '{args.outputs_dir}' does not exist or is not a directory.")

    rows = load_rows(args.outputs_dir, dataset_filter=args.dataset)
    print(f"[tabulate] Scanned '{args.outputs_dir}' -> found {len(rows)} run(s) "
          f"with a metrics_summary.json"
          + (f" (dataset='{args.dataset}')" if args.dataset else ""))

    long_path = f"{args.out_prefix}_long.csv"
    by_seed_path = f"{args.out_prefix}_by_seed.csv"

    write_long_csv(rows, long_path)
    write_by_seed_csv(rows, by_seed_path)
    print_console_table(rows)

    print(f"[tabulate] Wrote per-run table    -> {long_path}")
    print(f"[tabulate] Wrote by-seed pivot    -> {by_seed_path}")


if __name__ == "__main__":
    main()