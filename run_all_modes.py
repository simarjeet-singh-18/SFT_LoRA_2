"""
Sweeps all three training configurations this repo supports -- 'sft',
'full_finetune', and 'sft_lora_ortho' -- back to back, optionally across
several random seeds AND several datasets, and aggregates the results into
one summary file.

Each individual run is just `train_sfp_lora.py --mode <mode> --seed <seed>
--dataset <dataset> ...` invoked as a subprocess, so every run gets its own
output folder (named via run_naming.build_run_folder_name, exactly as if
you'd typed the command yourself) containing metrics_summary.json,
history.csv, and the usual plots. This script's only job is to (a) launch
that grid, (b) let each run train until ITS OWN early stopping fires
(--epochs -1 is train_sfp_lora.py's default, so "runs until convergence" is
already the behavior -- this script does not override --epochs unless you
pass it yourself via extra args), and (c) collect every run's
metrics_summary.json into one aggregate table, grouped by (dataset, mode).

USAGE
-----
Single dataset, single seed, all three modes:
    python run_all_modes.py --datasets pets

Three seeds x three modes on one dataset (9 runs), forwarding extra shared
flags to every run (everything after `--` is passed through verbatim to
train_sfp_lora.py):
    python run_all_modes.py --datasets pets --seeds 0 1 2 -- --num-samples 1000 --batch-size 32

Multiple datasets x multiple seeds x three modes (e.g. 2 datasets x 3 seeds
x 3 modes = 18 runs):
    python run_all_modes.py --datasets pets svhn dtd --seeds 0 1 2

--dataset (singular) still works as a one-item alias for --datasets, for
backward compatibility with the previous version of this script.

Output:
    <output-dir>/aggregate_summary.json   -- one row per (dataset, mode, seed)
                                              run, plus a per-(dataset, mode)
                                              mean/std block computed across seeds
    <output-dir>/aggregate_summary.xlsx   -- same data as an Excel workbook
                                              (2 sheets: "runs" and "per_group_stats"),
                                              if openpyxl is available
    <output-dir>/aggregate_summary.csv    -- same "runs" table as CSV, always
                                              written (works with or without
                                              openpyxl, and is easy to diff)
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

from run_naming import build_run_folder_name

MODES = ["sft", "full_finetune", "sft_lora_ortho"]

# Fields pulled out of each run's metrics_summary.json into the flat aggregate
# table. Keep in sync with the keys train_sfp_lora.py's `summary` dict writes.
SUMMARY_FIELDS = [
    "mode", "dataset", "seed", "num_samples",
    "best_val_acc", "best_epoch", "final_test_acc", "final_test_loss",
    "epochs_trained", "max_epochs", "patience", "lora_rank",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sft / full_finetune / sft_lora_ortho x N seeds x N datasets back to back and aggregate results.")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                         help="One or more datasets to sweep, e.g. --datasets pets svhn dtd. "
                              "Each dataset gets the full seeds x modes grid.")
    parser.add_argument("--dataset", type=str, default=None,
                         help="Singular alias for --datasets with exactly one dataset "
                              "(backward compatible with the previous version of this script). "
                              "Ignored if --datasets is also given.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                         help="One or more seeds to repeat the full mode sweep with, e.g. --seeds 0 1 2. "
                              "Default is a single run at seed 42.")
    parser.add_argument("--modes", type=str, nargs="+", default=MODES, choices=MODES,
                         help="Which of the three modes to include in the sweep. Default: all three.")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                         help="Same root output dir train_sfp_lora.py uses. Each run gets its own "
                              "subfolder under here (named from its own CLI args); this script's "
                              "aggregate files are written directly under this root.")
    parser.add_argument("--python", type=str, default=sys.executable,
                         help="Python interpreter to launch each training run with.")
    parser.add_argument("--train-script", type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_sfp_lora.py"))
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the commands that would be run without actually running them.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                         help="Everything after `--` is forwarded verbatim to every train_sfp_lora.py "
                              "invocation (e.g. -- --num-samples 1000 --batch-size 32 --lora-rank 8). "
                              "Do not pass --mode, --seed, --dataset, or --output-dir here -- this "
                              "script already sets those per-run.")
    args = parser.parse_args()

    # argparse.REMAINDER keeps a leading "--" token if present; strip it.
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    reserved = {"--mode", "--seed", "--dataset", "--output-dir"}
    for tok in args.extra_args:
        if tok in reserved or any(tok.startswith(r + "=") for r in reserved):
            parser.error(f"--extra-args must not set {tok!r}; this script already controls that flag per-run.")

    if args.datasets:
        pass  # use as given
    elif args.dataset:
        args.datasets = [args.dataset]
    else:
        args.datasets = ["pets"]

    return args


def collect_run_summary(output_root: str, run_name: str, dataset: str, mode: str, seed: int) -> dict:
    """
    Loads <output_root>/<run_name>/metrics_summary.json (written by
    plotting.save_metrics_summary at the end of a train_sfp_lora.py run) and
    flattens the fields we care about for the aggregate table.
    """
    summary_path = os.path.join(output_root, run_name, "metrics_summary.json")
    if not os.path.exists(summary_path):
        return {
            "dataset": dataset, "mode": mode, "seed": seed, "run_name": run_name,
            "status": "missing_summary",
            "summary_path": summary_path,
        }

    with open(summary_path, "r") as f:
        full_summary = json.load(f)

    row = {"status": "ok", "run_name": run_name, "seed": seed}
    for field in SUMMARY_FIELDS:
        row[field] = full_summary.get(field)
    # `mode`/`dataset` in metrics_summary.json only exist on runs made with
    # this version of train_sfp_lora.py; fall back to what we launched this
    # run with if either is missing (e.g. an older summary format).
    row["mode"] = full_summary.get("mode", mode)
    row["dataset"] = full_summary.get("dataset", dataset)
    return row


def compute_per_group_stats(rows: list, datasets: list, modes: list) -> dict:
    """
    Groups successful runs by (dataset, mode) and computes mean/std of
    best_val_acc and final_test_acc across seeds, so a multi-seed x
    multi-dataset x multi-mode sweep gives you the usual "mean +/- std over
    seeds, per dataset, per mode" comparison table directly, instead of
    having to open every individual run folder.

    Returns a dict keyed by "dataset__mode" (flat, so it serializes cleanly
    to both JSON and a single Excel sheet).
    """
    stats = {}
    for dataset in datasets:
        for mode in modes:
            group_rows = [r for r in rows
                          if r.get("dataset") == dataset and r.get("mode") == mode and r.get("status") == "ok"]
            if not group_rows:
                continue
            val_accs = [r["best_val_acc"] for r in group_rows if r.get("best_val_acc") is not None]
            test_accs = [r["final_test_acc"] for r in group_rows if r.get("final_test_acc") is not None]
            key = f"{dataset}__{mode}"
            stats[key] = {
                "dataset": dataset,
                "mode": mode,
                "n_runs": len(group_rows),
                "seeds": [r["seed"] for r in group_rows],
                "best_val_acc_mean": statistics.mean(val_accs) if val_accs else None,
                "best_val_acc_std": statistics.stdev(val_accs) if len(val_accs) > 1 else 0.0 if val_accs else None,
                "final_test_acc_mean": statistics.mean(test_accs) if test_accs else None,
                "final_test_acc_std": statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0 if test_accs else None,
            }
    return stats


def write_aggregate_outputs(output_dir: str, rows: list, per_group_stats: dict):
    os.makedirs(output_dir, exist_ok=True)

    # --- JSON (always written; primary machine-readable output) ---
    json_path = os.path.join(output_dir, "aggregate_summary.json")
    with open(json_path, "w") as f:
        json.dump({"runs": rows, "per_group_stats": per_group_stats}, f, indent=2)
    print(f"[Sweep] Aggregate JSON -> {json_path}")

    # --- CSV (always written; no extra dependency, easiest to diff/eyeball) ---
    csv_path = os.path.join(output_dir, "aggregate_summary.csv")
    if rows:
        all_keys = []
        for r in rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Sweep] Aggregate CSV -> {csv_path}")

    # --- Excel (best-effort; only if pandas + openpyxl are installed) ---
    try:
        import pandas as pd
        xlsx_path = os.path.join(output_dir, "aggregate_summary.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="runs", index=False)
            stats_rows = list(per_group_stats.values())
            pd.DataFrame(stats_rows).to_excel(writer, sheet_name="per_group_stats", index=False)
        print(f"[Sweep] Aggregate Excel -> {xlsx_path}")
    except ImportError:
        print("[Sweep] Skipped .xlsx output (pandas/openpyxl not installed) -- "
              "aggregate_summary.json and .csv above have the same data.")


def main():
    args = parse_args()
    rows = []

    total_runs = len(args.datasets) * len(args.seeds) * len(args.modes)
    run_num = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            for mode in args.modes:
                run_num += 1
                argv = ["--dataset", dataset, "--mode", mode, "--seed", str(seed),
                        "--output-dir", args.output_dir] + args.extra_args
                run_name = build_run_folder_name(argv)
                cmd = [args.python, args.train_script] + argv

                print(f"\n[Sweep] ({run_num}/{total_runs}) dataset={dataset} mode={mode} seed={seed}")
                print(f"[Sweep] Command: {' '.join(cmd)}")
                print(f"[Sweep] Output folder: {os.path.join(args.output_dir, run_name)}")

                if args.dry_run:
                    continue

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"[Sweep] WARNING: run failed (exit code {result.returncode}); "
                          f"continuing with remaining runs.")
                    rows.append({"dataset": dataset, "mode": mode, "seed": seed, "run_name": run_name,
                                 "status": "failed", "returncode": result.returncode})
                    continue

                rows.append(collect_run_summary(args.output_dir, run_name, dataset, mode, seed))

    if args.dry_run:
        print(f"\n[Sweep] --dry-run: {total_runs} run(s) listed above, none executed.")
        return

    per_group_stats = compute_per_group_stats(rows, args.datasets, args.modes)
    write_aggregate_outputs(args.output_dir, rows, per_group_stats)

    print("\n==================================================")
    print(f"[Sweep] Completed {len(rows)}/{total_runs} run(s).")
    for key, stats in per_group_stats.items():
        val = stats["best_val_acc_mean"]
        test = stats["final_test_acc_mean"]
        val_str = f"{val:.2f}%" if val is not None else "n/a"
        test_str = f"{test:.2f}%" if test is not None else "n/a"
        print(f"[Sweep]   {stats['dataset']:14s} / {stats['mode']:16s} (n={stats['n_runs']}): "
              f"best_val_acc={val_str} | final_test_acc={test_str}")
    print("==================================================")


if __name__ == "__main__":
    main()
