#!/usr/bin/env python3
"""
analyze_results.py -- collect and summarize sweep results.

Walks a results tree (default: ./logs2) for every metrics_summary.json produced by
train_sfp_lora.py and pulls out the metrics that matter for the method comparison:

    training time   -> efficiency.train_compute_seconds (+ train_wall_seconds)
    param count     -> efficiency.trainable_params / total_params / trainable_pct
    val accuracy    -> best_val_acc
    test accuracy   -> final_test_acc
    peak GPU memory -> efficiency.peak_gpu_mem_reserved_mb (+ allocated)
    throughput      -> efficiency.steady_state_throughput_samples_per_sec

Outputs (written next to --root by default):
  * results_all.csv            -- one row per run (every seed/dataset/method/block)
  * results_by_method.csv      -- aggregated per (dataset, method) over blocks & seeds
  * results_pivot_testacc.csv  -- method x dataset table of mean / best-block test acc
And prints readable tables to stdout.

Stdlib only -- no pandas required (runs anywhere the training env runs).

Examples:
  python3 analyze_results.py
  python3 analyze_results.py --root logs2 --datasets cifar100,svhn --metric test_acc
  python3 analyze_results.py --seeds 18 --sort-by test_acc
"""
import os
import re
import csv
import json
import glob
import argparse
import statistics
from collections import defaultdict, OrderedDict


# ------------------------------------------------------------------ extraction

def _get(d, *keys, default=None):
    """Nested get: _get(summary, 'efficiency', 'trainable_params')."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def method_label_from_json(s):
    """Reconstruct the short method label from summary fields (fallback when the
    directory layout isn't the expected logs2/<seed>/<dataset>/<method>/...)."""
    if s.get("full_finetune"):
        return "full_ft"
    at = s.get("adapter_type")
    lr = s.get("lora_rank") or 0
    tuner = s.get("paca_tuner")
    l1 = s.get("lora_ortho_lambda1")
    lam = ("_l" + _fmt_lambda(l1)) if l1 else ""
    if at in ("unilora", "unidora"):
        # SFP + Uni-LoRA / Uni-DoRA (one shared subspace vector)
        return "unisflora" if at == "unilora" else "unisfdora"
    if at in ("paca", "rpaca"):
        if tuner in ("lora", "dora"):
            base = "sfdoca" if at == "paca" else "sfrdoca"   # fused
            return base + lam
        return "sfpaca" if at == "paca" else "sfrpaca"       # direct
    if at in ("lora", "dora") and lr > 0:
        return f"sf{at}{lam if lam else '_noortho'}"
    return "sft"


def _fmt_lambda(x):
    try:
        f = float(x)
        return ("%g" % f)
    except (TypeError, ValueError):
        return str(x)


def method_from_path(path, dataset, root):
    """The path segment right after the <dataset> directory is the method label
    (matches how run_sweep.slurm lays out logs2/<seed>/<dataset>/<method>/...)."""
    parts = os.path.normpath(os.path.relpath(path, root)).split(os.sep)
    if dataset and dataset in parts:
        i = parts.index(dataset)
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_run(path, root):
    """Parse one metrics_summary.json into a flat row dict, or None if unreadable."""
    try:
        with open(path) as f:
            s = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    block = s.get("pruned_block_idx")
    if isinstance(block, (list, tuple)):
        block = block[0] if block else None

    dataset = s.get("dataset")
    method = method_from_path(path, dataset, root) or method_label_from_json(s)

    thr = _get(s, "efficiency", "steady_state_throughput_samples_per_sec")
    if thr is None:
        thr = _get(s, "efficiency", "throughput_samples_per_sec")
    gpu = _get(s, "efficiency", "peak_gpu_mem_reserved_mb")
    if gpu is None:
        gpu = _get(s, "efficiency", "peak_gpu_mem_allocated_mb")

    return OrderedDict([
        ("seed", s.get("seed")),
        ("dataset", dataset),
        ("method", method),
        ("block", block),
        ("mode", s.get("mode")),
        ("val_acc", s.get("best_val_acc")),
        ("test_acc", s.get("final_test_acc")),
        ("test_loss", s.get("final_test_loss")),
        ("trainable_params", _get(s, "efficiency", "trainable_params")),
        ("total_params", _get(s, "efficiency", "total_params")),
        ("trainable_pct", _get(s, "efficiency", "trainable_pct")),
        ("train_compute_s", _get(s, "efficiency", "train_compute_seconds")),
        ("train_wall_s", _get(s, "efficiency", "train_wall_seconds")),
        ("throughput_sps", thr),
        ("peak_gpu_mem_mb", gpu),
        ("epochs_trained", s.get("epochs_trained")),
        ("device", _get(s, "efficiency", "device")),
        ("path", os.path.dirname(path)),
    ])


# ------------------------------------------------------------------ aggregation

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.mean(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def aggregate(rows):
    """Group by (dataset, method); aggregate over blocks and seeds."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"])].append(r)

    out = []
    for (ds, method), rs in groups.items():
        tests = [(r["test_acc"], r["block"], r["seed"]) for r in rs
                 if isinstance(r["test_acc"], (int, float))]
        best_test, best_block, best_seed = (max(tests) if tests else (None, None, None))
        out.append(OrderedDict([
            ("dataset", ds),
            ("method", method),
            ("n_runs", len(rs)),
            ("test_acc_mean", _mean([r["test_acc"] for r in rs])),
            ("test_acc_std", _std([r["test_acc"] for r in rs])),
            ("test_acc_best", best_test),
            ("best_block", best_block),
            ("best_seed", best_seed),
            ("val_acc_mean", _mean([r["val_acc"] for r in rs])),
            ("trainable_params", _mean([r["trainable_params"] for r in rs])),
            ("trainable_pct", _mean([r["trainable_pct"] for r in rs])),
            ("train_compute_s_mean", _mean([r["train_compute_s"] for r in rs])),
            ("throughput_sps_mean", _mean([r["throughput_sps"] for r in rs])),
            ("peak_gpu_mem_mb_mean", _mean([r["peak_gpu_mem_mb"] for r in rs])),
        ]))
    out.sort(key=lambda d: (str(d["dataset"]), -(d["test_acc_mean"] or -1)))
    return out


# ------------------------------------------------------------------ formatting

def _f(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _big(x):
    return "-" if x is None else f"{int(round(x)):,}"


def print_table(rows, cols, headers=None, aligns=None):
    headers = headers or cols
    aligns = aligns or ["<"] * len(cols)
    cells = [[str(_render(r.get(c))) for c in cols] for r in rows]
    widths = [max(len(headers[i]), *(len(row[i]) for row in cells)) if cells
              else len(headers[i]) for i in range(len(cols))]
    line = "  ".join(f"{headers[i]:{aligns[i]}{widths[i]}}" for i in range(len(cols)))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(cols))))
    for row in cells:
        print("  ".join(f"{row[i]:{aligns[i]}{widths[i]}}" for i in range(len(cols))))


def _render(x):
    if isinstance(x, float):
        return f"{x:.2f}"
    if x is None:
        return "-"
    return x


# ------------------------------------------------------------------ pivot

def pivot_testacc(agg):
    """method x dataset table of (mean test acc, best-block test acc)."""
    methods = sorted({a["method"] for a in agg})
    datasets = sorted({a["dataset"] for a in agg})
    mean_tbl = {(a["method"], a["dataset"]): a["test_acc_mean"] for a in agg}
    best_tbl = {(a["method"], a["dataset"]): a["test_acc_best"] for a in agg}
    return methods, datasets, mean_tbl, best_tbl


# ------------------------------------------------------------------ plotting

# Canonical left-to-right ordering of methods on the x-axis so plots are directly
# comparable across datasets. Unknown methods are appended alphabetically.
PREFERRED_METHOD_ORDER = [
    "sft",
    "sflora_l0.9", "sflora_l1.5", "sfdora_l0.9", "sfdora_l1.5",
    "sfpaca", "sfrpaca", "sfdoca", "sfrdoca", "sfdoca_ortho", "sfrdoca_ortho",
    "unisflora", "unisfdora",
]


def _method_order_key(m):
    return (PREFERRED_METHOD_ORDER.index(m), "") if m in PREFERRED_METHOD_ORDER else (len(PREFERRED_METHOD_ORDER), m)


TIME_METRICS = {
    # choice -> (per-row value function, axis label, "higher is faster?")
    "compute":    (lambda r: r["train_compute_s"],                     "train compute (s)",        False),
    "wall":       (lambda r: r["train_wall_s"],                        "train wall (s)",           False),
    "per_epoch":  (lambda r: (r["train_compute_s"] / r["epochs_trained"]
                              if r.get("train_compute_s") and r.get("epochs_trained") else None),
                   "compute per epoch (s/epoch)", False),
    "throughput": (lambda r: r["throughput_sps"],                      "throughput (samples/s)",   True),
}


def best_block_per_method(rows, metric="compute"):
    """
    For each (dataset, method), pick the block with the highest test accuracy
    (averaging over seeds first when there are several), and return that block's
    accuracy, id, secondary time/speed metric and trainable-param %.

    `metric` selects the secondary curve (see TIME_METRICS):
      compute    - total train-loop seconds (noisy: scales with early-stopped epochs)
      wall       - total wall-clock seconds
      per_epoch  - seconds per epoch = compute / epochs_trained (removes the epoch-count
                   noise; ~2% stable within a method)
      throughput - samples/sec (hardware-fair speed; higher = faster; ~2% stable)

    Returns: { dataset: [ {method, block, test_acc, time_s, trainable_pct, n_seeds}, ... ] }
    ordered by PREFERRED_METHOD_ORDER. The secondary value is stored under "time_s"
    regardless of which metric was chosen.
    """
    value_fn = TIME_METRICS[metric][0]
    by_dmb = defaultdict(list)  # (dataset, method, block) -> runs (one per seed)
    for r in rows:
        if r["test_acc"] is None or r["block"] is None:
            continue
        by_dmb[(r["dataset"], r["method"], r["block"])].append(r)

    per_block = {}
    for (ds, method, block), rs in by_dmb.items():
        per_block[(ds, method, block)] = {
            "test_acc": _mean([r["test_acc"] for r in rs]),
            "time_s": _mean([value_fn(r) for r in rs]),
            "trainable_pct": _mean([r["trainable_pct"] for r in rs]),
            "n_seeds": len({r["seed"] for r in rs}),
        }

    best = defaultdict(dict)  # dataset -> method -> record (best block)
    for (ds, method, block), v in per_block.items():
        cur = best[ds].get(method)
        if cur is None or (v["test_acc"] is not None and v["test_acc"] > cur["test_acc"]):
            best[ds][method] = {"method": method, "block": block, **v}

    out = {}
    for ds, mdict in best.items():
        out[ds] = sorted(mdict.values(), key=lambda d: _method_order_key(d["method"]))
    return out


def plot_dataset(dataset, records, out_path, time_field_label="train compute (s)"):
    """
    One figure per dataset: best test accuracy per method (left y-axis, with the
    winning block id annotated at each point) and the corresponding training time
    (right y-axis, different color). X-axis = methods, each tick labelled with the
    method name and its (constant-per-method) trainable-parameter percentage.
    Returns True if the plot was written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe, matches plotting.py
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable ({e}); skipping plots.")
        return False

    if not records:
        return False

    methods = [r["method"] for r in records]
    xs = list(range(len(methods)))
    accs = [r["test_acc"] for r in records]
    times = [r["time_s"] for r in records]
    blocks = [r["block"] for r in records]
    pcts = [r["trainable_pct"] for r in records]

    ACC_COLOR = "#1f77b4"   # blue
    TIME_COLOR = "#d62728"  # red

    fig, ax_acc = plt.subplots(figsize=(max(8, 1.15 * len(methods)), 6))
    ax_time = ax_acc.twinx()

    # Accuracy line (left axis) + block-id annotation at each point
    ax_acc.plot(xs, accs, "-o", color=ACC_COLOR, linewidth=2, markersize=7,
                label="best test accuracy", zorder=3)
    for x, a, b in zip(xs, accs, blocks):
        if a is None:
            continue
        ax_acc.annotate(f"blk {b}", (x, a), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=ACC_COLOR, fontweight="bold")

    # Time line (right axis, different color)
    ax_time.plot(xs, times, "--s", color=TIME_COLOR, linewidth=1.8, markersize=6,
                 label=time_field_label, zorder=2)

    # X ticks: method + constant trainable-param %
    labels = []
    for m, p in zip(methods, pcts):
        pct = f"{p:.2f}%" if isinstance(p, (int, float)) else "-"
        labels.append(f"{m}\n({pct})")
    ax_acc.set_xticks(xs)
    ax_acc.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax_acc.set_ylabel("best test accuracy (%)", color=ACC_COLOR)
    ax_acc.tick_params(axis="y", labelcolor=ACC_COLOR)
    ax_time.set_ylabel(time_field_label, color=TIME_COLOR)
    ax_time.tick_params(axis="y", labelcolor=TIME_COLOR)
    ax_acc.set_xlabel("method  (trainable-param % in parentheses)")
    ax_acc.set_title(f"{dataset}: best accuracy vs {time_field_label} per method")
    ax_acc.grid(True, axis="y", linestyle=":", alpha=0.4)

    lines = ax_acc.get_lines() + ax_time.get_lines()
    ax_acc.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Summarize SFP sweep results.")
    ap.add_argument("--root", default="/export/home/achyut/Simarjeet/SFT_LoRA_2/logs2", help="Results tree to scan (default: logs2)")
    ap.add_argument("--out-dir", default=None, help="Where to write CSVs (default: --root)")
    ap.add_argument("--datasets", default=None, help="Comma-separated filter, e.g. cifar100,svhn")
    ap.add_argument("--seeds", default=None, help="Comma-separated seed filter, e.g. 18,42")
    ap.add_argument("--methods", default=None, help="Comma-separated method filter, e.g. sft,sfpaca")
    ap.add_argument("--metric", default="test_acc",
                    choices=["test_acc", "val_acc"], help="Metric for the pivot table")
    ap.add_argument("--sort-by", default="test_acc",
                    help="Column to sort the per-run CSV by (default: test_acc)")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip the per-dataset accuracy-vs-time plots.")
    ap.add_argument("--plot-time", default="throughput",
                    choices=["compute", "wall", "per_epoch", "throughput"],
                    help="Which speed/cost metric to draw as the second curve on the per-dataset "
                         "plots. 'throughput' (samples/s, default) and 'per_epoch' (compute/epoch) are "
                         "hardware-fair and ~2%% stable; 'compute'/'wall' are total training seconds, "
                         "which are noisy because early stopping ends runs at very different epoch counts.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    paths = glob.glob(os.path.join(args.root, "**", "metrics_summary.json"), recursive=True)
    if not paths:
        print(f"No metrics_summary.json found under '{args.root}'. Nothing to do.")
        return

    rows = [r for r in (load_run(p, args.root) for p in paths) if r]

    # Optional filters
    def keep(r):
        if args.datasets and r["dataset"] not in args.datasets.split(","):
            return False
        if args.seeds and str(r["seed"]) not in args.seeds.split(","):
            return False
        if args.methods and r["method"] not in args.methods.split(","):
            return False
        return True
    rows = [r for r in rows if keep(r)]
    if not rows:
        print("No runs matched the given filters.")
        return

    # Sort per-run rows
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["method"]), r["seed"] or 0,
                             r["block"] if isinstance(r["block"], int) else 99))

    # ---- results_all.csv ----
    all_csv = os.path.join(out_dir, "results_all.csv")
    field_order = [k for k in rows[0].keys()]
    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- aggregate ----
    agg = aggregate(rows)
    by_method_csv = os.path.join(out_dir, "results_by_method.csv")
    with open(by_method_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        for a in agg:
            w.writerow(a)

    # ---- pivot ----
    methods, datasets, mean_tbl, best_tbl = pivot_testacc(agg)
    pivot_csv = os.path.join(out_dir, "results_pivot_testacc.csv")
    with open(pivot_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{d}_mean" for d in datasets]
                   + [f"{d}_best" for d in datasets])
        for m in methods:
            w.writerow([m]
                       + [_f(mean_tbl.get((m, d))) for d in datasets]
                       + [_f(best_tbl.get((m, d))) for d in datasets])

    # ---- console report ----
    print(f"\nScanned {len(paths)} summary file(s); {len(rows)} run(s) after filters.")
    print(f"Datasets: {sorted({r['dataset'] for r in rows})}")
    print(f"Seeds:    {sorted({r['seed'] for r in rows})}")
    print(f"Methods:  {sorted({r['method'] for r in rows})}")

    cols = ["method", "n_runs", "test_acc_mean", "test_acc_std", "test_acc_best",
            "best_block", "val_acc_mean", "trainable_params", "trainable_pct",
            "train_compute_s_mean", "throughput_sps_mean", "peak_gpu_mem_mb_mean"]
    headers = ["method", "n", "test~mean", "±std", "test~best", "blk", "val~mean",
               "train~prm", "train%", "t_train(s)", "thru(sps)", "gpu(MB)"]
    aligns = ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"]

    for ds in sorted({a["dataset"] for a in agg}):
        print(f"\n=== {ds} ===")
        sub = [a for a in agg if a["dataset"] == ds]
        disp = []
        for a in sub:
            d = dict(a)
            d["trainable_params"] = _big(a["trainable_params"])
            disp.append(d)
        print_table(disp, cols, headers, aligns)

    # Per-method overall (averaged across datasets & blocks & seeds)
    print("\n=== overall (averaged across datasets) ===")
    permethod = defaultdict(list)
    for r in rows:
        permethod[r["method"]].append(r)
    overall = []
    for m, rs in permethod.items():
        overall.append(OrderedDict([
            ("method", m),
            ("n_runs", len(rs)),
            ("test_acc_mean", _mean([r["test_acc"] for r in rs])),
            ("val_acc_mean", _mean([r["val_acc"] for r in rs])),
            ("trainable_params", _big(_mean([r["trainable_params"] for r in rs]))),
            ("trainable_pct", _mean([r["trainable_pct"] for r in rs])),
            ("train_compute_s_mean", _mean([r["train_compute_s"] for r in rs])),
            ("throughput_sps_mean", _mean([r["throughput_sps"] for r in rs])),
            ("peak_gpu_mem_mb_mean", _mean([r["peak_gpu_mem_mb"] for r in rs])),
        ]))
    overall.sort(key=lambda d: -(d["test_acc_mean"] or -1))
    ocols = ["method", "n_runs", "test_acc_mean", "val_acc_mean", "trainable_params",
             "trainable_pct", "train_compute_s_mean", "throughput_sps_mean", "peak_gpu_mem_mb_mean"]
    oheaders = ["method", "n", "test~mean", "val~mean", "train~prm", "train%",
                "t_train(s)", "thru(sps)", "gpu(MB)"]
    print_table(overall, ocols, oheaders, ["<", ">", ">", ">", ">", ">", ">", ">", ">"])

    print(f"\nWrote:\n  {all_csv}\n  {by_method_csv}\n  {pivot_csv}")

    # ---- per-dataset accuracy-vs-time plots ----
    if not args.no_plots:
        _, time_label, _ = TIME_METRICS[args.plot_time]
        best = best_block_per_method(rows, metric=args.plot_time)
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        written = []
        for ds in sorted(best):
            out_path = os.path.join(plots_dir, f"{ds}_acc_vs_time.png")
            if plot_dataset(ds, best[ds], out_path, time_field_label=time_label):
                written.append(out_path)
        if written:
            print("Plots:")
            for p in written:
                print(f"  {p}")


if __name__ == "__main__":
    main()