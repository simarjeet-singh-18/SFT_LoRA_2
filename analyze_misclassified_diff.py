"""
Compares misclassified-image sets ACROSS configs, using the sample_index column
already logged in each config's misclassified_log.csv (written by
evaluate_and_save_misclassified in train_sfp_lora.py).

Motivation: with SFT-only as the baseline, you don't care about images every
config gets wrong (the intersection) -- those are probably just hard for this
model/dataset regardless of method. What's actually diagnostic is:

  EXTRA (baseline correct, this config wrong):
      images the baseline (SFT-only) got RIGHT that a comparison config (e.g.
      SFT+LoRA, SFT+DoRA) got WRONG -- mistakes that method specifically
      INTRODUCED. This is usually the more important set to look at.

  RECOVERED (baseline wrong, this config correct):
      images the baseline got WRONG that the comparison config got RIGHT --
      i.e. what that method actually fixed.

  INTERSECTION (both wrong) is computed but not copied out anywhere, per your
  framing that it's not the interesting part.

CORRECTNESS CAVEAT: this relies on `sample_index` meaning the same test image
across every config's run. That holds as long as every run being compared used
the same --dataset and the same effective test-set order (test_loader is
constructed with shuffle=False in this repo, so order only depends on the
dataset itself, not --seed -- but double check this assumption still holds if
you ever change how the test split/loader is built). All configs sharing the
same --dataset are safe to run this on.

USAGE
-----
    python3 analyze_misclassified_diff.py \\
        --output-dir /export/home/achyut/Sarvesh/SFT_FINAL_CODE/outputs \\
        --baseline folder_sft_seed_0 \\
        --compare folder_sft_lora_seed_0 folder_sft_lora_orthogonal_seed_0 \\
                  folder_sft_dora_seed_0_ablation folder_sft_dora_orthogonal_seed_0_ablation \\
                  folder_sft_dora_seed_0_ablation_paramcomp

For each --compare folder, this:
  1. Prints a summary: |baseline|, |compare|, |intersection|, |extra|, |recovered|
  2. Copies the "extra" images (the ones worth looking at) into
     <output-dir>/<compare_folder>_extra_vs_baseline/, alongside a
     extra_vs_baseline.csv listing them (same columns as the original CSV, so
     you keep true/predicted labels + confidence for each).
  3. Does the same for "recovered" images, into
     <output-dir>/<compare_folder>_recovered_vs_baseline/ (useful context, even
     though "extra" is the set you said you care about most).

Doesn't touch or require re-running any training -- purely reads the CSVs and
image files that already exist in --output-dir from your runs.
"""

import argparse
import csv
import os
import shutil


def load_misclassified_log(folder: str) -> dict:
    """
    Returns {sample_index: row_dict} for every row in folder/misclassified_log.csv.
    row_dict keeps all original columns (filename, true/pred label, confidence).
    """
    csv_path = os.path.join(folder, "misclassified_log.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"No misclassified_log.csv found in {folder} -- was this run launched "
            f"with --save-misclassified-images? (it defaults to on, so this "
            f"usually means the folder path itself is wrong -- double check "
            f"--output-dir and the folder name.)"
        )
    rows = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row["sample_index"])] = row
    return rows


def copy_subset(rows: dict, indices: set, src_folder: str, dst_folder: str):
    os.makedirs(dst_folder, exist_ok=True)
    csv_path = os.path.join(dst_folder, os.path.basename(dst_folder) + ".csv")
    fieldnames = ["filename", "sample_index", "true_label_idx", "true_label_name",
                  "pred_label_idx", "pred_label_name", "confidence"]
    copied = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in sorted(indices):
            row = rows[idx]
            src_path = os.path.join(src_folder, row["filename"])
            if not os.path.isfile(src_path):
                print(f"    [warn] {src_path} listed in CSV but not found on disk -- skipping copy "
                      f"(row still included in the CSV below).")
            else:
                shutil.copy2(src_path, os.path.join(dst_folder, row["filename"]))
                copied += 1
            writer.writerow(row)
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Diff misclassified-image sets between a baseline config and one or more comparison configs.")
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Root output dir containing the misclassified folders (same --output-dir "
                              "you passed to train_sfp_lora.py).")
    parser.add_argument("--baseline", type=str, required=True,
                         help="Baseline folder name, e.g. folder_sft_seed_0 (SFT-only).")
    parser.add_argument("--compare", type=str, nargs="+", required=True,
                         help="One or more comparison folder names to diff against the baseline, e.g. "
                              "folder_sft_lora_seed_0 folder_sft_dora_orthogonal_seed_0_ablation ...")
    args = parser.parse_args()

    baseline_folder = os.path.join(args.output_dir, args.baseline)
    baseline_rows = load_misclassified_log(baseline_folder)
    baseline_set = set(baseline_rows.keys())
    print(f"[Diff] Baseline: {args.baseline}  ({len(baseline_set)} misclassified images)")

    for compare_name in args.compare:
        compare_folder = os.path.join(args.output_dir, compare_name)
        compare_rows = load_misclassified_log(compare_folder)
        compare_set = set(compare_rows.keys())

        intersection = baseline_set & compare_set
        extra = compare_set - baseline_set          # baseline got right, this config got wrong
        recovered = baseline_set - compare_set       # baseline got wrong, this config got right

        print(f"\n[Diff] {compare_name}")
        print(f"    |baseline|      = {len(baseline_set)}")
        print(f"    |compare|       = {len(compare_set)}")
        print(f"    |intersection|  = {len(intersection)}  (both wrong -- likely just hard examples, skipped)")
        print(f"    |extra|         = {len(extra)}  (baseline RIGHT, this config WRONG -- mistakes this "
              f"method introduced)")
        print(f"    |recovered|     = {len(recovered)}  (baseline WRONG, this config RIGHT -- mistakes "
              f"this method fixed)")

        if extra:
            extra_dir = os.path.join(args.output_dir, f"{compare_name}_extra_vs_baseline")
            n_copied = copy_subset(compare_rows, extra, compare_folder, extra_dir)
            print(f"    -> copied {n_copied}/{len(extra)} 'extra' images -> {extra_dir}")
        else:
            print(f"    -> no 'extra' images (this config didn't get anything wrong that baseline got right)")

        if recovered:
            recovered_dir = os.path.join(args.output_dir, f"{compare_name}_recovered_vs_baseline")
            n_copied = copy_subset(baseline_rows, recovered, baseline_folder, recovered_dir)
            print(f"    -> copied {n_copied}/{len(recovered)} 'recovered' images -> {recovered_dir}")


if __name__ == "__main__":
    main()
