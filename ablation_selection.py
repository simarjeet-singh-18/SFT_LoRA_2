"""
Alternative to SNIP-based block selection: instead of a gradient-based saliency
PROXY for "how important is this block", this directly measures each block's
ACTUAL effect on loss/accuracy by temporarily bypassing it (replacing it with an
identity pass-through) and re-running calibration data through the model, for
every block in turn.

Methodology:
  1. Run calibration data through the UNMODIFIED pretrained model once -> this is
     the baseline loss/accuracy. Either a capped number of batches (num_batches,
     default 6) or the ENTIRE dataloader (num_batches <= 0), depending on how this
     is called.
  2. For each block i: temporarily replace model.blocks[i] with nn.Identity()
     (safe here because every ViT block preserves the residual stream's dimension,
     so an identity swap doesn't break shapes), re-run the SAME calibration data,
     record loss/accuracy, then restore the original block.
  3. The block whose removal changes accuracy/loss the LEAST is judged least
     important -- i.e. the safest one to permanently replace with a filter block.

CAVEAT (same one SNIP itself has): this all runs on the model BEFORE any
fine-tuning, with the classifier head still at its fresh (randomly initialized for
this task's num_classes) state -- so absolute accuracy numbers will look
low/near-chance. What matters here is the RELATIVE change from removing each
block, not the absolute numbers, exactly as with SNIP's saliency scores. This
caveat holds regardless of how much calibration data is used -- using the full
dataset makes the measurement less NOISY (averaged over more samples), but
doesn't change the fact that the head hasn't been trained yet.

Kept as a separate module (rather than folded into snip_selection.py) so the two
selection strategies stay independently readable and swappable via
train_sfp_lora.py's --block-selection-method flag.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def _run_calibration_batches(model: nn.Module, dataloader, device: str, num_batches: int):
    """
    Runs calibration data through model in eval mode, returns
    (avg_loss, accuracy_pct, n_samples).

    num_batches > 0: caps at that many batches (original behavior).
    num_batches <= 0: no cap -- runs every batch in dataloader (full-dataset
    inference), for a less noisy but more expensive measurement.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0
    batches_used = 0
    use_full_dataset = num_batches <= 0
    for x, y in dataloader:
        if not use_full_dataset and batches_used >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)
        batches_used += 1
    avg_loss = total_loss / total_count if total_count > 0 else float("nan")
    accuracy = 100.0 * total_correct / total_count if total_count > 0 else float("nan")
    return avg_loss, accuracy, total_count


def select_block_with_ablation(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    num_batches: int = 6,
    keep: str = "low",
    return_scores: bool = False,
):
    """
    Selects the block whose removal (identity bypass) changes accuracy/loss the
    LEAST (keep='low', default -- matches SNIP's default convention: the least
    important/most "safe to replace" block) or MOST (keep='high').

    num_batches > 0 (default 6): cap calibration to that many batches per
    measurement (cheaper, noisier).
    num_batches <= 0: use the ENTIRE dataloader for every measurement (baseline
    AND all num_blocks per-block ablations) -- num_blocks+1 full passes over the
    dataset total. More expensive but less noisy; the "average over ALL data"
    version of this method rather than a small-sample estimate.

    Returns selected_idx, or (selected_idx, result_dict) if return_scores=True.
    result_dict = {
        "baseline_loss": float, "baseline_acc": float, "num_calibration_samples": int,
        "num_calibration_batches": int or "full",
        "per_block": {idx: {"loss_with_block_removed", "acc_with_block_removed",
                             "loss_increase", "acc_drop"}, ...},
        "selected_block": int,
    }

    Note: temporarily mutates model.blocks[i] for each i in turn (always restored
    in a try/finally before moving to the next block, and before returning), so the
    model is left exactly as it was passed in once this function returns.
    """
    model.to(device)
    use_full_dataset = num_batches <= 0
    baseline_loss, baseline_acc, n_samples = _run_calibration_batches(model, dataloader, device, num_batches)
    coverage_desc = f"the FULL dataset ({n_samples} sample(s), every batch)" if use_full_dataset \
        else f"{n_samples} calibration sample(s) ({num_batches} batch(es))"
    print(f"[Ablation Search] Baseline (no block removed) over {coverage_desc}: "
          f"loss={baseline_loss:.4f}, acc={baseline_acc:.2f}%")
    print("[Ablation Search] NOTE: measured on the PRETRAINED backbone with its ORIGINAL "
          "(not-yet-fine-tuned) classifier head, so absolute accuracy will look low/near-chance -- "
          "what matters is the RELATIVE change in loss/accuracy as each block is bypassed below, "
          "same caveat SNIP saliency itself has (also computed pre-training).")

    num_blocks = len(model.blocks)
    per_block = {}
    for idx in range(num_blocks):
        original_block = model.blocks[idx]
        model.blocks[idx] = nn.Identity()
        try:
            loss, acc, _ = _run_calibration_batches(model, dataloader, device, num_batches)
        finally:
            model.blocks[idx] = original_block  # always restore, even if the forward pass raised
        loss_increase = loss - baseline_loss
        acc_drop = baseline_acc - acc
        per_block[idx] = {
            "loss_with_block_removed": loss,
            "acc_with_block_removed": acc,
            "loss_increase": loss_increase,
            "acc_drop": acc_drop,
        }

    # keep='low': block whose removal causes the LEAST degradation is least
    # important. Rank primarily by acc_drop (the more directly interpretable
    # metric), tie-broken by loss_increase.
    sorted_blocks = sorted(per_block.items(), key=lambda kv: (kv[1]["acc_drop"], kv[1]["loss_increase"]))
    selected_idx = sorted_blocks[0][0] if keep == "low" else sorted_blocks[-1][0]

    print("[Ablation Search] Per-block ablation results (sorted by acc_drop, ascending = least important first):")
    for idx, scores in sorted_blocks:
        marker = "  <== SELECTED (least impact on accuracy)" if idx == selected_idx else ""
        print(f"  - Block {idx:02d}: acc_drop={scores['acc_drop']:+.2f}%, "
              f"loss_increase={scores['loss_increase']:+.4f}{marker}")
    print(f"[Ablation Search] Selected Block {selected_idx} (keep='{keep}')")

    result = {
        "baseline_loss": baseline_loss,
        "baseline_acc": baseline_acc,
        "num_calibration_samples": n_samples,
        "num_calibration_batches": "full" if use_full_dataset else num_batches,
        "per_block": per_block,
        "selected_block": selected_idx,
    }
    if return_scores:
        return selected_idx, result
    return selected_idx


if __name__ == "__main__":
    print("[ablation_selection] Execution script initialized.")