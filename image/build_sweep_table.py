"""
Build a ranked comparison of the run_rsna_sweep.sh runs using each run's
ACTUAL checkpointed val/mean_auroc (i.e. the value ModelCheckpoint selected
on, mode="max" across the whole run) rather than whatever value a run's
curve happens to show at the last logged TensorBoard step. For a run that
peaked early and declined since, those two numbers can differ a lot -- the
checkpoint (and therefore test_preds.csv) reflects the peak, not the
endpoint.

ModelCheckpoint's filename template is "{epoch}-{val/mean_auroc:.3f}". The
metric name val/mean_auroc contains a literal "/", which Lightning's naive
string formatting turns into a real subdirectory boundary rather than
escaping it, so the checkpoint doesn't land as a single file -- it's:
    epoch={X}-val/              (directory)
        mean_auroc={Y}.ckpt     (file)
(see resnetv2_ct.yaml's checkpoint_path for a real example of this same
quirk). This script parses that structure directly instead of trusting a
chart reading.

Note: the filename's val_auroc is rounded to 3 decimals (the ":.3f" in the
template) -- ModelCheckpoint's actual epoch-selection decision internally
uses full float precision, so which epoch got picked is unaffected, only
the value shown here is 3-decimal precision.
"""

import glob
import os

CHECKPOINT_DIR = "/data/processed/INSPECT/checkpoints"

# (description used in exp.name=sweep_<description>, freeze_backbone_epochs,
# backbone_lr_mult, dropout_prob, weighted_sample) -- mirrors run_rsna_sweep.sh
SWEEP_ROWS = [
    ("current",                 2, 0.1,  0.3, True),
    ("unfrozen_bb_but_dropout", 0, 0.1,  0.3, True),
    ("longer_freeze",           5, 0.1,  0.3, True),
    ("no_lr_multi",             2, 1.0,  0.3, True),
    ("no_freeze_no_lr_mult",    0, 1.0,  0.3, True),
    ("aggressive_lr_lowering",  2, 0.03, 0.3, True),
    ("stronger_regularization", 2, 0.1,  0.5, True),
    ("no_weighting",            2, 0.1,  0.3, False),
    ("dropout_path",            2, 0.1,  0.5, True),
]


def latest_run_dir(description):
    pattern = os.path.join(CHECKPOINT_DIR, f"sweep_{description}_pe_present_on_image_*")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def find_best_checkpoint(run_dir):
    """Return (epoch, val_mean_auroc) parsed from the epoch={X}-val/
    mean_auroc={Y}.ckpt structure. save_top_k=1, so normally exactly one
    match; if more than one is somehow present, take the most recently
    written."""
    matches = glob.glob(os.path.join(run_dir, "epoch=*-val", "mean_auroc=*.ckpt"))
    if not matches:
        return None, None

    ckpt_path = sorted(matches, key=os.path.getmtime)[-1]
    epoch_dir = os.path.basename(os.path.dirname(ckpt_path))  # "epoch={X}-val"
    ckpt_file = os.path.basename(ckpt_path)                   # "mean_auroc={Y}.ckpt"

    try:
        epoch = int(epoch_dir.split("=")[1].split("-")[0])
        val_auroc = float(ckpt_file.split("=")[1].replace(".ckpt", ""))
    except (IndexError, ValueError):
        return None, None
    return epoch, val_auroc


def main():
    rows = []
    for description, freeze, lr_mult, dropout, weighted in SWEEP_ROWS:
        run_dir = latest_run_dir(description)
        if run_dir is None:
            rows.append((description, freeze, lr_mult, dropout, weighted, None, None,
                         f"NO RUN DIR matching sweep_{description}_pe_present_on_image_*"))
            continue
        epoch, val_auroc = find_best_checkpoint(run_dir)
        if val_auroc is None:
            rows.append((description, freeze, lr_mult, dropout, weighted, None, None,
                         f"no epoch=*-val/mean_auroc=*.ckpt found in {run_dir}"))
            continue
        rows.append((description, freeze, lr_mult, dropout, weighted, epoch, val_auroc, run_dir))

    # Best (highest checkpointed val_auroc) first; missing/failed rows last.
    rows.sort(key=lambda r: (r[6] is None, -(r[6] or 0)))

    header = (
        f"{'Run':<28}{'Freeze':>7}{'LRmult':>8}{'Dropout':>9}{'Weighted':>10}"
        f"{'Epoch':>7}{'BestValAUROC':>14}"
    )
    print(header)
    print("-" * len(header))
    for description, freeze, lr_mult, dropout, weighted, epoch, val_auroc, note in rows:
        epoch_str = str(epoch) if epoch is not None else "N/A"
        auroc_str = f"{val_auroc:.3f}" if val_auroc is not None else "N/A"
        print(
            f"{description:<28}{freeze:>7}{lr_mult:>8}{dropout:>9}{str(weighted):>10}"
            f"{epoch_str:>7}{auroc_str:>14}"
        )
        if val_auroc is None:
            print(f"    -> {note}")


if __name__ == "__main__":
    main()
