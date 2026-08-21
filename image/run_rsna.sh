# Fine-tune ResNetV2-101x3 (BiT, 6144-d) on RSPECT (RSNA PE) dataset.
# Reverted from resnext101_32x8d (2048-d) back to resnetv2 after finding
# dataset.transform.channels=window was repeating a single PE window into
# all 3 channels instead of using the 3 distinct windows (lung/PE/
# mediastinal) run_featurize.py expects — now fixed in rsna.yaml/stanford.yaml.
#
# Starting weights: resnetv2_ct_pretrain.yaml -> the official
# StanfordShahLab/resnetv2_ct checkpoint from HuggingFace (downloaded to
# /data/processed/INSPECT/checkpoints/resnetv2_ct.ckpt), NOT the local
# resnetv2_ct.yaml checkpoint that used to live here (dated 2026-07-24,
# predated the windowing fix above, almost certainly fine-tuned under the
# old repeated-window scheme — reusing a repeat-window-trained checkpoint
# with real 3-window inputs is a documented failure mode in this repo, see
# resnext_101_imagenet.yaml / vision_backbones.py docstrings: out-of-
# distribution features, training collapses to loss=0). resnetv2_ct.yaml
# now points at this same official checkpoint too.
#
# After this run, copy the best val AUROC checkpoint to resnetv2_ct.yaml's
# checkpoint_path, then run_featurize.py will pick it up automatically —
# extract.yaml already defaults to model=resnetv2_ct, dataset=stanford
# (channels: window already fixed there).
#
# lr=0.0005 (below, inherited unchanged from the resnext101_32x8d runs) is
# aggressive for a full end-to-end fine-tune of resnetv2_101x3_bitm_in21k —
# a BiT (Big Transfer) model. Google's own BiT paper is explicit that these
# need much more conservative fine-tuning LRs than standard ResNets; too
# high an LR on a full fine-tune tends to rapidly degrade an already-strong
# pretrained representation. The first fine-tune here (2026-08-10) peaked at
# epoch 0 and declined every epoch after — consistent with exactly that.
# Dropped 0.0005 -> 0.00003 (~15x) for this run; combined with
# resnetv2_ct_pretrain.yaml's new dropout_prob=0.3.
#
# Second fine-tune (2026-08-11, lr=0.00003 + dropout_prob=0.3) showed the
# same epoch-0-best-then-declining pattern. dropout_prob only regularizes
# the ~6K-param classifier head; it does nothing for the ~200M-param
# GroupNorm/weight-standardized-conv backbone, which is what's actually
# free to drift/overfit against a small, heavily-oversampled positive pool
# (2,368 positive-exam studies, ~3x WeightedRandomSampler oversampling —
# see Custom/INSPECT_Baseline_Reconstruction.md §22.3). This run instead:
#   1. Freezes the backbone for the first freeze_backbone_epochs epochs
#      (linear-probe warmup: only the classifier head trains).
#   2. Unfreezes with the backbone at lr * backbone_lr_mult (10x smaller
#      than the head's lr) rather than the same LR for every param.
# Watch train/mean_auroc vs val/mean_auroc in wandb after this run: if
# train keeps climbing while val still declines post-unfreeze, that
# confirms backbone overfitting rather than an LR/precision/scheduler bug.
#
# Third fine-tune (2026-08-19 to 2026-08-21): run_rsna_sweep.sh screened 9
# combinations of freeze_backbone_epochs / backbone_lr_mult / dropout_prob /
# weighted_sample at a reduced max_epochs=10 (see build_sweep_table.py for
# the ranked results). Findings:
#   - backbone_lr_mult is the load-bearing lever, not freezing -- the two
#     clearly worst runs (no_lr_multi, no_freeze_no_lr_mult) both had
#     backbone_lr_mult=1.0 regardless of freeze setting.
#   - dropout_prob (0.3 vs 0.5) and weighted_sample (true vs false) made no
#     measurable difference -- 6 of 9 configs converged to within 0.001
#     val/mean_auroc of each other despite varying these independently, the
#     signature of an intrinsic ceiling rather than more signal to chase.
#   - Winner: aggressive_lr_lowering (freeze=2, backbone_lr_mult=0.03,
#     dropout=0.3, weighted=true), val/mean_auroc=0.937 -- tied with
#     unfrozen_bb_but_dropout (freeze=0), but that run's identical peak was
#     at epoch 0 (declined every epoch after, the same failure signature
#     freezing/lr_mult were built to fix), while aggressive_lr_lowering's
#     peak was at epoch 3 -- genuine multi-epoch improvement, not a lucky
#     first pass. Picked on that basis over the raw tie.
# This run uses that winning config at full length (max_epochs=30, not the
# sweep's screening length) to produce the actual checkpoint for
# resnetv2_ct.yaml -- copy its best val AUROC checkpoint there when done.
#
# Other backbones tried previously (kept for reference):
#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=swinv2 dataset=rsna
#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=dinov2 dataset=rsna dataset.transform.final_size=224
#CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnext_101_ct_pretrain dataset=rsna dataset.transform.final_size=224 dataset.batch_size=256 trainer.accumulate_grad_batches=1 lr=0.0005
#CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnetv2 dataset=rsna dataset.transform.final_size=224 dataset.batch_size=64 trainer.accumulate_grad_batches=4 lr=0.0005
#CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnetv2_ct_pretrain dataset=rsna dataset.transform.final_size=224 dataset.batch_size=64 trainer.accumulate_grad_batches=4 lr=0.0005
#CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnetv2_ct_pretrain dataset=rsna dataset.transform.final_size=224 dataset.batch_size=64 trainer.accumulate_grad_batches=4 lr=0.00003 trainer.freeze_backbone_epochs=2 backbone_lr_mult=0.1

CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnetv2_ct_pretrain dataset=rsna \
    exp.name=rsna_final_aggressive_lr_lowering \
    dataset.transform.final_size=224 \
    dataset.batch_size=64 \
    dataset.weighted_sample=true \
    trainer.accumulate_grad_batches=4 \
    trainer.max_epochs=30 \
    lr=0.0005 \
    trainer.freeze_backbone_epochs=2 \
    backbone_lr_mult=0.03 \
    model.dropout_prob=0.3
