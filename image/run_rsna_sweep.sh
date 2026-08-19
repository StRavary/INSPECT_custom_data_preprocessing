#!/bin/bash
# Hyperparameter sweep over run_classify.py's backbone-freezing +
# discriminative-LR + classifier-dropout + weighted-sample options, for
# resnetv2_ct RSPECT fine-tuning. Calls the existing run_classify.py
# directly (same as run_rsna.sh) -- no forked/experimental script needed,
# since backbone_lr_mult/trainer.freeze_backbone_epochs are already
# opt-in and don't affect any run that doesn't set them (i.e. every
# production run_classify_*.sh target is unaffected by this file).
#
# LR note: unlike run_rsna.sh (lr=0.00003, from back when LR was applied
# uniformly to the whole network), every row here uses lr=0.0005 as the
# CLASSIFIER HEAD's LR -- with discriminative LRs, cfg.lr is the head's
# rate; backbone_lr_mult scales the backbone down from there
# (backbone_lr = lr * backbone_lr_mult). 5e-4 is a standard Adam-family
# default and the head is freshly initialized, so there's no
# catastrophic-forgetting risk the way there was fine-tuning the whole
# backbone at that LR.
#
# Each run gets its own exp.name so TensorBoard can tell them apart:
#   tensorboard --logdir /data/processed/INSPECT/checkpoints
# will auto-discover all 9 (plus anything else in that directory) as
# separate overlaid runs. Watch in particular: train/mean_auroc vs
# val/mean_auroc (and mean_auprc), train/backbone_frozen (step function --
# confirms the freeze schedule actually took effect and shows exactly when
# each run unfroze), and the per-param-group LR curves under whatever tag
# LearningRateMonitor assigns (confirms backbone/head are actually training
# at the intended rates).
#
# MAX_EPOCHS: 9 sequential full 30-epoch runs is a lot of wall-clock time.
# Defaults to a shorter screening length here -- bump to 30 (or unset, to
# fall back to classify.yaml's default) for a full run once you've picked
# a winner from the short runs.
MAX_EPOCHS=10

BASE_ARGS="model=resnetv2_ct_pretrain dataset=rsna \
    dataset.transform.final_size=224 \
    dataset.batch_size=64 \
    trainer.accumulate_grad_batches=4 \
    trainer.max_epochs=${MAX_EPOCHS} \
    lr=0.0005"

run() {
    local name="$1"
    local freeze="$2"
    local lr_mult="$3"
    local dropout="$4"
    local weighted="$5"

    echo "***************" "$name" "********************************************"
    CUDA_VISIBLE_DEVICES=0 python run_classify.py $BASE_ARGS \
        exp.name="sweep_${name}" \
        trainer.freeze_backbone_epochs=${freeze} \
        backbone_lr_mult=${lr_mult} \
        model.dropout_prob=${dropout} \
        dataset.weighted_sample=${weighted}
}

#              name                         freeze  lr_mult  dropout  weighted
run            current                      2       0.1      0.3      true
run            unfrozen_bb_but_dropout      0       0.1      0.3      true
run            longer_freeze                5       0.1      0.3      true
run            no_lr_multi                  2       1.0      0.3      true
run            no_freeze_no_lr_mult         0       1.0      0.3      true
run            aggressive_lr_lowering       2       0.03     0.3      true
run            stronger_regularization      2       0.1      0.5      true
run            no_weighting                 2       0.1      0.3      false
# NOTE: same hyperparameters as stronger_regularization (0.5 is the
# literature default for FC-layer dropout either way -- see prior
# discussion) -- distinct exp.name only, so this is currently a
# replicate/variance check rather than a distinct config. Change the
# dropout value below if you meant something else by "Dropout_path".
run            dropout_path                 2       0.1      0.5      true
