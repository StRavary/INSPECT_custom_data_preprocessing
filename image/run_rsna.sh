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
# Other backbones tried previously (kept for reference):
#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=swinv2 dataset=rsna
#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=dinov2 dataset=rsna dataset.transform.final_size=224
#CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnext_101_ct_pretrain dataset=rsna dataset.transform.final_size=224 dataset.batch_size=256 trainer.accumulate_grad_batches=1 lr=0.0005
#CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnetv2 dataset=rsna dataset.transform.final_size=224 dataset.batch_size=64 trainer.accumulate_grad_batches=4 lr=0.0005

CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnetv2_ct_pretrain dataset=rsna \
    dataset.transform.final_size=224 \
    dataset.batch_size=64 \
    trainer.accumulate_grad_batches=4 \
    lr=0.0005
