# Fine-tune ResNeXt101-32x8d on RSPECT (RSNA PE) dataset.
# Matches Table 14: architecture=resnext101_32x8d, optimizer=AdamW, lr=0.0005,
# loss=BCEWithLogitsLoss, pretrain=BigTransfer (ImageNet large-scale).
# Starting weights: resnext_101_ct.yaml checkpoint (CT-pretrained from HuggingFace).
# After this run, copy the best val AUROC checkpoint to resnext_101_ct.yaml checkpoint_path.
#
# Other backbones tried previously (kept for reference):
#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=swinv2 dataset=rsna
#CUDA_VISIBLE_DEVICES=1 python run_classify.py model=dinov2 dataset=rsna dataset.transform.final_size=224
#CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnetv2 dataset=rsna dataset.transform.final_size=224 dataset.batch_size=64 trainer.accumulate_grad_batches=4

# Use resnext_101_imagenet (ImageNet pretrained, no checkpoint) as the starting
# point — NOT the repeat-trained resnext101_rsna.ckpt. Starting from the
# repeat-trained checkpoint with window inputs collapses gradients to zero.
CUDA_VISIBLE_DEVICES=0 python run_classify.py model=resnext_101_imagenet dataset=rsna \
    dataset.transform.final_size=224 \
    dataset.batch_size=256 \
    trainer.accumulate_grad_batches=1 \
    lr=0.0005
