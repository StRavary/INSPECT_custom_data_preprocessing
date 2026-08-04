# Featurize with the INSPECT paper backbone (ResNeXt101-32x8d, 2048-d)
# Checkpoint: RSNA-fine-tuned ResNeXt101 (resnext101_rsna.ckpt, val AUROC=0.906 on RSNA PE detection)
python run_featurize.py model=resnext_101_ct \
	dataset=stanford \
	dataset.transform.final_size=224 \
	dataset.batch_size=256 \
	dataset.transform.channels=window


python convert_to_hdf5.py