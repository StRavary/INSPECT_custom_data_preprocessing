# Featurize with the INSPECT paper backbone (ResNeXt101-32x8d, 2048-d)
# Matches Table 14: resnext101_32x8d, BigTransfer pretrain, fine-tuned on RSPECT (RSNA PE).
# Checkpoint path set in radfusion3/configs/model/resnext_101_ct.yaml after RSNA fine-tuning.
python run_featurize.py model=resnext_101_ct \
	dataset=stanford \
	dataset.transform.final_size=224 \
	dataset.batch_size=256 \
	dataset.transform.channels=window

python convert_to_hdf5.py