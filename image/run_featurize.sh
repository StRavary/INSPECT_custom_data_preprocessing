# Featurize with the INSPECT paper backbone (ResNeXt101-32x8d, 2048-d)
# Checkpoint: StanfordShahLab/ResNeXt101_ct from HuggingFace
python run_featurize.py model=resnext_101_ct \
	dataset=stanford \
	dataset.transform.final_size=224 \
	dataset.batch_size=256 \
	dataset.transform.channels=window


python convert_to_hdf5.py