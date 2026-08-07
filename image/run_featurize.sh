# Featurize with the INSPECT paper backbone (ResNeXt101-32x8d, 2048-d)
# Matches Table 14: resnext101_32x8d, BigTransfer pretrain, fine-tuned on RSPECT (RSNA PE).
# Checkpoint path set in radfusion3/configs/model/resnext_101_ct.yaml after RSNA fine-tuning.
#
# RUN_NAME tags this featurization run. Per-scan .npy files go to:
#   /data/processed/INSPECT/CNN_embeddings/${RUN_NAME}_<timestamp>/
# The HDF5 is written to:
#   /data/processed/INSPECT/CNN_embeddings/${RUN_NAME}_uncompressed.hdf5
# Change RUN_NAME to avoid overwriting previous embeddings.
RUN_NAME="resnext_window"

python run_featurize.py model=resnext_101_ct \
	dataset=stanford \
	dataset.transform.final_size=224 \
	dataset.batch_size=256 \
	dataset.transform.channels=window \
	exp.name=${RUN_NAME}

python convert_to_hdf5.py --run_name ${RUN_NAME}