import torch
import cv2
import pydicom
import numpy as np
import pandas as pd
from PIL import Image as _PILImage
import h5py
import nibabel as nib

from ..constants import *
from torch.utils.data import Dataset
from pathlib import Path
import os
from ..utils import read_tar_dicom
import io
import pickle


class DatasetBase(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.cfg = cfg
        self.transform = transform
        self.split = split
        self.hdf5_dataset = None

        # NIfTI last-file cache: avoids reloading the same file for each slice
        self._nifti_cache_path = None
        self._nifti_cache_data = None

        # Try Stanford-internal pickle first (original source)
        pickle_path = "/share/pi/nigam/projects/zphuo/data/omop_extract_PHI/som-nero-phi-nigam-starr.frazier/dict_slice_thickness.pkl"
        if os.path.exists(pickle_path):
            self.dict_slice_thickness = pickle.load(open(pickle_path, "rb"))
        else:
            # Fall back to SliceThickness column in series metadata TSV
            # Keys are image_id without .nii.gz extension
            self.dict_slice_thickness = {}
            csv_path = getattr(getattr(cfg, "dataset", cfg), "csv_path", None)
            if csv_path and os.path.exists(csv_path):
                try:
                    sep = "\t" if str(csv_path).endswith(".tsv") else ","
                    meta = pd.read_csv(csv_path, sep=sep, usecols=["image_id", "SliceThickness"])
                    meta["image_id"] = meta["image_id"].astype(str).str.replace(".nii.gz", "", regex=False)
                    self.dict_slice_thickness = dict(zip(meta["image_id"], meta["SliceThickness"].fillna(1.0)))
                except Exception as e:
                    print(f"[WARN] Could not load SliceThickness from metadata: {e}")

    def is_rsna(self):
        csv_path = str(getattr(self.cfg.dataset, "csv_path", "")).lower()
        ds_type = str(getattr(self.cfg.dataset, "type", "")).lower()
        return "rsna" in csv_path or "rspect" in csv_path or "rsna" in ds_type

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_hdf5(self, key, hdf5_path, slice_idx=None):
        if self.hdf5_dataset is None:
            self.hdf5_dataset = h5py.File(hdf5_path, "r")

        if key not in self.hdf5_dataset:
            return None

        if slice_idx is None:
            arr = self.hdf5_dataset[key][:]
        else:
            arr = self.hdf5_dataset[key][slice_idx]

        # df_dicom_headers["patient_datetime"] = df_dicom_headers.apply(
        #     lambda x: f"{x.PatientID}_{x.StudyTime}", axis=1
        # )

        # Add positional encoding column
        # When slice thickness is available (Stanford internal), use thickness * slice_idx
        # (encodes absolute z-position in mm). Otherwise use normalized slice index (0→1)
        # as a substitute positional signal. This matches position_encoding: true in classify.yaml.
        n_slices = arr.shape[0]
        if self.dict_slice_thickness:
            pos = np.array([
                self.dict_slice_thickness.get(key, 1.0) * i for i in range(n_slices)
            ], dtype=np.float32)
            # Normalize to [0, 1]: raw values are thickness_mm × slice_idx
            # (e.g. 1.5mm × 249 = 373) which is orders of magnitude larger
            # than CNN features and saturates LSTM gates → NaN loss.
            pos = pos / max(pos[-1], 1.0)
        else:
            pos = np.arange(n_slices, dtype=np.float32) / max(n_slices - 1, 1)

        arr = np.concatenate([arr, pos[:, None]], axis=1)

        return arr

    def read_dicom(self, file_path: str, resize_size=None, channels=None):
        """Legacy DICOM reader, kept for compatibility"""
        if resize_size is None:
            resize_size = self.cfg.dataset.transform.resize_size
        if channels is None:
            channels = self.cfg.dataset.transform.channels

        # read dicom
        file_path = str(file_path)
        if self.is_rsna():
            dcm = pydicom.dcmread(file_path)
        else:
            patient_id = file_path.split("/")[-1].split("_")[0]
            tar_content = read_tar_dicom(
                os.path.join(self.cfg.dataset.dicom_dir, patient_id + ".tar")
            )
            dcm = pydicom.dcmread(io.BytesIO(tar_content[file_path]))

        try:
            pixel_array = dcm.pixel_array
        except:
            print(file_path)
            if channels == "repeat":
                pixel_array = np.zeros((resize_size, resize_size))
            else:
                pixel_array = np.zeros((3, resize_size, resize_size))

        # rescale
        try:
            intercept = dcm.RescaleIntercept
            slope = dcm.RescaleSlope
        except:
            intercept = 0
            slope = 1

        pixel_array = pixel_array * slope + intercept

        # resize
        if resize_size != pixel_array.shape[-1]:
            pixel_array = cv2.resize(
                pixel_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )

        return pixel_array

    def read_image(self, file_path: str, resize_size=None, channels=None, slice_idx: int = 0):
        """Read medical image in DICOM or NIfTI format"""
        if resize_size is None:
            resize_size = self.cfg.dataset.transform.resize_size
        if channels is None:
            channels = self.cfg.dataset.transform.channels

        # Read image based on format
        if file_path.endswith('.nii.gz'):
            # Use last-file cache: DataLoader sorts by path so consecutive
            # __getitem__ calls for the same scan hit the cache.
            try:
                if self._nifti_cache_path != file_path:
                    nifti_img = nib.load(file_path)
                    self._nifti_cache_data = nifti_img.get_fdata()
                    self._nifti_cache_path = file_path
                volume = self._nifti_cache_data
                if len(volume.shape) == 2:
                    pixel_array = volume
                elif len(volume.shape) == 3:
                    # shape: (H, W, num_slices)
                    actual_idx = min(slice_idx, volume.shape[2] - 1)
                    pixel_array = volume[:, :, actual_idx]
                else:
                    # 4D: (H, W, num_slices, T)
                    actual_idx = min(slice_idx, volume.shape[2] - 1)
                    pixel_array = volume[:, :, actual_idx, 0]
            except Exception as e:
                print(f"[WARN] Corrupt or unreadable NIfTI, skipping: {file_path} ({e})")
                self._nifti_cache_path = None
                self._nifti_cache_data = None
                pixel_array = np.zeros((resize_size, resize_size))
        else:
            # Read DICOM
            if self.is_rsna():
                dcm = pydicom.dcmread(file_path)
            else:
                patient_id = file_path.split("/")[-1].split("_")[0]
                tar_content = read_tar_dicom(
                    os.path.join(self.cfg.dataset.dicom_dir, patient_id + ".tar")
                )
                dcm = pydicom.dcmread(io.BytesIO(tar_content[file_path]))

            try:
                pixel_array = dcm.pixel_array
                try:
                    intercept = dcm.RescaleIntercept
                    slope = dcm.RescaleSlope
                except:
                    intercept = 0
                    slope = 1
                pixel_array = pixel_array * slope + intercept
            except:
                print(f"Error reading {file_path}")
                if channels == "repeat":
                    pixel_array = np.zeros((resize_size, resize_size))
                else:
                    pixel_array = np.zeros((3, resize_size, resize_size))

        # Resize image
        if resize_size != pixel_array.shape[-1]:
            pixel_array = np.array(
                _PILImage.fromarray(pixel_array.astype(np.float32)).resize(
                    (resize_size, resize_size), _PILImage.LANCZOS
                )
            )

        return pixel_array

    def windowing(self, pixel_array: np.array, window_center: int, window_width: int):
        lower = window_center - window_width // 2
        upper = window_center + window_width // 2
        pixel_array = np.clip(pixel_array.copy(), lower, upper)
        pixel_array = (pixel_array - lower) / (upper - lower)

        return pixel_array

    def process_numpy(self, numpy_path, idx):
        slice_array = np.load(numpy_path)[idx]

        resize_size = self.cfg.dataset.transform.resize_size
        channels = self.cfg.dataset.transform.channels

        if resize_size != slice_array.shape[-1]:
            slice_array = cv2.resize(
                slice_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )

        # window
        if self.cfg.dataset.transform.channels == "repeat":
            ct_slice = self.windowing(
                slice_array, 400, 1000
            )  # use PE window by default
            # create 3 channels after converting to Tensor
            # using torch.repeat won't take up 3x memory
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400),  # MEDIASTINAL window
            ]
            ct_slice = np.stack(ct_slice)

        return ct_slice

    def process_slice(
        self,
        slice_info: pd.Series = None,
        dicom_dir: Path = None,
        slice_path: str = None,
    ):
        """process slice with windowing, resize and tranforms"""

        if slice_path is None:
            slice_path = dicom_dir / slice_info[INSTANCE_PATH_COL]
        slice_array = self.read_dicom(slice_path)

        # window
        if self.cfg.dataset.transform.channels == "repeat":
            ct_slice = self.windowing(
                slice_array, 400, 1000
            )  # use PE window by default
            # create 3 channels after converting to Tensor
            # using torch.repeat won't take up 3x memory
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400),  # MEDIASTINAL window
            ]
            ct_slice = np.stack(ct_slice)

        return ct_slice

    def fix_slice_number(self, df: pd.DataFrame):
        num_slices = min(self.cfg.dataset.num_slices, df.shape[0])
        if self.cfg.dataset.sample_strategy == "random":
            slice_idx = np.random.choice(
                np.arange(df.shape[0]), replace=False, size=num_slices
            )
            slice_idx = list(np.sort(slice_idx))
            df = df.iloc[slice_idx, :]
        elif self.cfg.dataset.sample_strategy == "fix":
            df = df.iloc[:num_slices, :]
        else:
            raise Exception("Sampling strategy either 'random' or 'fix'")
        return df

    def fix_series_slice_number(self, series):
        num_slices = min(self.cfg.dataset.num_slices, series.shape[0])
        if num_slices == self.cfg.dataset.num_slices:
            if self.cfg.dataset.sample_strategy == "random":
                slice_idx = np.random.choice(
                    np.arange(series.shape[0]), replace=False, size=num_slices
                )
                slice_idx = list(np.sort(slice_idx))
                features = series[slice_idx, :]
            elif self.cfg.dataset.sample_strategy == "fix":
                pad = int((series.shape[0] - num_slices) / 2)  # select middle slices
                start = pad
                end = pad + num_slices
                features = series[start:end, :]
            else:
                raise Exception("Sampling strategy either 'random' or 'fix'")
            mask = np.ones(num_slices)
        else:
            mask = np.zeros(self.cfg.dataset.num_slices)
            mask[:num_slices] = 1
            shape = [self.cfg.dataset.num_slices] + list(series.shape[1:])
            features = np.zeros(shape)

            features[:num_slices] = series

        return features, mask

    def fill_series_to_num_slicess(self, series, num_slices):
        x = torch.zeros(()).new_full((num_slices, *series.shape[1:]), 0.0)
        x[: series.shape[0]] = series
        return x
