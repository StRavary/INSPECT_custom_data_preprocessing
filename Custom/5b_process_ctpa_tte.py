"""
process_ctpa_tte.py
===================
TTE-pretrained 3D backbone vectorization for INSPECT CTPA volumes.

Mirrors process_ctpa.py (ResNetV2-101x3, slice-by-slice) but uses a 3D backbone
pretrained with time-to-event survival objectives (ICLR 2025, som-shahlab/tte-pretraining).

Key differences vs. process_ctpa.py:
  - Backbone is 3D: processes the full volume in one forward pass (no GRU needed)
  - Global average pool over the 3D feature map → single fixed-size embedding
  - Weights downloaded from StanfordShahLab HuggingFace hub
  - Supports three backbones: SwinUNETR (default), DenseNet121, ResNet152

Output: {patient_id}_tte_vector.pt  — same convention as _ctpa_vector.pt
        Shape: (embedding_dim,) float32 on CPU

Input format: NIfTI (.nii.gz), same as process_ctpa.py
"""

import os
import glob
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    Resize,
)
from monai.networks.nets import SwinUNETR, DenseNet121

# =========================================================================
# Backbone configs
# Each entry: (hf_repo, hf_filename, embedding_dim, build_fn)
# =========================================================================

BACKBONE_CONFIGS = {
    "swinunetr": {
        "repo":      "StanfordShahLab/tte-pretraining",
        "filename":  "swinunetr_tte.pt",
        "embed_dim": 768,   # SwinUNETR hidden_size default
    },
    "densenet121": {
        "repo":      "StanfordShahLab/tte-pretraining",
        "filename":  "densenet121_tte.pt",
        "embed_dim": 1024,  # DenseNet121 final feature map channels
    },
    "resnet152": {
        "repo":      "StanfordShahLab/tte-pretraining",
        "filename":  "resnet152_tte.pt",
        "embed_dim": 2048,  # ResNet152 layer4 channels
    },
}

VOLUME_SHAPE = (96, 96, 96)   # Spatial resize — matches TTE pretraining input size.
                               # SwinUNETR was pretrained at 96^3; increase if VRAM allows.


# =========================================================================
# 3D Encoder wrappers
# =========================================================================

class SwinUNETREncoder(nn.Module):
    """
    SwinUNETR backbone with the decoder head removed.
    Outputs global-average-pooled features from the encoder bottleneck.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.backbone = SwinUNETR(
            img_size=VOLUME_SHAPE,
            in_channels=1,
            out_channels=14,    # original pretraining head size; will be discarded
            feature_size=48,    # SwinUNETR-S; use 96 for SwinUNETR-B if weights match
            use_checkpoint=False,
        )
        self._load(checkpoint_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        # Strip common prefixes from tte-pretraining checkpoints
        clean = {}
        for k, v in state.items():
            for prefix in ("model.", "module.", "backbone."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            clean[k] = v
        missing, unexpected = self.backbone.load_state_dict(clean, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (e.g. {missing[:2]})")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})")

    def forward(self, x):
        # x: (1, 1, D, H, W)
        # SwinUNETR encoder returns a list of hidden states; use the last one
        hidden_states = self.backbone.swinViT(x, normalize=True)
        # hidden_states[-1]: (1, C, d, h, w) — bottleneck
        feat = hidden_states[-1]                         # (1, C, d, h, w)
        return feat.mean(dim=[2, 3, 4])                  # (1, C)


class DenseNet121Encoder(nn.Module):
    """
    MONAI DenseNet121 with the classifier replaced by global average pooling.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,   # placeholder; classifier will be stripped
        )
        self._load(checkpoint_path)
        # Replace classifier with identity so forward() returns features
        self.backbone.class_layers = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        clean = {}
        for k, v in state.items():
            for prefix in ("model.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            clean[k] = v
        missing, unexpected = self.backbone.load_state_dict(clean, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (e.g. {missing[:2]})")

    def forward(self, x):
        return self.backbone(x)   # (1, embed_dim) after replaced head


ENCODER_CLASSES = {
    "swinunetr":   SwinUNETREncoder,
    "densenet121": DenseNet121Encoder,
    # ResNet152 requires torchvision / medicalnet weights — add analogously
}


# =========================================================================
# Dataset — same NIfTI loading pipeline as process_ctpa.py
# =========================================================================

class NIfTI3DDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),                                      # (1, D, H, W)
            ScaleIntensityRange(a_min=-1000, a_max=600,
                                b_min=0.0, b_max=1.0, clip=True),    # CT HU windowing
            Resize(spatial_size=VOLUME_SHAPE),                         # (1, 96, 96, 96)
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # Structure: .../CTPA/PE1677746.nii.gz → patient_id = PE1677746
        patient_id = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]
        try:
            volume = self.transforms(path)
            volume = volume.as_tensor() if hasattr(volume, "as_tensor") else torch.as_tensor(volume)
        except Exception as e:
            print(f"  LOAD ERROR {patient_id}: {e}")
            return None, patient_id
        return volume, patient_id   # volume: (1, 96, 96, 96)


# =========================================================================
# Vectorization loop
# =========================================================================

def run_tte_vectorization(data_directory, output_destination, model, device,
                           suffix="_tte_vector"):
    file_paths = glob.glob(os.path.join(data_directory, "**/*.nii.gz"), recursive=True)
    file_paths = [p for p in file_paths if not p.startswith(output_destination)]
    print(f"Found {len(file_paths)} NIfTI volumes.")

    os.makedirs(output_destination, exist_ok=True)
    file_paths = [
        p for p in file_paths
        if not os.path.exists(os.path.join(
            output_destination,
            os.path.splitext(os.path.splitext(os.path.basename(p))[0])[0] + f"{suffix}.pt"
        ))
    ]
    print(f"{len(file_paths)} remaining after skipping already-processed.")

    dataset = NIfTI3DDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True,
                            collate_fn=lambda x: x)

    failed = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            volume, patient_id = batch[0]
            pid = patient_id

            if volume is None:
                print(f"  [{i+1}/{len(dataloader)}] SKIPPING {pid} (unreadable)")
                failed.append((pid, "unreadable"))
                continue

            try:
                x = volume.unsqueeze(0).to(device)     # (1, 1, D, H, W)
                with torch.amp.autocast(device.type):
                    embedding = model(x)                # (1, embed_dim)

                out_path = os.path.join(output_destination, f"{pid}{suffix}.pt")
                torch.save(embedding.cpu().squeeze(0), out_path)
                print(f"  [{i+1}/{len(dataloader)}] {pid}  shape={tuple(embedding.shape)}")

            except Exception as e:
                print(f"  [{i+1}/{len(dataloader)}] ERROR {pid}: {e}")
                failed.append((pid, str(e)))

    if failed:
        fail_log = os.path.join(output_destination, "_failed_tte.txt")
        with open(fail_log, "w") as f:
            for pid, err in failed:
                f.write(f"{pid}\t{err}\n")
        print(f"\n{len(failed)} failures → {fail_log}")

    print(f"\nDone. {len(dataloader) - len(failed)}/{len(dataloader)} vectorized.")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    BACKBONE = "swinunetr"   # swap to "densenet121" or "resnet152"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Backbone: {BACKBONE}")

    cfg = BACKBONE_CONFIGS[BACKBONE]
    print(f"Downloading TTE weights: {cfg['repo']}/{cfg['filename']} ...")
    weights_file = hf_hub_download(repo_id=cfg["repo"], filename=cfg["filename"])

    EncoderClass = ENCODER_CLASSES[BACKBONE]
    model = EncoderClass(checkpoint_path=weights_file).to(device)
    model.eval()

    RAW_DATA_DIR   = "/mnt/disks/gcs-bucket-mount"
    VECTOR_OUT_DIR = f"/home/steven_rav/tte_vectors/{BACKBONE}"

    if os.path.exists(RAW_DATA_DIR):
        run_tte_vectorization(RAW_DATA_DIR, VECTOR_OUT_DIR, model, device,
                               suffix=f"_{BACKBONE}_tte_vector")
    else:
        print(f"ERROR: data directory not found: {RAW_DATA_DIR}")
