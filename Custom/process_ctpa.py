import os
import glob
import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    Resize
)

# =========================================================================
# Stanford ResNetV2-101x3 Backbone Encoder Architecture
# =========================================================================
# Feature dimensions for resnetv2_101 with width_factor=3:
#   widths scale as [768, 1536, 3072, 6144] → num_features = 6144
#
# attention_max aggregation output dim = 2 * rnn_output_dim
#   with hidden_size=512, bidirectional=True → embedding dim = 2048
# =========================================================================

SLICE_ENCODER_FEATURES = 6144   # resnetv2_101 × width_factor=3
CHUNK_SIZE = 32                  # max slices per GPU forward pass (tune for VRAM)



class StanfordCTMultimodalEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        hidden_size=512,
        bidirectional=True,
        aggregation="spatial_mean",  # safe default: mean-pool pretrained ResNet features
        chunk_size=CHUNK_SIZE,
    ):
        super().__init__()
        print("Initializing ResNetV2-101x3 Backbone...")

        self.chunk_size = chunk_size

        # resnetv2_101x3_bit: BiT variant, GroupNorm + StdConv2d,
        # blocks [3,4,23,3], width_factor=3 → num_features=6144. Matches checkpoint exactly.
        self.slice_encoder = timm.create_model(
            'resnetv2_101x3_bit.goog_in21k_ft_in1k',
            pretrained=False,
            in_chans=3,
            num_classes=0
        )

        print(f"Loading Stanford Shah Lab CT weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # FIX: checkpoint keys are prefixed "model.model." — strip all leading "model." segments
        clean_state_dict = {}
        for k, v in state_dict.items():
            while k.startswith("model."):
                k = k[len("model."):]
            clean_state_dict[k] = v

        missing, unexpected = self.slice_encoder.load_state_dict(clean_state_dict, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

        self.slice_encoder.eval()
        for param in self.slice_encoder.parameters():
            param.requires_grad = False

        self.sequence_encoder = nn.GRU(
            input_size=SLICE_ENCODER_FEATURES,  # FIX: was 2048, must match width_factor=3
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.25
        )

        rnn_output_dim = hidden_size * 2 if bidirectional else hidden_size

        self.attention_weights = nn.Sequential(
            nn.Linear(rnn_output_dim, 1),
            nn.Softmax(dim=1)
        )

        self.aggregation = aggregation

    def _encode_slices(self, x):
        """Run slice encoder in chunks to avoid OOM on large CT volumes."""
        chunks = [
            self.slice_encoder(x[i : i + self.chunk_size])
            for i in range(0, x.shape[0], self.chunk_size)
        ]
        return torch.cat(chunks, dim=0)

    def forward(self, x_sequence):
        # x_sequence: (N_slices, 1, H, W)
        if x_sequence.shape[1] == 1:
            x_sequence = x_sequence.repeat(1, 3, 1, 1)  # → (N_slices, 3, H, W)

        # Chunked ResNet encoding over all slices
        spatial_vectors = self._encode_slices(x_sequence)   # (N_slices, 6144)

        # "spatial_mean" / "spatial_max": aggregate ResNet features directly.
        # Use these for vectorization — the GRU/attention are untrained and will
        # produce NaN over long sequences unless explicitly fine-tuned first.
        if self.aggregation == "spatial_mean":
            return spatial_vectors.mean(dim=0, keepdim=True)   # (1, 6144)
        elif self.aggregation == "spatial_max":
            return spatial_vectors.max(dim=0, keepdim=True)[0] # (1, 6144)

        # GRU-based aggregation — only use if GRU weights have been trained.
        spatial_vectors = spatial_vectors.unsqueeze(0)          # (1, N_slices, 6144)
        rnn_out, _ = self.sequence_encoder(spatial_vectors)     # (1, N_slices, rnn_dim)

        if self.aggregation == "mean":
            return torch.mean(rnn_out, dim=1)
        elif self.aggregation == "max":
            return torch.max(rnn_out, dim=1)[0]
        elif self.aggregation == "attention_max":
            attn_scores = self.attention_weights(rnn_out)         # (1, N_slices, 1)
            attn_out = torch.sum(rnn_out * attn_scores, dim=1)    # (1, rnn_dim)
            max_out = torch.max(rnn_out, dim=1)[0]                # (1, rnn_dim)
            return torch.cat([attn_out, max_out], dim=1)          # (1, 2*rnn_dim)
        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation}")


# =========================================================================
# High-Throughput 1-Channel Cloud Dataset Loader
# =========================================================================

class Cloud1ChannelDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensityRange(a_min=-1000, a_max=600, b_min=0.0, b_max=1.0, clip=True),
            Resize(spatial_size=(448, 256, 256))
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Structure: .../full/CTPA/PE1677746.nii.gz — patient ID is the filename stem
        patient_id = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]

        try:
            # transforms: (1,448,256,256) → squeeze → (448,256,256) → unsqueeze(1) → (448,1,256,256)
            # FIX: MONAI returns MetaTensor; convert to plain Tensor to avoid downstream serialization issues
            volume = self.transforms(path).squeeze(0).unsqueeze(1)
            volume = volume.as_tensor() if hasattr(volume, 'as_tensor') else torch.as_tensor(volume)
        except Exception as e:
            print(f"  LOAD ERROR {patient_id}: {e}")
            return None, patient_id
        return volume, patient_id


# =========================================================================
# Vectorization Processing Loop
# =========================================================================

def run_stanford_vectorization(data_directory, output_destination, model, device):
    file_paths = glob.glob(os.path.join(data_directory, "**/*.nii.gz"), recursive=True)

    # Exclude any .nii.gz that ended up inside the output dir
    file_paths = [p for p in file_paths if not p.startswith(output_destination)]

    print(f"Found {len(file_paths)} NIfTI volumes to process.")
    if len(file_paths) == 0:
        print("Warning: no files found via recursive glob. Checking root directory...")
        file_paths = [
            os.path.join(data_directory, f)
            for f in os.listdir(data_directory)
            if f.endswith('.nii.gz')
        ]

    # Pre-filter already-processed files so they are never loaded from disk
    os.makedirs(output_destination, exist_ok=True)
    file_paths = [
        p for p in file_paths
        if not os.path.exists(os.path.join(
            output_destination,
            os.path.splitext(os.path.splitext(os.path.basename(p))[0])[0] + '_ctpa_vector.pt'
        ))
    ]
    print(f"{len(file_paths)} studies remaining after skipping already-processed.")

    dataset = Cloud1ChannelDataset(file_paths)

    # FIX: num_workers=0 for gcsfuse — parallel workers cause hangs on FUSE mounts.
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True,
                            collate_fn=lambda x: x)  # allow None items through

    failed = []

    print(f"Processing {len(dataloader)} studies → {output_destination}")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            volumes, patient_ids = batch[0]
            pid = patient_ids

            if volumes is None:
                print(f"  [{i+1}/{len(dataloader)}] SKIPPING {pid} (unreadable file)")
                failed.append((pid, "unreadable"))
                continue

            try:
                output_file = os.path.join(output_destination, f"{pid}_ctpa_vector.pt")
                slices = volumes.to(device)  # (448, 1, 256, 256)

                # FIX: use device.type so autocast is a no-op on CPU fallback
                with torch.amp.autocast(device.type):
                    final_vector = model(slices)

                torch.save(final_vector.cpu().squeeze(0), output_file)
                print(f"  [{i+1}/{len(dataloader)}] Saved embedding for: {pid}")

            except Exception as e:
                print(f"  [{i+1}/{len(dataloader)}] ERROR on {pid}: {e}")
                failed.append((pid, str(e)))

    if failed:
        fail_log = os.path.join(output_destination, "_failed_cases.txt")
        with open(fail_log, "w") as f:
            for pid, err in failed:
                f.write(f"{pid}\t{err}\n")
        print(f"\n{len(failed)} failures logged to {fail_log}")

    print(f"\nDone. {len(dataloader) - len(failed)}/{len(dataloader)} studies vectorized.")


# =========================================================================
# Entry Point
# =========================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Downloading weights from StanfordShahLab Hub...")
    weights_file = hf_hub_download(
        repo_id="StanfordShahLab/resnetv2_ct",
        filename="resnetv2_ct.ckpt"
    )

    model = StanfordCTMultimodalEncoder(checkpoint_path=weights_file).to(device)
    model.eval()

    RAW_DATA_DIR   = "/mnt/disks/gcs-bucket-mount"
    VECTOR_OUT_DIR = "/home/steven_rav/ctpa_vectors"   # FIX: local writeable path; sync to GCS after

    if os.path.exists(RAW_DATA_DIR):
        run_stanford_vectorization(RAW_DATA_DIR, VECTOR_OUT_DIR, model, device)
    else:
        print(f"ERROR: data directory not found: {RAW_DATA_DIR}")
