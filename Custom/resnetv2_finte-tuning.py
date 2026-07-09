import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import monai.transforms as transforms
from monai.data import Dataset
import timm 
from huggingface_hub import hf_hub_download

def load_resnet_encoder(device="cuda", num_classes=1):
    print(f"Downloading StanfordShahLab ResNetV2 checkpoint...")
    checkpoint_path = hf_hub_download(repo_id="StanfordShahLab/resnetv2_ct", filename="model.ckpt")
    
    print("Initializing base architecture (resnetv2_101x3_bit.goog_in21k_ft_in1k)...")
    # Use goog_in21k_ft_in1k to get GroupNorm + StdConv2d instead of BatchNorm
    model = timm.create_model('resnetv2_101x3_bit.goog_in21k_ft_in1k', pretrained=False, num_classes=num_classes)
    
    # Load the Shah Lab weights (weights_only=False required for older Lightning checkpoints)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Remove all nested 'model.' prefixes (e.g. from PyTorch Lightning / radfusion3 wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        while k.startswith("model."):
            k = k.replace("model.", "", 1)
        new_state_dict[k] = v
    state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")
    return model.to(device)

def get_train_transforms():
    # Define your MONAI transforms for the CTPAs
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        # Standard PE windowing (e.g., W:400, L:40)
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-160, a_max=240, b_min=0.0, b_max=1.0, clip=True),
        # ResNetV2 usually expects 3 channels (RGB). You can duplicate the grayscale CT channel 3 times.
        transforms.RepeatChanneld(keys=["image"], repeats=3),
        transforms.Resized(keys=["image"], spatial_size=(224, 224)), # Resize to model's expected input
        transforms.ToTensorD(keys=["image", "label"])
    ])

def finetuning(model, dataloader, epochs=5, device="cuda"):
    criterion = nn.BCEWithLogitsLoss() # Assuming binary PE classification
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch["image"], batch["label"]
            
            # Issue #6 Convert MONAI MetaTensors to standard PyTorch tensors to prevent serialization bugs
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            if hasattr(labels, "as_tensor"):
                labels = labels.as_tensor()
                
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader)}")
        
    return model

def save_for_rnn_module(model, save_path="resnetv2_finetuned_encoder.pth"):
    # Save only the weights to be injected into the CNN-RNN script later
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned encoder saved to {save_path}")

if __name__ == "__main__":
    # Example usage:
    # 1. Load the model
    encoder = load_resnet_encoder(device="cuda", num_classes=1)
    
    # 2. Setup your dataloader (requires your PyArrow ingestion logic)
    dataset = Dataset(data="", transform=get_train_transforms()) # to be defined once RSPECT is downloaded
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # 3. Fine-tune
    encoder = finetuning(encoder, dataloader, epochs=5, device="cuda")
    
    # 4. Save for the RNN script
    save_for_rnn_module(encoder)
    pass