import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# [FIX] Use raw strings (r"") to prevent Windows path errors
DATA_DIR = r"D:\Model\data\train"
VAL_DIR = r"D:\Model\data\val"
OUTPUT_DIR = r"D:\Model\Team Zero\Output"

# Training settings
EPOCHS = 25
BATCH_SIZE = 16
LR = 1e-3
IMG_H, IMG_W = 518, 952  # Dimensions (Multiple of 14 for DINOv2)

# Classes
VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}
CLASS_NAMES = ['BG','Trees','LushBush','DryGrass','DryBush','Clutter','Flowers','Logs','Rocks','Land','Sky']
N_CLASSES = 11

# ============================================================================
# DATASET & TRANSFORMS
# ============================================================================

def get_transforms(h, w, train=True):
    if train:
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            # [FIX] Removed GaussNoise to prevent version conflicts (var_limit error)
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

class SegDataset(Dataset):
    def __init__(self, data_dir, transform):
        # [FIX] Robust path joining
        self.img_dir = os.path.join(data_dir, 'rgb')
        self.mask_dir = os.path.join(data_dir, 'egmented')
        self.transform = transform

        # Debugging: Check if folders exist
        if not os.path.exists(self.img_dir):
            print(f"ERROR: Image folder not found at: {self.img_dir}")
            print(f"       Please check if the folder is named 'rgb' or 'images'")
            raise FileNotFoundError(f"Missing folder: {self.img_dir}")
            
        if not os.path.exists(self.mask_dir):
            print(f"ERROR: Mask folder not found at: {self.mask_dir}")
            raise FileNotFoundError(f"Missing folder: {self.mask_dir}")

        all_img_files = set([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        all_mask_files = set([f for f in os.listdir(self.mask_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.ids = sorted(list(all_img_files.intersection(all_mask_files)))

        if not self.ids:
            raise ValueError(f"No matching image/mask pairs found in {data_dir}. Check filenames!")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        new_mask = np.zeros_like(mask, dtype=np.int64)
        for raw, cls in VALUE_MAP.items():
            new_mask[mask == raw] = cls

        aug = self.transform(image=img, mask=new_mask)
        return aug['image'], aug['mask'].long()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SegHead(nn.Module):
    def __init__(self, in_ch, n_cls, th, tw):
        super().__init__()
        self.th, self.tw = th, tw
        h = 512 if in_ch >= 768 else 256

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h, 1), nn.BatchNorm2d(h), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1, groups=h), nn.Conv2d(h, h, 1), nn.BatchNorm2d(h), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1, groups=h), nn.Conv2d(h, h, 1), nn.BatchNorm2d(h), nn.GELU(),
            nn.Conv2d(h, h//2, 3, padding=1), nn.BatchNorm2d(h//2), nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(h//2, n_cls, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        # Handle cases where batch size might drop (last batch)
        x = x.reshape(B, self.th, self.tw, C).permute(0,3,1,2)
        return self.net(x)

# ============================================================================
# UTILS & METRICS
# ============================================================================

class Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.w = weights

    def forward(self, p, t):
        ce = self.ce(p, t)
        probs = F.softmax(p, dim=1)
        t_oh = F.one_hot(t, p.shape[1]).permute(0,3,1,2).float()
        inter = (probs * t_oh).sum(dim=(0,2,3))
        union = (probs + t_oh).sum(dim=(0,2,3))
        dice = (2*inter + 1e-6) / (union + 1e-6)
        
        if self.w is not None:
            dice_loss = 1 - (dice * self.w.to(dice.device)).sum() / self.w.sum()
        else:
            dice_loss = 1 - dice.mean()
        return ce + 0.5 * dice_loss

def compute_iou(pred, target):
    pred = pred.argmax(dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for c in range(N_CLASSES):
        p, t = pred==c, target==c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union > 0:
            ious.append((inter/union).item())
    return np.mean(ious) if ious else 0

def get_class_weights(data_dir):
    print("Computing class weights...")
    mask_dir = os.path.join(data_dir, 'segmented')
    
    # [FIX] Check if folder exists before listing
    if not os.path.exists(mask_dir):
         print(f"Warning: segmented folder not found at {mask_dir}. Using default weights.")
         return torch.ones(N_CLASSES)

    files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')][:50]
    counts = Counter()
    
    if not files:
        print("Warning: No mask files found for weight calculation. Using default.")
        return torch.ones(N_CLASSES)

    for f in files:
        m = np.array(Image.open(os.path.join(mask_dir, f)))
        for raw, cls in VALUE_MAP.items():
            counts[cls] += (m == raw).sum()
    
    total = sum(counts.values())
    if total == 0: return torch.ones(N_CLASSES)
    
    w = []
    for i in range(N_CLASSES):
        freq = counts.get(i, 1) / total
        w.append((1/(freq+1e-6))**0.5)
    w = np.array(w)
    return torch.tensor(w / w.sum() * N_CLASSES, dtype=torch.float32)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("="*50)
    print("  STARTING GPU TRAINING")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # [FIX] Auto-optimize for RTX 4050 (6GB VRAM)
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f}GB")
        global BATCH_SIZE, IMG_H, IMG_W
        if vram < 7: # If less than 7GB (RTX 4050 is ~6GB)
            BATCH_SIZE = 8
            IMG_H, IMG_W = 266, 476  # Lower resolution to fit VRAM
            print(f"Config: Low VRAM Mode -> Batch: {BATCH_SIZE}, Res: {IMG_H}x{IMG_W}")
        else:
            print(f"Config: High VRAM Mode -> Batch: {BATCH_SIZE}, Res: {IMG_H}x{IMG_W}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare Data
    try:
        train_ds = SegDataset(DATA_DIR, get_transforms(IMG_H, IMG_W, train=True))
        val_ds = SegDataset(VAL_DIR, get_transforms(IMG_H, IMG_W, train=False))
        train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Dataset Loaded: {len(train_ds)} train, {len(val_ds)} val")
    except Exception as e:
        print(f"\nCRITICAL ERROR LOADING DATA: {e}")
        return

    # Load Backbone
    print("Loading DINOv2-base backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    # Get Embeddings Dimensions
    with torch.no_grad():
        dummy = torch.randn(1,3,IMG_H,IMG_W).to(device)
        feat = backbone.forward_features(dummy)["x_norm_patchtokens"]
        n_emb, th, tw = feat.shape[2], IMG_H//14, IMG_W//14
    print(f"Feature Map: {th}x{tw} patches")

    # Initialize Model
    model = SegHead(n_emb, N_CLASSES, th, tw).to(device)
    
    # Optimizer & Loss
    weights = get_class_weights(DATA_DIR).to(device)
    criterion = Loss(weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_iou = 0
    history = {'loss':[], 'iou':[]}

    print(f"\nTraining for {EPOCHS} epochs...")
    
    for ep in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_ld, desc=f"Epoch {ep+1}/{EPOCHS}", colour="cyan")
        
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                with torch.no_grad():
                    feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
                out = model(feat)
                out = F.interpolate(out, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation Loop
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_ld:
                imgs = imgs.to(device)
                masks = masks.to(device)
                with autocast():
                    feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    out = model(feat)
                    out = F.interpolate(out, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                val_iou += compute_iou(out, masks)

        avg_val_iou = val_iou / len(val_ld)
        scheduler.step()
        
        # Save History
        history['loss'].append(train_loss/len(train_ld))
        history['iou'].append(avg_val_iou)

        print(f"  Loss: {train_loss/len(train_ld):.4f} | mIoU: {avg_val_iou:.4f}", end="")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best.pth'))
            print(" [Saved Best]")
        else:
            print()

    # Final Save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final.pth'))
    print(f"\nDONE! Best mIoU: {best_iou:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()