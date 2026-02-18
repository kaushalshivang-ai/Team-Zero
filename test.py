import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# Update strictly for your PC
TEST_RGB_DIR  = r"D:\Model\Offroad_Segmentation_testImages\rgb"
TEST_SEG_DIR  = r"D:\Model\Offroad_Segmentation_testImages\segmented"
MODEL_PATH    = r"D:\Model\Team Zero\Output\best.pth"
OUTPUT_DIR    = r"D:\Model\Team Zero\test\output"

# DINOv2 Training Dimensions
IMG_H, IMG_W  = 518, 952  
NUM_CLASSES   = 11
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

COLORS = np.array([
    [0, 0, 0], [34, 139, 34], [0, 200, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 128], [255, 105, 180], [101, 67, 33], [169, 169, 169], 
    [135, 206, 235], [70, 130, 180]
], dtype=np.uint8)

MASK_MAPPING = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
}

# ============================================================================
# 2. MODEL DEFINITION
# ============================================================================
class SegHead(nn.Module):
    def __init__(self, in_ch, n_cls):
        super().__init__()
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
        # Dynamically calculate H/W from tokens to prevent shape mismatch errors
        H_tokens = IMG_H // 14
        W_tokens = IMG_W // 14
        
        # Reshape to (Batch, Height, Width, Channels) then permute to (B, C, H, W)
        x = x.reshape(B, H_tokens, W_tokens, C).permute(0,3,1,2)
        return self.net(x)

# ============================================================================
# 3. DATA LOADERS
# ============================================================================
LUT = np.zeros(10001, dtype=np.uint8)
for val, idx in MASK_MAPPING.items(): LUT[val] = idx

class TestDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps = [os.path.join(masks_dir, i) for i in self.ids] if masks_dir else None
        
        self.transform = A.Compose([
            A.Resize(IMG_H, IMG_W),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Default placeholder if no mask exists
        mask_tensor = torch.tensor([-1]) 
        
        if self.masks_fps:
            mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            mask = np.clip(mask, 0, 10000).astype(np.int32)
            mask = LUT[mask]
            mask_tensor = torch.from_numpy(mask).long()

        aug = self.transform(image=image)
        return aug['image'], mask_tensor, self.ids[i]

# ============================================================================
# 4. METRICS & PLOTTING
# ============================================================================
def mask_to_color(mask_np):
    return COLORS[mask_np]

def compute_metrics(pred, target):
    ious, ap50s = [], []
    pred, target = pred.flatten(), target.flatten()
    for cls in range(NUM_CLASSES):
        target_c = target == cls
        if target_c.sum() == 0: 
            ious.append(float('nan'))
            ap50s.append(float('nan'))
            continue
        pred_c = pred == cls
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        iou = inter / (union + 1e-6)
        ious.append(iou)
        ap50s.append(1.0 if iou >= 0.50 else 0.0)
    return ious, ap50s

def save_result(img_t, pred, target, fname):
    img = img_t.cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Only show 3rd column if target mask exists
    has_target = (target.size > 0)
    cols = 3 if has_target else 2
    
    fig, ax = plt.subplots(1, cols, figsize=(6*cols, 5))
    ax[0].imshow(img); ax[0].set_title("Original")
    ax[1].imshow(mask_to_color(pred)); ax[1].set_title("Prediction")
    if has_target:
        ax[2].imshow(mask_to_color(target)); ax[2].set_title("Ground Truth")
    
    patches = [mpatches.Patch(color=COLORS[i]/255.0, label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=6)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, fname.replace('.', '_result.')), dpi=100)
    plt.close()

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Loading DINOv2 Backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval().to(DEVICE)
    
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMG_H, IMG_W).to(DEVICE)
        feat = backbone.forward_features(dummy)["x_norm_patchtokens"]
        n_emb = feat.shape[2]

    print(f"2. Loading Model from {MODEL_PATH}...")
    model = SegHead(n_emb, NUM_CLASSES).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        try:
            # strict=False allows loading even if there's a slight attribute mismatch
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
            print("   Model loaded successfully!")
        except Exception as e:
            print(f"   CRITICAL ERROR loading model: {e}")
            return
    else:
        print(f"   ERROR: File not found: {MODEL_PATH}")
        return
    model.eval()

    print("3. Starting Testing...")
    ds = TestDataset(TEST_RGB_DIR, TEST_SEG_DIR if os.path.exists(TEST_SEG_DIR) else None)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    all_ious, all_ap50s = [], []
    f_metrics = open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w")
    f_metrics.write(f"{'Image':<30} {'mIoU':<10} {'mAP50':<10}\n" + "-"*55 + "\n")

    with torch.no_grad():
        for img, mask, fname in tqdm(loader):
            img = img.to(DEVICE)
            fname = fname[0]
            
            feat = backbone.forward_features(img)["x_norm_patchtokens"]
            out = model(feat)
            out = F.interpolate(out, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
            pred = torch.argmax(out, dim=1).cpu().numpy()[0]
            
            # [FIX] Robust check for mask existence
            # mask.numel() > 1 ensures we have a full image mask, not just the placeholder [-1]
            if mask.numel() > 1:
                mask_np = mask.numpy()[0]
                ious, ap50s = compute_metrics(pred, mask_np)
                all_ious.append(ious)
                all_ap50s.append(ap50s)
                
                miou = np.nanmean(ious) * 100
                map50 = np.nanmean(ap50s) * 100
                f_metrics.write(f"{fname:<30} {miou:6.2f}%    {map50:6.2f}%\n")
                save_result(img[0], pred, mask_np, fname)
            else:
                f_metrics.write(f"{fname:<30} {'--':<10} {'--':<10}\n")
                save_result(img[0], pred, np.array([]), fname)

    f_metrics.close()
    
    if all_ious:
        iou_mat = np.array(all_ious)
        ap50_mat = np.array(all_ap50s)
        
        class_miou = np.nanmean(iou_mat, axis=0)
        class_map50 = np.nansum(ap50_mat, axis=0) / (np.sum(~np.isnan(ap50_mat), axis=0) + 1e-6)
        
        print("\n" + "="*50)
        print(f"{'CLASS':<15} {'mIoU':>10} {'mAP50':>10}")
        print("-"*50)
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:<15} {class_miou[i]*100:9.1f}% {class_map50[i]*100:9.1f}%")
        print("="*50)
        print(f"{'OVERALL':<15} {np.nanmean(class_miou)*100:9.1f}% {np.mean(class_map50)*100:9.1f}%")
    else:
        print("\nNo valid masks found. Generated predictions only.")

if __name__ == "__main__":
    main()