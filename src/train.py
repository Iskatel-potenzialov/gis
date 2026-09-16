import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class NPZDataset(Dataset):
    def __init__(self, file_list, transform=None, norm_stats=None, target_size=256, is_train=True):
        self.files = file_list
        self.transform = transform
        self.norm_stats = norm_stats
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path, allow_pickle=True)
        X = d["x"]
        mask = d["mask"]
        
        # Нормализация (если передана)
        if self.norm_stats is not None:
            for ch in range(X.shape[-1]):
                mean = self.norm_stats["mean"][ch]
                std = self.norm_stats["std"][ch]
                if std > 0:
                    X[..., ch] = (X[..., ch] - mean) / std
                    
        if self.transform:
            augmented = self.transform(image=X, mask=mask)
            X = augmented["image"]
            mask = augmented["mask"]
            
        X = np.transpose(X, (2, 0, 1)).astype(np.float32) # (C, H, W)
        mask = mask.astype(np.float32)
        if mask.ndim == 2:
            mask = mask[None, ...] # (1, H, W)
            
        return torch.from_numpy(X), torch.from_numpy(mask)

# ==========================================
# МЕТРИКИ И ВИЗУАЛИЗАЦИЯ
# ==========================================
def iou_score(pred, target, thr=0.5, eps=1e-6):
    pred_bin = (pred >= thr).astype(np.uint8)
    target_bin = (target >= 0.5).astype(np.uint8)
    inter = (pred_bin & target_bin).sum()
    union = (pred_bin | target_bin).sum()
    return (inter + eps) / (union + eps)

def visualize_batch(X_batch, mask_batch, pred_probs_batch, n=3, alpha=0.35, norm_stats=None):
    B = min(n, X_batch.shape[0])
    for i in range(B):
        X = X_batch[i].cpu().numpy() # (C, H, W)
        mask = mask_batch[i, 0].cpu().numpy()
        pred_probs = pred_probs_batch[i, 0].cpu().numpy()
        pred_bin = (pred_probs >= 0.5).astype(np.uint8)

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        
        # Первый ряд - исходные каналы
        for j in range(4):
            if j < X.shape[0]:
                axs[0, j].imshow(X[j], cmap='gray')
            else:
                axs[0, j].imshow(np.zeros_like(mask), cmap='gray')
            axs[0, j].set_title(f'Channel {j}')
            axs[0, j].axis('off')
            
        # Второй ряд - результаты
        axs[1, 0].imshow(mask, cmap="gray")
        axs[1, 0].set_title('True Mask')
        axs[1, 0].axis('off')
        
        im = axs[1, 1].imshow(pred_probs, cmap="Blues", vmin=0, vmax=1)
        axs[1, 1].set_title('Prediction Probabilities')
        axs[1, 1].axis('off')
        plt.colorbar(im, ax=axs[1, 1], fraction=0.046)
        
        axs[1, 2].imshow(pred_bin, cmap="gray")
        axs[1, 2].set_title('Binary Prediction')
        axs[1, 2].axis('off')
        
        overlay = np.zeros((*mask.shape, 3))
        overlay[..., 0] = mask # Red for True
        overlay[..., 1] = pred_bin # Green for Pred
        axs[1, 3].imshow(overlay)
        axs[1, 3].set_title('Overlay (Red=True, Green=Pred)')
        axs[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()