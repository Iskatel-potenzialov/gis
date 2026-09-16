import os
import re
import json
import glob
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

TARGET_SIZE = 256
MAX_MASK_RATIO = 0.10

# ==========================================
# ЗАГРУЗКА И СБОРКА ТЕНЗОРА (8 КАНАЛОВ)
# ==========================================
def load_rgb_as_3ch(path, target_size=TARGET_SIZE):
    im = Image.open(path).convert('RGB')
    if target_size is not None:
        im = im.resize((target_size, target_size), Image.BILINEAR)
    return np.array(im)

def load_gray_as_1ch(path, target_size=TARGET_SIZE, interp_bilinear=True):
    im = Image.open(path).convert('L')
    if target_size is not None:
        if interp_bilinear:
            im = im.resize((target_size, target_size), Image.BILINEAR)
        else:
            im = im.resize((target_size, target_size), Image.NEAREST)
    return np.array(im)

def build_tensor_and_mask(filepaths):
    """Создает тензор с новым набором каналов (8 каналов)"""
    ch_arr = load_rgb_as_3ch(filepaths['ch'], TARGET_SIZE) # (H,W,3)
    g_arr = load_gray_as_1ch(filepaths['g'], TARGET_SIZE, interp_bilinear=True) # (H,W)
    i_arr = load_gray_as_1ch(filepaths['i'], TARGET_SIZE, interp_bilinear=True) # (H,W)
    mask_arr = load_gray_as_1ch(filepaths['mask'], TARGET_SIZE, interp_bilinear=False) # (H,W)

    tensor = np.zeros((TARGET_SIZE, TARGET_SIZE, 8), dtype=np.float32)
    tensor[..., 0:3] = ch_arr / 255.0
    tensor[..., 3] = g_arr / 255.0
    tensor[..., 4] = i_arr / 255.0
    
    # Твоя кастомная физика рельефа
    ch1 = tensor[..., 3]
    g = tensor[..., 4]
    tensor[..., 5] = ch1 * g
    tensor[..., 6] = ch1 - g
    tensor[..., 7] = ch1 + g 

    # Нормализация по каналам
    tensor_normalized = np.zeros_like(tensor)
    norm_params = []
    for ch in range(8):
        c = tensor_normalized[..., ch]
        if np.isnan(c).any():
            c = np.nan_to_num(c, nan=0.0)
        lo, hi = float(np.nanmin(c)), float(np.nanmax(c))
        if hi > lo:
            tensor_normalized[..., ch] = (c - lo) / (hi - lo)
        else:
            tensor_normalized[..., ch] = 0.0
        norm_params.append({"min": lo, "max": hi})

    mask_bin = (mask_arr > 0).astype(np.uint8)
    return tensor_normalized, mask_bin, norm_params

# ==========================================
# УДАЛЕНИЕ ГРАНИЧНЫХ ОБЪЕКТОВ
# ==========================================
def remove_border_objects(mask_bin, images):
    """Удаляет объекты, которые касаются краев тайла"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), connectivity=8)
    
    if num_labels <= 1:
        return mask_bin, images
    
    remove_mask = np.zeros_like(mask_bin, dtype=bool)
    h, w = mask_bin.shape
    
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        # Жесткая проверка касания границ
        touches_border = (x == 0 or y == 0 or x + width == w or y + height == h)
        if touches_border:
            remove_mask = np.logical_or(remove_mask, (labels == label))
            
    mask_bin[remove_mask] = 0
    return mask_bin, images

# ==========================================
# ФИЛЬТРАЦИЯ ДАТАСЕТА ПО ПЛОЩАДИ
# ==========================================
def filter_npz_by_mask_size(npz_dir, output_dir, min_ratio=0.001, max_ratio=0.085, remove_empty_ratio=0.5):
    """
    MIN_RATIO = 0.001 # 0.1% - удаляем слишком маленькие НЕПУСТЫЕ маски
    MAX_RATIO = 0.085 # 8.5% - удаляем слишком большие НЕПУСТЫЕ маски
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    
    kept_files = 0
    removed_files = 0
    empty_masks = []
    
    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=True)
        mask = d["mask"]
        
        if mask.sum() == 0:
            empty_masks.append(npz_path)
            continue
            
        total_pixels = mask.shape[0] * mask.shape[1]
        obj_ratio = mask.sum() / total_pixels
        
        if min_ratio <= obj_ratio <= max_ratio:
            # Копируем файл
            out_path = os.path.join(output_dir, os.path.basename(npz_path))
            np.savez_compressed(out_path, x=d["x"], mask=mask, norm_params=d["norm_params"])
            kept_files += 1
        else:
            removed_files += 1
            
    # Удаляем часть пустых масок
    import random
    random.seed(42)
    num_to_remove = int(len(empty_masks) * remove_empty_ratio)
    files_to_remove = random.sample(empty_masks, num_to_remove)
    
    for f in empty_masks:
        if f not in files_to_remove:
            out_path = os.path.join(output_dir, os.path.basename(f))
            d = np.load(f, allow_pickle=True)
            np.savez_compressed(out_path, x=d["x"], mask=d["mask"], norm_params=d["norm_params"])
            kept_files += 1
        else:
            removed_files += 1
            
    return kept_files, removed_files