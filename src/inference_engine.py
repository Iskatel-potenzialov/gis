import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

def apply_feature_engineering(tile_array: np.ndarray) -> np.ndarray:
    """
    Твоя кастомная физика рельефа. 
    Превращает сырые каналы в производные (ch1*g, ch1-g и т.д.), 
    чтобы модель увидела тени и градиенты.
    """
    # Предполагаем, что на входе уже 8 каналов, но мы пересобираем их с физикой
    # Индексы зависят от того, как ты их собирал при обучении
    ch1 = tile_array[..., 3]
    g = tile_array[..., 6]
    
    # Создаем производные
    ch1_g = ch1 * g
    ch1_minus_g = ch1 - g
    ch1_plus_g = ch1 + g
    
    # Собираем новый тензор
    engineered = np.stack([
        tile_array[..., 0], tile_array[..., 1], tile_array[..., 2],
        ch1, ch1_g, ch1_minus_g, ch1_plus_g, tile_array[..., 7]
    ], axis=-1)
    
    return engineered.astype(np.float32)

def prepare_tile_tensor(tile_array: np.ndarray) -> torch.Tensor:
    """
    Превращает сырой numpy-тайл (H, W, C) в тензор (1, C, H, W).
    """
    # 1. Feature engineering
    tensor = apply_feature_engineering(tile_array)
    
    # 2. Нормализация в [0, 1] (или Z-score, если ты используешь его на инференсе)
    tensor = np.clip(tensor, 0, 1) 
    
    # 3. HWC -> CHW -> NCHW
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    
    return torch.from_numpy(tensor).float()

def run_batched_inference(
    model: nn.Module,
    tiles_data: List[np.ndarray],
    device: str,
    batch_size: int = 8,
    threshold: float = 0.5
) -> List[np.ndarray]:
    """
    Чистая функция инференса. Принимает список тайлов, возвращает список бинарных масок.
    """
    model.eval()
    all_masks = []
    
    for i in range(0, len(tiles_data), batch_size):
        batch_arrays = tiles_data[i : i + batch_size]
        batch_tensors = [prepare_tile_tensor(arr) for arr in batch_arrays]
        batch_input = torch.cat(batch_tensors, dim=0).to(device)
        
        with torch.no_grad():
            logits = model(batch_input)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        binary_masks = (probs > threshold).astype(np.uint8)
        
        for mask in binary_masks:
            all_masks.append(mask.squeeze(0)) 
            
    return all_masks