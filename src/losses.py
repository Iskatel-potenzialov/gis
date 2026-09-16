import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedAdaptiveFocalLoss(nn.Module):
    """
    Твой кастомный лосс, объединяющий Focal Loss, Dice Loss и динамический 
    штраф за пропуск мелких объектов.
    
    Параметры (точно по твоему исходному коду):
    - alpha: баланс классов (например, 0.9)
    - gamma: фокусировка на сложных примерах (например, 1.5)
    - pos_weight: вес положительного класса (курганов) для BCE
    - small_object_weight: множитель штрафа за пропуск маленьких объектов (например, 3.0)
    - small_obj_area_frac: порог площади, ниже которой объект считается "мелким" (например, 0.002)
    - dice_weight: вес компонента Dice Loss (например, 1.0)
    """
    def __init__(
        self, 
        alpha: float = 0.9, 
        gamma: float = 1.5, 
        pos_weight: float = None,
        small_object_weight: float = 3.0,
        small_obj_area_frac: float = 0.002,
        dice_weight: float = 1.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.small_object_weight = small_object_weight
        self.small_obj_area_frac = small_obj_area_frac
        self.dice_weight = dice_weight
        self.eps = eps
        
        # Фиксируем pos_weight, если он передан
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, return_components: bool = False):
        """
        logits: (B, 1, H, W) - сырые предсказания модели
        targets: (B, 1, H, W) - бинарные маски (0 или 1)
        """
        B = logits.shape[0]
        device = logits.device
        
        # Приводим targets к нужному типу
        targets = targets.float()
        
        # ==========================================
        # 1. FOCAL LOSS COMPONENT
        # ==========================================
        # BCEWithLogitsLoss с учетом pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Вероятности для Focal weighting
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal modulation: уменьшаем вес легких примеров (фона)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = (focal_weight * bce_loss).mean()
        
        # ==========================================
        # 2. SMALL OBJECT PENALTY (Твоя фишка!)
        # ==========================================
        # Ищем маски, где площадь объекта меньше порога
        # Считаем долю положительных пикселей в каждом примере батча
        pos_fraction = targets.view(B, -1).sum(dim=1) / targets.view(B, -1).shape[1]
        
        # Если объект мелкий (доля пикселей < small_obj_area_frac), 
        # мы умножаем его вклад в лосс на small_object_weight
        small_obj_mask = (pos_fraction < self.small_obj_area_frac).float()
        
        # Динамический вес: для мелких объектов вес = small_object_weight, для крупных = 1.0
        dynamic_weight = 1.0 + small_obj_mask * (self.small_object_weight - 1.0)
        
        # Применяем динамический вес к focal loss (усредняем по батчу с учетом весов)
        # Пересчитываем focal loss по батчу, чтобы применить веса
        focal_loss_per_sample = (focal_weight * bce_loss).view(B, -1).sum(dim=1) / (bce_loss.view(B, -1).shape[1] + self.eps)
        weighted_focal_loss = (focal_loss_per_sample * dynamic_weight).mean()
        
        # ==========================================
        # 3. DICE LOSS COMPONENT
        # ==========================================
        if self.dice_weight > 0:
            probs_flat = probs.view(B, -1)
            targets_flat = targets.view(B, -1)
            
            inter = (probs_flat * targets_flat).sum(dim=1)
            union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
            
            dice_loss = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)
            dice_loss = dice_loss.mean()
        else:
            dice_loss = torch.tensor(0.0, device=device)
            
        # ==========================================
        # ИТОГОВЫЙ ЛОСС
        # ==========================================
        total_loss = weighted_focal_loss + self.dice_weight * dice_loss
        
        if return_components:
            return total_loss, weighted_focal_loss, dice_loss
        return total_loss