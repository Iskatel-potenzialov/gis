import sys
import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Добавляем корень проекта в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import load_npz, remove_border_objects, filter_by_area_ratio
from src.augmentations import generate_synthetic_sample, blend
from src.models import UNet
from src.losses import ImprovedAdaptiveFocalLoss
from src.train import NPZDataset, iou_score, visualize_batch

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s │ %(levelname)-7s │ %(message)s',
        datefmt='%H:%M:%S'
    )

def load_npz(path):
    """Загрузка .npz файла"""
    data = np.load(path, allow_pickle=True)
    return data['x'], data['mask']

def preprocess_data(X, mask, norm_stats, target_size=256):
    """Предобработка данных"""
    # Нормализация
    if norm_stats is not None:
        for ch in range(X.shape[-1]):
            mean = norm_stats['mean'][ch]
            std = norm_stats['std'][ch]
            if std > 0:
                X[..., ch] = (X[..., ch] - mean) / std
    
    # Ресайз
    from PIL import Image
    X_img = Image.fromarray((X * 255).astype(np.uint8))
    X_img = X_img.resize((target_size, target_size), Image.BILINEAR)
    X = np.array(X_img) / 255.0
    
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize((target_size, target_size), Image.NEAREST)
    mask = np.array(mask_img)
    
    # Удаление граничных объектов
    mask_bin = (mask > 0).astype(np.uint8)
    mask_bin, _ = remove_border_objects(mask_bin, None)
    
    # Фильтрация по площади
    mask_bin, has_valid = filter_by_area_ratio(mask_bin, min_ratio=0.001, max_ratio=0.10)
    
    return X, mask_bin

def generate_mock_data(num_samples=100, output_dir='mock_dataset'):
    """Генерация фейковых данных для тестирования пайплайна"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Создаем фейковый тензор 256x256x8
        X = np.random.rand(256, 256, 8).astype(np.float32)
        
        # Создаем фейковую маску с "курганом"
        mask = np.zeros((256, 256), dtype=np.uint8)
        if np.random.random() > 0.3:  # 70% шанс наличия объекта
            cx, cy = np.random.randint(50, 200, 2)
            size = np.random.randint(20, 50)
            mask[cx-size//2:cx+size//2, cy-size//2:cy+size//2] = 1
        
        # Сохраняем
        np.savez_compressed(
            os.path.join(output_dir, f'sample_{i:04d}.npz'),
            x=X,
            mask=mask,
            norm_params=[{'min': 0.0, 'max': 1.0} for _ in range(8)]
        )
    
    logging.info(f"✅ Создано {num_samples} фейковых примеров в {output_dir}")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Training pipeline for kurgan detection')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training .npz files')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation .npz files')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--norm_stats', type=str, default=None, help='Path to norm_stats.json')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--generate_mock', action='store_true', help='Generate mock data for testing')
    parser.add_argument('--mock_samples', type=int, default=100, help='Number of mock samples')
    
    args = parser.parse_args()
    
    # Генерация фейковых данных для тестирования
    if args.generate_mock:
        logging.info(" Генерация фейковых данных...")
        generate_mock_data(args.mock_samples, 'mock_train')
        generate_mock_data(args.mock_samples // 2, 'mock_val')
        args.train_dir = 'mock_train'
        args.val_dir = 'mock_val'
    
    # Загрузка нормализации
    norm_stats = None
    if args.norm_stats and os.path.exists(args.norm_stats):
        with open(args.norm_stats, 'r') as f:
            norm_stats = json.load(f)
        logging.info(f"📊 Загружена нормализация из {args.norm_stats}")
    
    # Создание датасетов
    train_files = [os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith('.npz')]
    val_files = [os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir) if f.endswith('.npz')]
    
    logging.info(f" Найдено {len(train_files)} тренировочных и {len(val_files)} валидационных файлов")
    
    train_ds = NPZDataset(train_files, transform=None, norm_stats=norm_stats, target_size=256, is_train=True)
    val_ds = NPZDataset(val_files, transform=None, norm_stats=norm_stats, target_size=256, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Модель и лосс
    model = UNet(in_ch=8, out_ch=1, base_filters=64).to(args.device)
    criterion = ImprovedAdaptiveFocalLoss(alpha=0.9, gamma=1.5, dice_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Цикл обучения
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_iou = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, mask_batch in train_loader:
            X_batch = X_batch.to(args.device)
            mask_batch = mask_batch.to(args.device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, mask_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for X_batch, mask_batch in val_loader:
                X_batch = X_batch.to(args.device)
                mask_batch = mask_batch.to(args.device)
                
                logits = model(X_batch)
                loss = criterion(logits, mask_batch)
                val_loss += loss.item()
                
                # IoU
                probs = torch.sigmoid(logits).cpu().numpy()
                mask_np = mask_batch.cpu().numpy()
                val_iou += iou_score(probs, mask_np, thr=0.5)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")
        
        # Сохранение лучшей модели
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            checkpoint_path = os.path.join(args.output_dir, f'best_model_epoch{epoch+1}_iou{avg_val_iou:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': avg_val_iou,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logging.info(f"💾 Сохранена лучшая модель: {checkpoint_path}")
    
    logging.info(f"🎉 Обучение завершено! Лучший Val IoU: {best_val_iou:.4f}")

if __name__ == '__main__':
    main()