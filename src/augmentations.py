import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = 256
SCALE_FACTORS = [0.7, 0.6, 0.5, 0.5]
SCALE_PROBABILITIES = [0.25, 0.25, 0.25, 0.25]

def blend(fg, bg, alpha):
    """fg, bg — могут быть 2D или 3D"""
    if fg.ndim == 2:
        fg = fg[..., None]
    if bg.ndim == 2:
        bg = bg[..., None]
    alpha = alpha[..., None] # (H,W,1)
    return (fg * alpha + bg * (1 - alpha)).astype(np.uint8)

def generate_synthetic_sample(orig_files, fon_files, target_size=256, border_margin=8):
    """
    Твоя кастомная аугментация: Copy-Paste с Alpha-Blending.
    Бесшовная вставка кургана в случайный фон.
    """
    # Загрузка оригинала
    orig_c = np.array(Image.open(orig_files['c']).convert('RGB'))
    orig_ch = np.array(Image.open(orig_files['ch']).convert('RGB'))
    orig_g = np.array(Image.open(orig_files['g']).convert('L')) 
    orig_i = np.array(Image.open(orig_files['i']).convert('L')) 
    orig_mask = np.array(Image.open(orig_files['mask']).convert('L'))

    # Загрузка фона
    fon_c = np.array(Image.open(fon_files['c']).convert('RGB'))
    fon_ch = np.array(Image.open(fon_files['ch']).convert('RGB'))
    fon_g = np.array(Image.open(fon_files['g']).convert('L'))
    fon_i = np.array(Image.open(fon_files['i']).convert('L'))

    # Создание плавной маски (alpha) через Gaussian Blur
    kernel_size = 15
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(orig_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    alpha_2d = np.clip(blurred / 255.0, 0, 1) # (256,256)

    # Бесшовное смешивание для каждого канала
    synth_c = blend(orig_c, fon_c, alpha_2d)
    synth_ch = blend(orig_ch, fon_ch, alpha_2d)
    synth_g = blend(orig_g, fon_g, alpha_2d)
    synth_i = blend(orig_i, fon_i, alpha_2d)
    
    # Маска остается бинарной
    synth_mask = (orig_mask > 0).astype(np.uint8) * 255

    return {
        'c': synth_c, 'ch': synth_ch, 'g': synth_g, 'i': synth_i, 'mask': synth_mask
    }