import logging
import math
from typing import List, Tuple
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

def get_pixel_resolution(src) -> Tuple[float, float]:
    """Безопасно извлекает размер пикселя в метрах из GeoTIFF."""
    px = abs(src.transform.a)
    py = abs(src.transform.e)
    if px <= 0 or py <= 0:
        raise ValueError(f"Некорректный affine transform. Размер пикселя <= 0.")
    return px, py

def generate_meter_windows(
    src, 
    tile_size_meters: float, 
    overlap: float = 0.3
) -> List[Tuple[Window, Affine]]:
    """
    Генерирует окна (тайлы) строго в реальных метрах.
    Возвращает список кортежей: (rasterio.Window, tile_Affine_transform).
    """
    px, py = get_pixel_resolution(src)
    
    # Переводим метры в пиксели
    win_w_px = max(1, math.ceil(tile_size_meters / px))
    win_h_px = max(1, math.ceil(tile_size_meters / py))
    
    # Шаг с учетом перекрытия (overlap)
    step_x = max(1, int(win_w_px * (1.0 - overlap)))
    step_y = max(1, int(win_h_px * (1.0 - overlap)))
    
    windows = []
    for col_off in range(0, src.width, step_x):
        for row_off in range(0, src.height, step_y):
            # Обрезаем окна, чтобы не вылезти за границы растра
            w = min(win_w_px, src.width - col_off)
            h = min(win_h_px, src.height - row_off)
            
            if w < 10 or h < 10:  # Игнорируем мусорные обрезки на краях
                continue
                
            window = Window(col_off, row_off, w, h)
            tile_transform = rasterio.windows.transform(window, src.transform)
            windows.append((window, tile_transform))
            
    logger.info(f"🗺️ Сгенерировано {len(windows)} окон (размер {win_w_px}x{win_h_px} px, overlap {overlap})")
    return windows