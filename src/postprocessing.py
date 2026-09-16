import logging
import cv2
import numpy as np
import rasterio
from typing import List, Tuple, Dict
from rasterio.transform import Affine
from pyproj import Transformer

logger = logging.getLogger(__name__)

def stitch_masks(
    masks: List[np.ndarray],
    windows_meta: List[Tuple[rasterio.windows.Window, Affine]],
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Сшивает предсказания тайлов в единую маску, учитывая перекрытия (overlap).
    Использует np.maximum, чтобы избежать швов на границах.
    """
    full_mask = np.zeros(target_shape, dtype=np.uint8)
    
    for mask, (window, _) in zip(masks, windows_meta):
        row_off, col_off = int(window.row_off), int(window.col_off)
        h, w = mask.shape
        
        r_end = min(row_off + h, target_shape[0])
        c_end = min(col_off + w, target_shape[1])
        
        full_mask[row_off:r_end, col_off:c_end] = np.maximum(
            full_mask[row_off:r_end, col_off:c_end], 
            mask[:r_end-row_off, :c_end-col_off]
        )
        
    return full_mask

def extract_polygons(
    mask: np.ndarray, 
    transform: Affine, 
    src_crs: str,
    target_crs: str = "EPSG:4326",
    min_area_pixels: int = 50
) -> Dict:
    """
    Векторизует пиксельную маску в полигоны, репроецирует координаты 
    и упаковывает в GeoJSON FeatureCollection.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Настраиваем трансформер координат (например, из UTM/3857 в WGS84)
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    
    features = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_pixels:
            continue
            
        # Упрощение полигона (убираем зигзаги пикселей)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        geo_coords = []
        for point in approx.reshape(-1, 2):
            # Пиксели -> Локальные координаты (UTM/Meters)
            px, py = rasterio.transform.xy(transform, point[1], point[0])
            # Локальные -> Глобальные (Lat/Lon)
            lon, lat = transformer.transform(px, py)
            geo_coords.append([lon, lat])
            
        if geo_coords and geo_coords[0] != geo_coords[-1]:
            geo_coords.append(geo_coords[0])
            
        if len(geo_coords) >= 4:  # Валидный полигон
            features.append({
                "type": "Feature",
                "properties": {"area_px": float(cv2.contourArea(cnt))},
                "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
            })
            
    logger.info(f"🗺️ Векторизация завершена. Найдено {len(features)} валидных полигонов.")
    return {"type": "FeatureCollection", "features": features}