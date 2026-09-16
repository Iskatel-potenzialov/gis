import sys
import os
import json
import logging
import argparse
import torch
import rasterio

# Добавляем корень проекта в путь, чтобы импортировать src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geo_tiling import generate_meter_windows
from src.inference_engine import run_batched_inference
from src.postprocessing import stitch_masks, extract_polygons
# from src.models import UNetWithInputMix  # Твоя модель

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s │ %(levelname)-7s │ %(message)s',
        datefmt='%H:%M:%S'
    )

def load_model(weights_path: str, device: str):
    logging.info(f"⏳ Загрузка модели из {weights_path}...")
    # model = UNetWithInputMix(in_ch=8, out_ch=1, base_filters=64)
    # checkpoint = torch.load(weights_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state'])
    # model.to(device)
    # return model
    pass # Заглушка для примера

def process_region(tiff_path: str, model, output_geojson: str, device: str):
    logging.info(f"📁 Обработка территории: {os.path.basename(tiff_path)}")
    
    with rasterio.open(tiff_path) as src:
        # 1. Нарезка на тайлы в метрах
        windows_meta = generate_meter_windows(src, tile_size_meters=200, overlap=0.3)
        
        # 2. Чтение данных и Инференс
        tiles_data = []
        for window, _ in windows_meta:
            tile = src.read(window=window)
            # Rasterio читает как (C, H, W), нам нужно (H, W, C)
            tile = np.transpose(tile, (1, 2, 0)) 
            tiles_data.append(tile)
            
        pred_masks = run_batched_inference(model, tiles_data, device=device, threshold=0.5)
        
        # 3. Сборка единой маски
        full_mask = stitch_masks(pred_masks, windows_meta, (src.height, src.width))
        
        # 4. Векторизация и сохранение
        geojson_data = extract_polygons(
            full_mask, 
            src.transform, 
            src_crs=src.crs.to_string(),
            target_crs="EPSG:4326" # WGS84 для карт
        )
        
    with open(output_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False)
        
    logging.info(f"✅ ЗАВЕРШЕНО | GeoJSON сохранен: {output_geojson}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Kurgan Detection Inference Pipeline")
    parser.add_argument("--input", required=True, help="Путь к GeoTIFF файлу")
    parser.add_argument("--output", required=True, help="Путь для сохранения GeoJSON")
    parser.add_argument("--weights", required=True, help="Путь к весам модели (.pth)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    try:
        model = load_model(args.weights, args.device)
        process_region(args.input, model, args.output, args.device)
    except Exception as e:
        logging.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()