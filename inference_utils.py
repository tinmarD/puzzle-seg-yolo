import cv2
from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path

def process_large_image(image_path, model_path, tile_size=640, overlap=50, conf_thresh=0.8):
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 1. Calculate Stride
    # Stride is how far the window moves each time
    stride = tile_size - overlap
    
    # 2. Initialize full-size binary mask
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # 3. Generate start coordinates for tiles
    # We ensure we cover the whole range by adding tiles at the very edges
    y_coords = list(range(0, h - tile_size, stride)) + [h - tile_size]
    x_coords = list(range(0, w - tile_size, stride)) + [w - tile_size]
    
    # Remove duplicates if image size is perfectly divisible
    y_coords = sorted(list(set(y_coords)))
    x_coords = sorted(list(set(x_coords)))

    for y in y_coords:
        for x in x_coords:
            # Extract Tile
            tile = img[y : y + tile_size, x : x + tile_size]
            
            # Run YOLO (retina_masks=True to keep 640x640 resolution)
            results = model.predict(tile, imgsz=tile_size, conf=0.25, retina_masks=True, verbose=False)
            # Select results with confidence above threshold
            if len(results[0]) == 0: 
                continue
            ind_sel = [True if r.boxes.conf.cpu().numpy().max() >= conf_thresh else False for r in results[0]]
            results_sel = results[0][ind_sel]
            
            if results_sel.masks is not None:
                # Merge all object masks in this tile
                masks_data = results_sel.masks.data.cpu().numpy()
                tile_mask = (np.max(masks_data, axis=0) * 255).astype(np.uint8)

                # 4. Merge into full mask using np.maximum
                # This handles the overlap area by taking the highest value (white)
                current_roi = full_mask[y : y + tile_size, x : x + tile_size]
                full_mask[y : y + tile_size, x : x + tile_size] = np.maximum(current_roi, tile_mask)

    return full_mask


def process_test_images(input_test_dir, output_res_dir, model_path, tile_size=640, overlap=50, conf_thresh=0.8):
    input_test_dir = Path(input_test_dir)
    output_res_dir = Path(output_res_dir)
    output_res_dir.mkdir(parents=True, exist_ok=True)

    for img_file in input_test_dir.glob('*.*'):
        print(f"Processing {img_file.name}...")
        full_mask = process_large_image(str(img_file), model_path, tile_size, overlap, conf_thresh)
        
        # Save the resulting mask
        output_mask_path = output_res_dir / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(output_mask_path), full_mask)
        print(f"Saved mask to {output_mask_path}")


if __name__ == "__main__":
    # Example usage
    input_test_dir = '/home/tinmar/Desktop/Projects/Datasets/Puzzle/global_images/test_images'
    model_path = './runs/segment/train8/weights/best.pt'  # Path to your trained model
    model_dir = Path(model_path).parent.parent
    output_res_dir =  str(model_dir / 'test_masks')

    process_test_images(input_test_dir, output_res_dir, model_path)