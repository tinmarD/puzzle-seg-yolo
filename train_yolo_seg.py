import os
import mlflow
from ultralytics import YOLO, settings
from PIL import Image
import pandas as pd
import numpy as np
import torch


settings.update({'mlflow': True})

### ------ Parameters ------ ###
# model_weight_path = r'/home/tinmar/Desktop/Puzzle/Preprocessing/runs/segment/train15/weights/best.pt'
model_weight_path = ''
model_version = 'yolov8x-seg'
#model_version = 'yolo26l-seg.pt'
description = 'Added negative images'
data_conf_file = 'puzzle_yolo_seg_train.yaml'
mlflow_uri = 'http://192.168.1.197:5000'
test_images_dir = '/home/tinmar/Desktop/Puzzle/tiles/yolo_seg_training/images/test/'
## ---------------------------##


if model_weight_path:
    print("Continuing training from existing weights.")
    model = YOLO(model_weight_path)

# Load a COCO-pretrained YOLOv8n model
model = YOLO(model_version)

# Display model information (optional)
model.info()

# Tell Ultralytics to keep the run open
os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "yolo-seg"
#os.environ["MLFLOW_RUN"] = "RUN_NAME"

mlflow.set_tracking_uri(mlflow_uri)

# Train the model 
results = model.train(
    data=data_conf_file,
    epochs=400,
    imgsz=640,
    batch=4,
    hsv_h=0.25,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=45,
    translate=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.0,
    freeze=None
    )

# Add your custom attributes to the STILL ACTIVE run
# Ultralytics has already logged the params and metrics here.
mlflow.set_tag("model-ver", model_version)
    
# Set the Description (which is stored in a special tag)
mlflow.set_tag("mlflow.note.content", description)


# Run inference on test images
test_images_names = os.listdir(test_images_dir)
test_images_path = [os.path.join(test_images_dir, name) for name in test_images_names if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
results = model.predict(test_images_path, save=False, conf=0.25, imgsz=640, show_conf=False, show_labels=False, show_boxes=False)

# 2. Prepare data for the table
rows = []
for result in results:
    # Generate the visual overlay (Image + Mask)
    # result.plot() returns a BGR numpy array
    # overlay_bgr = result.plot(labels=False, boxes=False) 
    # overlay_rgb = Image.fromarray(overlay_bgr[..., ::-1]) 

    # masks_data = result.masks.data.cpu().numpy()
    # tile_mask = (np.max(masks_data, axis=0) * 255).astype(np.uint8)
    # mask_pil = Image.fromarray(tile_mask)

    h, w = result.orig_shape
    combined_binary_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None:
        # result.masks.data are the raw tensors
        # We move to CPU, upscale to original image size, and take the max
        # across all detected objects to get one single mask
        masks_tensor = result.masks.data # (N, H, W)
        
        # Upscale masks to original image resolution
        masks_tensor = torch.nn.functional.interpolate(
            masks_tensor.unsqueeze(1), 
            size=(h, w), 
            mode="bilinear"
        ).squeeze(1)
        
        # Merge all individual instance masks into one binary image
        # Any pixel belonging to any instance becomes 255 (white)
        merged_mask = (torch.any(masks_tensor > 0.5, dim=0).int() * 255)
        combined_binary_mask = merged_mask.cpu().numpy().astype(np.uint8)

    # 2. Convert to PIL for MLflow Table compatibility
    mask_pil = Image.fromarray(combined_binary_mask)

    # Add a row: [filename, original_image, overlay_mask, mean_confidence]
    rows.append({
        "path": result.path,
        "original_image": Image.open(result.path),
        "predicted_mask": mask_pil,
        "mean_confidence": float(result.boxes.conf.mean()) if result.boxes else 0.0
    })

results_df = pd.DataFrame(rows)

mlflow.log_table(
        data=results_df, 
        artifact_file="test_results/segmentation_summary.json"
)

# Manually end the run
mlflow.end_run()
