import os
import mlflow
from ultralytics import YOLO, settings

settings.update({'mlflow': True})

### ------ Parameters ------ ###
# model_weight_path = r'/home/tinmar/Desktop/Puzzle/Preprocessing/runs/segment/train15/weights/best.pt'
model_weight_path = ''
model_version = 'yolo11l-seg'
description = ''
data_conf_file = 'puzzle_yolov_seg_train.yaml'
mlflow_uri = 'http://192.168.1.197:5000'
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
    epochs=4,
    imgsz=640,
    batch=8,
    hsv_h=0.25,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=45,
    translate=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.0,
    )

# Add your custom attributes to the STILL ACTIVE run
# Ultralytics has already logged the params and metrics here.
mlflow.set_tag("model-ver", model_version)
    
# Set the Description (which is stored in a special tag)
mlflow.set_tag("mlflow.note.content", description)

# Manually end the run
mlflow.end_run()
