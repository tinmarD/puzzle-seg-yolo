import pandas as pd
import yaml
import mlflow
from pathlib import Path
import re

# Run path
run_path = '/home/tinmar/Desktop/Puzzle/Preprocessing/runs/segment/train10'
EXPERIMENT_NAME = 'yolo-seg'

# 1. Configuration
OLD_RUNS_DIR = Path(run_path) # Path to your previous run
mlflow.set_experiment(EXPERIMENT_NAME)

def migrate_run(run_path):
    run_path = Path(run_path)
    if not (run_path / "results.csv").exists():
        print(f"Skipping {run_path}: results.csv not found.")
        return

    # Load parameters
    with open(run_path / "args.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load metrics
    df = pd.read_csv(run_path / "results.csv")
    df.columns = df.columns.str.strip() 

    # --- SANITIZATION STEP ---
    # Replace ( ) / and other non-allowed chars with underscores
    def sanitize_key(name):
        return re.sub(r'[^a-zA-Z0-9._\-/ ]', '_', name)

    with mlflow.start_run(run_name=run_path.name):
        mlflow.log_params(params)

        for _, row in df.iterrows():
            epoch = int(row['epoch'])
            
            # Create a clean dictionary of metrics
            metrics = {}
            for k, v in row.items():
                if k != 'epoch':
                    clean_name = sanitize_key(k)
                    # MLflow requires metric values to be floats and not NaN
                    if pd.notnull(v):
                        metrics[clean_name] = float(v)
            
            mlflow.log_metrics(metrics, step=epoch)

# Run for a specific folder
BASE_PATH = Path("runs/segment")
for folder in BASE_PATH.iterdir():
    if folder.is_dir() and (folder / "results.csv").exists():
        print(f"Processing {folder.name}...")
        migrate_run(folder)

migrate_run(OLD_RUNS_DIR)
