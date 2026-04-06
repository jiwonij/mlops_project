# config.py

import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATASET_DIR = BASE_DIR / "dataset"

UPLOAD_DIR = BACKEND_DIR / "uploaded_files"
IMAGE_DIR = BACKEND_DIR / "view_model_architecture"
MODEL_IMG_DIR = BACKEND_DIR / "model_images"

MODEL_SAVE_PATH = ARTIFACT_DIR / "model.keras"
FEATURE_SCALER_PATH = ARTIFACT_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = ARTIFACT_DIR / "target_scaler.pkl"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

MODEL_PLOT_PATH = IMAGE_DIR / "forecast_model.png"
MODEL_SHAPES_PLOT_PATH = IMAGE_DIR / "forecast_model_shapes.png"
PREDICTION_PLOT_PATH = MODEL_IMG_DIR / "forecast_result.png"

DEFAULT_TRAIN_PATH = DATASET_DIR / "train.csv"
DEFAULT_TEST_PATH = DATASET_DIR / "test.csv"
BUILDING_INFO_PATH = DATASET_DIR / "building_info.csv"

RMSE_THRESHOLD = 15.0
RETRAIN_SCRIPT_PATH = BASE_DIR / "retrain.py"
PYTHON_EXECUTABLE = "python"


REQUIRED_DIRS = [
    UPLOAD_DIR,
    IMAGE_DIR,
    MODEL_IMG_DIR,
    ARTIFACT_DIR,
]