import uuid
import subprocess
from pathlib import Path

import pandas as pd

from backend.config import (
    UPLOAD_DIR,
    RMSE_THRESHOLD,
    RETRAIN_SCRIPT_PATH,
    PYTHON_EXECUTABLE,
)

from inference import predict
from evaluate import evaluate


# -------------------------------
# 1. 업로드 파일 저장
# -------------------------------
async def save_upload_file(file):
    contents = await file.read()

    saved_name = f"{uuid.uuid4()}_{file.filename}"
    save_path = UPLOAD_DIR / saved_name

    with open(save_path, "wb") as f:
        f.write(contents)

    return save_path, saved_name


# -------------------------------
# 2. 재학습 실행
# -------------------------------
def run_retraining():
    result = subprocess.run(
        [PYTHON_EXECUTABLE, str(RETRAIN_SCRIPT_PATH)],
        capture_output=True,
        text=True
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# -------------------------------
# 3. 전체 파이프라인
# -------------------------------
def run_prediction_pipeline(file_path: Path, original_filename: str, saved_name: str):

    # 1) CSV 로드
    df = pd.read_csv(file_path)

    # 2) 예측
    predictions = predict(df)

    # 3) 평가
    eval_result = evaluate(df, predictions)

    rmse = eval_result.get("rmse")

    # 4) 재학습 여부 판단
    retraining_triggered = False
    retraining_result = None

    if rmse is not None and rmse > RMSE_THRESHOLD:
        retraining_triggered = True
        retraining_result = run_retraining()

    # 5) 응답 구성
    response = {
        "filename": original_filename,
        "saved_filename": saved_name,
        "num_rows": len(df),
        "num_predictions": len(predictions),
        "predictions": predictions[:20],  # 일부만 반환
        "rmse": rmse,
        "rmse_threshold": RMSE_THRESHOLD,
        "retraining_triggered": retraining_triggered,
        "retraining_result": retraining_result,
        "message": eval_result.get("message"),
    }

    return response