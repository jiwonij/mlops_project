import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from preprocessing import prepare_data


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATASET_DIR = BASE_DIR / "dataset"


# =========================
# 1. artifact 로드
# =========================
def load_artifacts():
    model = load_model(ARTIFACT_DIR / "model.keras")

    with open(ARTIFACT_DIR / "feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)

    with open(ARTIFACT_DIR / "target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    with open(ARTIFACT_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, feature_scaler, target_scaler, label_encoder, metadata


# =========================
# 2. 시퀀스 생성
# =========================
def create_sequences(data, feature_cols, target_col, input_len, output_len):
    X = []
    pred_times = []
    actual_values = []

    for _, group in data.groupby("building_number"):
        group = group.sort_values("date_time").reset_index(drop=True)

        feature_array = group[feature_cols].values
        target_array = group[target_col].values
        time_array = group["date_time"].values

        for i in range(len(group) - input_len - output_len + 1):
            X.append(feature_array[i:i + input_len])
            pred_times.append(time_array[i + input_len:i + input_len + output_len])
            actual_values.append(target_array[i + input_len:i + input_len + output_len])

    if len(X) == 0:
        raise ValueError("시퀀스 생성 결과가 비어 있음")

    return np.array(X), np.array(pred_times), np.array(actual_values)


# =========================
# 3. 예측
# =========================
def predict(input_df: pd.DataFrame, building_info_df: pd.DataFrame):
    model, feature_scaler, target_scaler, label_encoder, metadata = load_artifacts()

    feature_cols = metadata["feature_cols"]
    target_col = metadata["target_col"]
    input_len = metadata["input_len"]
    output_len = metadata["output_len"]

    # 🔥 전처리 (train과 동일)
    df = prepare_data(input_df, building_info_df)

    # 🔥 label encoding (여기서만)
    unknown_types = sorted(set(df["building_type"].dropna().unique()) - set(label_encoder.classes_))
    if unknown_types:
        raise ValueError(f"학습에 없던 building_type 있음: {unknown_types}")

    df["building_type_encoded"] = label_encoder.transform(df["building_type"])

    # 🔥 feature 체크
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필요한 컬럼 없음: {missing_cols}")

    # 🔥 scaling
    scaled_df = df.copy()
    scaled_df[feature_cols] = feature_scaler.transform(scaled_df[feature_cols])
    scaled_df[[target_col]] = target_scaler.transform(scaled_df[[target_col]])

    # 🔥 sequence 생성
    X, pred_times, actual_values = create_sequences(
        scaled_df,
        feature_cols,
        target_col,
        input_len,
        output_len,
    )

    # 🔥 예측
    y_pred_scaled = model.predict(X, verbose=0)

    # 🔥 inverse transform
    y_pred = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(y_pred_scaled.shape)

    y_true = target_scaler.inverse_transform(
        actual_values.reshape(-1, 1)
    ).reshape(actual_values.shape)

    # 🔥 result_df 생성
    rows = []
    for i in range(len(pred_times)):
        for j in range(output_len):
            rows.append({
                "sample_idx": i,
                "forecast_step": j + 1,
                "date_time": pred_times[i][j],
                "actual": float(y_true[i][j]),
                "predicted": float(y_pred[i][j]),
            })

    result_df = pd.DataFrame(rows)

    return {
        "predictions": y_pred,
        "actuals": y_true,
        "result_df": result_df,
    }


# =========================
# 테스트 실행
# =========================
if __name__ == "__main__":
    result = predict(train_df, building_info_df)

    print(result["result_df"].head())
    print(result["result_df"].shape)