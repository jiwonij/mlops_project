import os
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

import torch

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import prepare_data


# =========================
# seed 고정
# =========================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# =========================
# sequence 생성
# =========================
def create_sequences(data, feature_cols, target_col, input_len=24, output_len=24):
    X, y, time_list = [], [], []

    for _, group in data.groupby("building_number"):
        group = group.sort_values("date_time").reset_index(drop=True)

        feature_array = group[feature_cols].values
        target_array = group[target_col].values
        time_array = group["date_time"].values

        for i in range(len(group) - input_len - output_len + 1):
            X.append(feature_array[i:i + input_len])
            y.append(target_array[i + input_len:i + input_len + output_len])
            time_list.append(time_array[i + input_len:i + input_len + output_len])

    return np.array(X), np.array(y), np.array(time_list)


# =========================
# main 학습 함수
# =========================
def main():
    seed_everything(42)

    base_dir = Path.cwd()
    dataset_dir = base_dir / "dataset"
    artifact_dir = base_dir / "artifacts"
    plot_dir = artifact_dir / "plots"

    train_path = dataset_dir / "train.csv"
    building_info_path = dataset_dir / "building_info.csv"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 준비
    df = prepare_data(train_path, building_info_path)

    # label encoding
    label_encoder = LabelEncoder()
    df["building_type_encoded"] = label_encoder.fit_transform(df["building_type"])

    input_len = 24
    output_len = 12
    target_col = "power_consumption"

    feature_cols = [
        "building_type_encoded",
        "total_area",
        "cooling_area",
        "temperature",
        "humidity",
        "windspeed",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "lag_1",
        "lag_24",
        "diff_1",
        "diff_24",
        "roll_mean_24",
        "roll_std_24",
    ]

    # train/valid split
    train_parts, valid_parts = [], []

    for _, group in df.groupby("building_number"):
        group = group.sort_values("date_time").reset_index(drop=True)
        split_idx = int(len(group) * 0.8)
        train_parts.append(group.iloc[:split_idx])
        valid_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    valid_df = pd.concat(valid_parts).reset_index(drop=True)

    # scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
    valid_df[feature_cols] = feature_scaler.transform(valid_df[feature_cols])

    train_df[[target_col]] = target_scaler.fit_transform(train_df[[target_col]])
    valid_df[[target_col]] = target_scaler.transform(valid_df[[target_col]])

    # sequence 생성
    X_train, y_train, _ = create_sequences(train_df, feature_cols, target_col, input_len, output_len)
    X_valid, y_valid, time_valid = create_sequences(valid_df, feature_cols, target_col, input_len, output_len)

    # 모델 정의
    model = Sequential([
        Input(shape=(input_len, len(feature_cols))),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(32, activation="relu"),
        Dense(output_len),
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # 학습
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=30,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1,
    )

    # 예측
    y_pred = model.predict(X_valid, verbose=0)

    y_valid_inv = target_scaler.inverse_transform(
        y_valid.reshape(-1, 1)
    ).reshape(y_valid.shape)

    y_pred_inv = target_scaler.inverse_transform(
        y_pred.reshape(-1, 1)
    ).reshape(y_pred.shape)

    rmse = float(np.sqrt(mean_squared_error(y_valid_inv.flatten(), y_pred_inv.flatten())))
    print(f"Integrated Model RMSE: {rmse:.4f}")

    # loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig(plot_dir / "loss_curve.png")
    plt.close()

    # 모델 저장
    model.save(artifact_dir / "model.keras")

    with open(artifact_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(artifact_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    with open(artifact_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "input_len": input_len,
        "output_len": output_len,
        "rmse": rmse,
    }

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved artifacts to artifacts/")


if __name__ == "__main__":
    main()