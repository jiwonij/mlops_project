import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


DEFAULT_RMSE_THRESHOLD = 450.0


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    non_zero_mask = np.abs(y_true) > eps
    if non_zero_mask.sum() == 0:
        return float("nan")

    mape = np.mean(
        np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    ) * 100
    return float(mape)


def evaluate_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rmse_threshold: float = DEFAULT_RMSE_THRESHOLD,
) -> dict:
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError(
            f"y_true와 y_pred 길이가 다릅니다. "
            f"y_true={len(y_true_flat)}, y_pred={len(y_pred_flat)}"
        )

    rmse = calculate_rmse(y_true_flat, y_pred_flat)
    mae = calculate_mae(y_true_flat, y_pred_flat)
    mape = calculate_mape(y_true_flat, y_pred_flat)

    retrain_required = rmse > rmse_threshold

    if retrain_required:
        message = (
            f"RMSE가 기준을 초과했습니다. "
            f"(rmse={rmse:.4f}, threshold={rmse_threshold:.4f})"
        )
    else:
        message = (
            f"RMSE가 기준 이내입니다. "
            f"(rmse={rmse:.4f}, threshold={rmse_threshold:.4f})"
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "rmse_threshold": rmse_threshold,
        "retrain_required": retrain_required,
        "message": message,
    }


def evaluate_result_df(
    result_df: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "predicted",
    rmse_threshold: float = DEFAULT_RMSE_THRESHOLD,
) -> dict:
    if actual_col not in result_df.columns:
        raise ValueError(f"'{actual_col}' 컬럼이 result_df에 없습니다.")
    if pred_col not in result_df.columns:
        raise ValueError(f"'{pred_col}' 컬럼이 result_df에 없습니다.")

    y_true = result_df[actual_col].to_numpy()
    y_pred = result_df[pred_col].to_numpy()

    metrics = evaluate_arrays(
        y_true=y_true,
        y_pred=y_pred,
        rmse_threshold=rmse_threshold,
    )

    metrics["num_rows"] = int(len(result_df))
    return metrics