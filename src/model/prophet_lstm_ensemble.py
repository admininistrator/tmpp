from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# Import trực tiếp các thư viện cần thiết
# (Giả định môi trường đã cài đủ theo environment.yml)
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, InputLayer


@dataclass
class EnsembleModels:
    prophet_model: object
    lstm_model: object
    scaler: object
    meta_model: Optional[object] = None


def prepare_series(df: pd.DataFrame, ds_col: str = "ds", y_col: str = "y") -> pd.DataFrame:
    """Chuẩn hóa DataFrame về dạng ds, y"""
    # Map tên cột nếu khác mặc định
    if ds_col not in df.columns or y_col not in df.columns:
        # Thử rename nếu cột ds, y chưa có nhưng cột date, count có (từ config)
        pass

    out = df.copy()
    # Đảm bảo có cột ds, y
    if ds_col != "ds":
        out = out.rename(columns={ds_col: "ds"})
    if y_col != "y":
        out = out.rename(columns={y_col: "y"})

    out["ds"] = pd.to_datetime(out["ds"])
    out = out.sort_values("ds").reset_index(drop=True)
    out["y"] = out["y"].astype(float)
    return out


def train_prophet(df: pd.DataFrame, **kwargs) -> object:
    """Huấn luyện Prophet"""
    # Chuẩn bị dữ liệu
    dfp = prepare_series(df)[["ds", "y"]]

    model = Prophet(**kwargs)
    model.fit(dfp, algorithm = 'lbfgs')
    return model


def _create_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(LSTM(50, activation="tanh"))  # Tăng unit lên 50 cho khớp config
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


def _series_to_supervised(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i: i + window])
        y.append(values[i + window])
    return np.array(X), np.array(y)


def train_lstm(
        df: pd.DataFrame,
        window: int = 24,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 1
) -> Tuple[object, object]:
    """
    Huấn luyện LSTM.
    Returns: (model, scaler) -> Chỉ trả về 2 giá trị để khớp với main.py
    """
    dfp = prepare_series(df)
    values = dfp["y"].values.reshape(-1, 1)

    # SỬA LỖI QUAN TRỌNG: Giữ lại object scaler để dùng sau này
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)  # Dữ liệu đã scale

    # Tạo dữ liệu supervised (X, y) từ dữ liệu đã scale
    # scaled_values là mảng 2 chiều (N, 1), cần flatten thành 1 chiều cho hàm _series
    X, y = _series_to_supervised(scaled_values.flatten(), window)

    if X.size == 0:
        raise ValueError(f"Không đủ dữ liệu ({len(values)}) cho cửa sổ window={window}")

    # Reshape X cho LSTM: [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = _create_lstm_model((window, 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False)

    return model, scaler


def prophet_forecast(prophet_model: object, periods: int, freq: str = "H") -> pd.DataFrame:
    future = prophet_model.make_future_dataframe(periods=periods, freq=freq)
    forecast = prophet_model.predict(future)
    return forecast[["ds", "yhat"]].rename(columns={"yhat": "y"})


def lstm_forecast(lstm_model: object,
                  scaler: object,
                  last_values: np.ndarray,  # Dữ liệu gốc chưa scale
                  periods: int,
                  window: int) -> np.ndarray:
    # Scale dữ liệu đầu vào
    last_values_scaled = scaler.transform(last_values.reshape(-1, 1)).flatten()

    seq = list(last_values_scaled)
    preds_scaled = []

    for _ in range(periods):
        # Lấy window phần tử cuối cùng
        x_input = np.array(seq[-window:]).reshape(1, window, 1)

        # Dự báo (kết quả trả về đang ở dạng scaled)
        yhat = lstm_model.predict(x_input, verbose=0)[0, 0]

        preds_scaled.append(yhat)
        seq.append(yhat)

    # Inverse transform để về giá trị thực
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_final = scaler.inverse_transform(preds_scaled).flatten()

    return preds_final


def ensemble_predict(
        prophet_model: object,
        lstm_model: object,
        scaler: object,
        df_train: pd.DataFrame,
        periods: int = 24,
        freq: str = "H",
        window: int = 24,
        weights_prophet: float = 0.5) -> pd.DataFrame:
    """
    Kết hợp kết quả dự báo của Prophet và LSTM theo trọng số.
    """
    if not (0.0 <= weights_prophet <= 1.0):
        raise ValueError(f"weights_prophet {weights_prophet} không hợp lệ (phải từ 0 đến 1)")

    weights_lstm = 1.0 - weights_prophet

    # 1. Dự báo Prophet
    prophet_fore = prophet_forecast(prophet_model, periods=periods, freq=freq)
    # Lấy đúng phần đuôi dự báo (tương lai)
    prophet_tail = prophet_fore.tail(periods).reset_index(drop=True)

    # 2. Dự báo LSTM
    # Lấy chuỗi dữ liệu cuối cùng từ lịch sử (df_train) để làm đầu vào
    dfp = prepare_series(df_train)
    last_vals = dfp["y"].values[-window:]  # Dữ liệu gốc chưa scale

    if len(last_vals) < window:
        raise ValueError(f"Dữ liệu lịch sử không đủ dài ({len(last_vals)}) so với window ({window})")

    lstm_preds = lstm_forecast(lstm_model, scaler, last_vals, periods=periods, window=window)

    # 3. Kết hợp (Weighted Average)
    result = prophet_tail.copy()
    result["y_lstm"] = lstm_preds
    result["y_prophet"] = result["y"]  # Lưu lại để so sánh nếu cần

    # Công thức Ensemble
    result["y"] = (weights_prophet * result["y_prophet"]) + (weights_lstm * result["y_lstm"])

    # Trả về bảng kết quả gồm: ds, y (tổng hợp), y_prophet, y_lstm
    return result[["ds", "y", "y_prophet", "y_lstm"]]

def save_ensemble(models: EnsembleModels, path: str) -> None:
    print(f"   -> Saving ensemble to {path}...")
    os.makedirs(path, exist_ok=True)

    # 1. Save Prophet
    with open(os.path.join(path, "prophet.pkl"), "wb") as f:
        pickle.dump(models.prophet_model, f)

    # 2. Save LSTM
    lstm_path = os.path.join(path, "lstm_model.keras")
    models.lstm_model.save(lstm_path)

    # 3. Save Scaler (SỬA LỖI: Phải dump models.scaler chứ không phải prophet_model)
    with open(os.path.join(path, "scaler.pkl"), "wb") as f:
        pickle.dump(models.scaler, f)

    # 4. Save Meta model (nếu có)
    if models.meta_model:
        with open(os.path.join(path, "meta_model.pkl"), "wb") as f:
            pickle.dump(models.meta_model, f)


def load_ensemble(path: str) -> EnsembleModels:
    # 1. Load Prophet
    with open(os.path.join(path, "prophet.pkl"), "rb") as f:
        prophet_model = pickle.load(f)

    # 2. Load LSTM
    lstm_path = os.path.join(path, "lstm_model.keras")
    if not os.path.exists(lstm_path):
        # Fallback cho định dạng cũ
        lstm_path = os.path.join(path, "lstm_model.h5")

    if os.path.exists(lstm_path):
        lstm_model = load_model(lstm_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy model LSTM tại {path}")

    # 3. Load Scaler
    with open(os.path.join(path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return EnsembleModels(prophet_model=prophet_model, lstm_model=lstm_model, scaler=scaler)


__all__ = [
    "train_prophet",
    "train_lstm",
    "save_ensemble",
    "load_ensemble",
    "EnsembleModels",
    "prophet_forecast",
    "lstm_forecast"
]