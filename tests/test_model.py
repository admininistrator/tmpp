import pytest
import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


try:
    from src.model import prophet_lstm_ensemble as model_pkg
except ImportError:
    # Fallback nếu chạy pytest trực tiếp
    from src.model import prophet_lstm_ensemble as model_pkg


# Tạo dữ liệu giả
@pytest.fixture
def mock_data():
    """Tạo ra 100 dòng dữ liệu giả lập (thời gian + số lượng)"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    # Random số lượng bệnh nhân từ 10 đến 50
    values = np.random.randint(10, 50, size=100)

    df = pd.DataFrame({'ds': dates, 'y': values})
    return df


def test_train_prophet(mock_data):
    print("\nTesting Prophet Training...")
    model = model_pkg.train_prophet(mock_data)

    # Kiểm tra object model có được tạo ra không
    assert model is not None
    # Kiểm tra xem model đã học chưa
    assert hasattr(model, 'history')


def test_train_lstm(mock_data):
    print("\nTesting LSTM Training...")
    model, scaler = model_pkg.train_lstm(mock_data, window=12, epochs=1)

    assert model is not None
    assert scaler is not None
    # Kiểm tra model có layer không
    assert len(model.layers) > 0


def test_ensemble_flow(mock_data):
    print("\nTesting Ensemble Flow...")

    p_model = model_pkg.train_prophet(mock_data)
    l_model, scaler = model_pkg.train_lstm(mock_data, window=12, epochs=1)

    # Dự báo thử 48 giờ tới
    future_steps = 48
    forecast = model_pkg.ensemble_predict(
        prophet_model=p_model,
        lstm_model=l_model,
        scaler=scaler,
        df_train=mock_data,
        periods=future_steps,
        freq='H',
        window=12,
        weights_prophet=0.5
    )

    #  Kiểm tra kết quả
    assert not forecast.empty, "Kết quả dự báo không được rỗng"
    assert len(forecast) == future_steps, f"Phải dự báo đủ {future_steps} bước"
    assert 'ds' in forecast.columns
    assert 'y' in forecast.columns
    # Kiểm tra không có giá trị NaN (dự báo phải có số)
    assert not forecast['y'].isnull().any()