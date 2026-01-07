import pytest
import pandas as pd
import numpy as np
import os
import sys

# Setup đường dẫn để import được src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import prophet_lstm_ensemble as model_pkg


@pytest.fixture
def mock_data():
    """Tạo dữ liệu giả để train nhanh"""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
    values = np.random.randint(10, 50, size=50)
    return pd.DataFrame({'ds': dates, 'y': values})


def test_save_and_load_ensemble(mock_data, tmp_path):

    # Train Model
    print("\n[STEP 1] Dang train model gia lap...")
    p_model = model_pkg.train_prophet(mock_data)
    l_model, scaler = model_pkg.train_lstm(mock_data, epochs=1)

    ensemble = model_pkg.EnsembleModels(p_model, l_model, scaler, None)

    # Lưu Model
    # tmp_path là thư mục tạm do pytest tạo ra (tự xóa sau khi test xong)
    save_dir = tmp_path / "model_test_output"
    save_path_str = str(save_dir)

    print(f"[STEP 2] Dang luu model vao: {save_path_str}")
    model_pkg.save_ensemble(ensemble, save_path_str)

    # KIỂM TRA FILE
    if not os.path.exists(save_path_str):
        pytest.fail(f" Thư mục không được tạo ra: {save_path_str}")

    files_created = os.listdir(save_path_str)
    print(f"\n [DEBUG] Danh sach file thuc te trong thu muc:\n    {files_created}\n")

    # Kiểm tra Prophet (Chấp nhận .json HOẶC .pkl)
    has_prophet_json = "prophet.json" in files_created
    has_prophet_pkl = "prophet.pkl" in files_created

    if not (has_prophet_json or has_prophet_pkl):
        pytest.fail(f" Khong tim thay file Prophet (.json hoac .pkl). Chi co: {files_created}")

    # Kiểm tra LSTM (Chấp nhận .keras HOẶC .h5)
    has_lstm_keras = "lstm_model.keras" in files_created
    has_lstm_h5 = "lstm_model.h5" in files_created

    if not (has_lstm_keras or has_lstm_h5):
        pytest.fail(f" Khong tim thay file LSTM (.keras hoac .h5). Chi co: {files_created}")

    # Kiểm tra Scaler
    if "scaler.pkl" not in files_created:
        pytest.fail(" Khong tim thay file scaler.pkl")

    # Load lại và Predict
    print("[STEP 3] Load model len va du bao thu...")
    loaded_ensemble = model_pkg.load_ensemble(save_path_str)

    assert loaded_ensemble.prophet_model is not None
    assert loaded_ensemble.lstm_model is not None

    forecast = model_pkg.ensemble_predict(
        loaded_ensemble.prophet_model,
        loaded_ensemble.lstm_model,
        loaded_ensemble.scaler,
        df_train=mock_data,
        periods=12
    )
    assert len(forecast) == 12
    print(" Test quy trinh Save/Load thanh cong!")