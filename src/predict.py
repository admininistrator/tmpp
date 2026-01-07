import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup đường dẫn để import được module trong src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.preprocessing import compiler
from src.model import prophet_lstm_ensemble as model_pkg


def main():
    print("BẮT ĐẦU QUÁ TRÌNH DỰ BÁO...")

    # --- 1. LOAD CONFIG ---
    print("[1/5] Đang đọc cấu hình...")
    config = load_config("../configs/settings.yaml")

    # --- 2. LOAD MODEL ---
    save_path = config['output']['model_save_path']
    print(f"[2/5] Đang tải Model từ: {save_path}")
    try:
        ensemble = model_pkg.load_ensemble(save_path)
        print("Đã load thành công: Prophet, LSTM, Scaler.")
    except Exception as e:
        print(f"Lỗi load model: {e}")
        print("Hãy chạy 'python -m src.main' để train model trước!")
        return

    # --- 3. CHUẨN BỊ DỮ LIỆU LỊCH SỬ ---
    # LSTM cần nhìn thấy quá khứ để dự đoán tương lai
    print("[3/5] Đang đọc dữ liệu lịch sử...")
    try:
        df_history = compiler.compile_data(config['data'])
    except Exception as e:
        print(f"Lỗi đọc dữ liệu: {e}")
        return

    # --- 4. THỰC HIỆN DỰ BÁO ---
    steps = config['model']['forecast_steps']
    print(f"[4/5] Đang chạy mô hình dự báo cho {steps} giờ tới...")

    forecast_df = model_pkg.ensemble_predict(
        prophet_model=ensemble.prophet_model,
        lstm_model=ensemble.lstm_model,
        scaler=ensemble.scaler,
        df_train=df_history,  # Dữ liệu quá khứ
        periods=steps,  # Số bước dự báo (lấy từ config)
        freq=config['model']['prophet']['freq'],
        window=config['model']['look_back'],
        weights_prophet=0.5  # Tỷ trọng 50-50 (có thể chỉnh sửa)
    )

    # --- 5. LƯU KẾT QUẢ & VẼ BIỂU ĐỒ ---
    print("[5/5] Lưu kết quả theo cấu trúc thư mục...")

    # A. Lưu file CSV kết quả (Vào data/processed/forecasts)
    """output_data_dir = os.path.join("data", "processed", "forecasts")
    os.makedirs(output_data_dir, exist_ok=True)

    csv_path = os.path.join(output_data_dir, "forecast_results.csv")
    forecast_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"File số liệu: {csv_path}")"""

    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(src_dir)

    output_data_dir = os.path.join(project_root, "data", "processed", "forecasts")
    os.makedirs(output_data_dir, exist_ok=True)
    csv_path = os.path.join(output_data_dir, "forecast_results.csv")
    forecast_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"File số liệu: {csv_path}")

    # B. Lưu biểu đồ (Vào figures/) - ĐÚNG CẤU TRÚC
    figures_dir = "figures"
    output_figures_dir = os.path.join(project_root, "figures")
    os.makedirs(output_figures_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Vẽ 100 giờ cuối của lịch sử
    last_history = df_history.tail(100)
    plt.plot(last_history['ds'], last_history['y'], label='Thực tế (Quá khứ)', color='black')

    # Vẽ đường dự báo
    plt.plot(forecast_df['ds'], forecast_df['y'], label='Dự báo (Tương lai)', color='red', linestyle='--')

    plt.title(f"Dự báo Dịch bệnh ({steps} giờ tới)", fontsize=14)
    plt.xlabel("Thời gian")
    plt.ylabel("Số lượng ca")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Lưu ảnh
    plot_path = os.path.join(output_figures_dir, "forecast_prediction.png")
    plt.savefig(plot_path)
    print(f"Biểu đồ: {plot_path}")

    print("\nHOÀN TẤT! Hãy kiểm tra thư mục 'figures' và 'data/processed/forecasts'.")


if __name__ == "__main__":
    main()