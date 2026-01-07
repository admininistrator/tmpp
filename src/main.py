import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Lấy đường dẫn tuyệt đối của file này (src/main.py)
current_file_path = os.path.abspath(__file__)

# Lấy thư mục cha của src (tức là Project Root)
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Thêm project_root vào sys.path để Python tìm được các module
sys.path.append(project_root)

# Import các module nội bộ
from src.utils.config_loader import load_config
from src.preprocessing import pipeline, compiler
from src.model import prophet_lstm_ensemble as model_pkg

# 1. MODULE HUẤN LUYỆN (TRAINING)

def run_training(config):
    print("\n--- BAT DAU HUAN LUYEN (TRAIN) ---")

    # 1. Chuẩn bị dữ liệu
    print("[1/3] Dang tong hop du lieu...")
    try:
        # Hàm compile_data lấy dữ liệu từ folder processed
        df_full = compiler.compile_data(config['data'])
    except Exception as e:
        print(f"Loi compile data: {e}")
        return

    # Tách dữ liệu: Cắt bỏ phần đuôi (tương lai giả định) để train
    steps = config['model']['forecast_steps']
    train_df = df_full.iloc[:-steps]

    print(f"   -> Du lieu train: {len(train_df)} dong (tu {train_df['ds'].min()} den {train_df['ds'].max()})")

    # 2. Train Models
    print("[2/3] Dang train Prophet va LSTM...")

    # Train Prophet
    prophet_model = model_pkg.train_prophet(train_df)

    # Train LSTM
    lstm_model, scaler = model_pkg.train_lstm(
        train_df,
        window=config['model']['look_back'],
        epochs=config['model']['lstm']['epochs']
    )

    # 3. Save Ensemble
    # Sử dụng project_root để đảm bảo đường dẫn đúng
    save_path = os.path.join(project_root, config['output']['model_save_path'])
    print(f"[3/3] Luu model vao: {save_path}")

    ensemble = model_pkg.EnsembleModels(prophet_model, lstm_model, scaler, None)
    model_pkg.save_ensemble(ensemble, save_path)

    print("Huan luyen hoan tat!")



# 2. MODULE DỰ BÁO (PREDICTION)
def run_prediction(config):
    print("\n--- BAT DAU DU BAO (PREDICT) ---")

    # Đường dẫn model (Tuyệt đối)
    save_path = os.path.join(project_root, config['output']['model_save_path'])

    # 1. Load Model
    print(f"[1/4] Dang tai Model tu: {save_path}")
    try:
        ensemble = model_pkg.load_ensemble(save_path)
    except Exception as e:
        print(f"Loi: {e}")
        print("Ban can chay lenh '--train' truoc de tao model!")
        return

    # 2. Load History Data
    print("[2/4] Dang lay du lieu lich su...")
    try:
        df_history = compiler.compile_data(config['data'])
    except Exception as e:
        print(f"Loi doc du lieu: {e}")
        return

    # 3. Predict
    steps = config['model']['forecast_steps']
    print(f"[3/4] Dang du bao {steps} gio tiep theo...")

    try:
        # Gọi hàm ensemble_predict
        forecast_df = model_pkg.ensemble_predict(
            prophet_model=ensemble.prophet_model,
            lstm_model=ensemble.lstm_model,
            scaler=ensemble.scaler,
            df_train=df_history,
            periods=steps,
            freq=config['model']['prophet']['freq'],
            window=config['model']['look_back'],
            weights_prophet=0.5
        )
    except AttributeError:
        print("Loi: Khong tim thay ham 'ensemble_predict'. Hay kiem tra file prophet_lstm_ensemble.py.")
        return

    # 4. Save & Plot
    print("[4/4] Luu ket qua va ve bieu do...")

    # --- FIX PATH: Lưu ra folder 'data/processed/forecasts' ở project root ---
    output_dir = os.path.join(project_root, "data", "processed", "forecasts")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "forecast_results.csv")
    forecast_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   File CSV: {csv_path}")

    figures_dir = os.path.join(project_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    last_hist = df_history.tail(100)
    plt.plot(last_hist['ds'], last_hist['y'], label='Thuc te (History)', color='black')
    plt.plot(forecast_df['ds'], forecast_df['y'], label='Du bao (Forecast)', color='red', linestyle='--')
    plt.title(f"Du bao xu huong ({steps} gio toi)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(figures_dir, "forecast_prediction.png")
    plt.savefig(plot_path)
    print(f"   Bieu do: {plot_path}")
    print("Du bao hoan tat!")

# 3. HÀM MAIN (ĐIỀU PHỐI CHUNG)

def main():
    # Thiết lập bộ đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description="He thong Du bao Dich benh (All-in-One)")

    # Các cờ lệnh (Flags)
    parser.add_argument('--etl', action='store_true', help='Chay xu ly du lieu (ETL)')
    parser.add_argument('--train', action='store_true', help='Huan luyen mo hinh')
    parser.add_argument('--predict', action='store_true', help='Chay du bao tuong lai')
    parser.add_argument('--all', action='store_true', help='Chay TAT CA (ETL -> Train -> Predict)')

    args = parser.parse_args()

    # Load Config (Dùng đường dẫn tuyệt đối)
    config_path = os.path.join(project_root, "configs", "settings.yaml")
    if not os.path.exists(config_path):
        print(f"Khong tim thay file cau hinh tai: {config_path}")
        return

    print(f"Loading config...")
    config = load_config(config_path)

    # --- GIAI ĐOẠN 1: ETL (Xử lý dữ liệu) ---
    # Đường dẫn file dữ liệu đã xử lý
    processed_path = os.path.join(project_root, config['data']['processed_path'])

    # Nếu chọn --etl, --all HOẶC file chưa tồn tại thì chạy ETL
    if args.etl or args.all or not os.path.exists(processed_path):
        print("\n--- KICH HOAT ETL ---")
        if not os.path.exists(processed_path):
            print("Chua tim thay du lieu sach. Tu dong chay ETL.")
        pipeline.run_pipeline(config)
    else:
        print("Du lieu da san sang. Bo qua ETL.")

    # --- GIAI ĐOẠN 2: TRAINING ---
    if args.train or args.all:
        run_training(config)

    # --- GIAI ĐOẠN 3: PREDICTION ---
    if args.predict or args.all:
        run_prediction(config)

    # Nếu người dùng chạy file mà không thêm cờ gì cả
    if not any([args.etl, args.train, args.predict, args.all]):
        print("\nCANH BAO: Ban chua chon che do chay!")
        print("------------------------------------------------")
        print("De du bao (Recommended):   python -m src.main --predict")
        print("De huan luyen lai model:   python -m src.main --train")
        print("De chay tat ca tu dau:     python -m src.main --all")


if __name__ == "__main__":
    main()