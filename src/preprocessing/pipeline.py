import os
import pandas as pd
import warnings

# Import module nội bộ
from . import keywords
from . import microblogs
from . import population
from . import weather
from . import compiler


def run_pipeline(config: dict):
    warnings.filterwarnings('ignore')

    # --- BẮT ĐẦU ĐOẠN SỬA ---
    # 1. Tính toán Project Root (Thư mục gốc dự án)
    # Lấy vị trí file pipeline.py -> lùi ra 3 cấp (src/preprocessing/pipeline.py -> Project Root)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    # 2. Tạo đường dẫn tuyệt đối cho Input (Raw)
    # Ghép Project Root với đường dẫn trong config
    RAW_PATH = os.path.join(project_root, config['data']['raw_path'])

    # 3. Tạo đường dẫn tuyệt đối cho Output (Processed)
    full_processed_path = os.path.join(project_root, config['data']['processed_path'])

    # Lấy thư mục cha (ví dụ: E:/.../data/processed/stat_hourly)
    PROCESSED_DIR = os.path.dirname(full_processed_path)
    # --- KẾT THÚC ĐOẠN SỬA ---

    print(f"BẮT ĐẦU ETL PIPELINE...")
    print(f"   Input Dir : {RAW_PATH}")
    print(f"   Output Dir: {PROCESSED_DIR}")

    # Tạo folder output nếu chưa có
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # BƯỚC 1: CLEAN
    print("[1/4] Running Cleaning scripts...")
    try:
        # Truyền đường dẫn tuyệt đối vào
        keywords.run_cleaning(RAW_PATH, PROCESSED_DIR)
        microblogs.run_cleaning(RAW_PATH, PROCESSED_DIR)
        population.run_cleaning(RAW_PATH, PROCESSED_DIR)
        weather.run_cleaning(RAW_PATH, PROCESSED_DIR)
    except Exception as e:
        print(f"LỖI Cleaning: {e}")
        return

    # BƯỚC 2: LOAD
    print("[2/4] Loading cleaned data...")
    try:
        mb_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'microblogs.csv'), encoding='latin-1')
        weather_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'weather.csv'))
        kw_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'keywords.csv'))
    except FileNotFoundError as e:
        print(f"LỖI Load: {e}")
        print(f" Hãy kiểm tra folder {PROCESSED_DIR} có file chưa?")
        return

    # BƯỚC 3: ANALYZE & COMPILE
    print("[3/4] Analyzing & Merging...")
    try:
        mb_with_weather = compiler.merge_weather_data(mb_df, weather_df)
        final_stats, keyword_df, df_exploded = compiler.analyze_keyword_stats(mb_with_weather, kw_df)
        location_stats = compiler.map_keyword_hourly_locations(df_exploded)
    except Exception as e:
        print(f"LỖI Logic Compiler: {e}")
        return

    # BƯỚC 4: SAVE
    print("[4/4] Saving Analysis Results...")
    # Lưu vào PROCESSED_DIR (đã là đường dẫn tuyệt đối)
    compiler.save_analysis_results(keyword_df, final_stats, location_stats, PROCESSED_DIR)

    print(f" ETL HOÀN TẤT! File kết quả tại: {PROCESSED_DIR}")