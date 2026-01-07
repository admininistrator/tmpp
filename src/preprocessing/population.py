import pandas as pd
import os


def run_cleaning(raw_path, processed_path):

    # Đọc file từ 'raw'
    try:
        df = pd.read_csv(os.path.join(raw_path, 'Population.csv'))
        print("Successfully loaded population.csv")
    except FileNotFoundError:
        print("    LỖI: Không tìm thấy 'population.csv' trong 'raw'")
        return
    except Exception as e:
        print(f"    LỖI khi đọc 'population.csv': {e}")
        return

    # Chuẩn hóa tên vùng
    df['Zone_Name'] = df['Zone_Name'].str.strip().str.title()

    # === Gắn cờ ===

    # Cờ tên vùng bị trống
    df["zone_empty"] = df["Zone_Name"].str.strip() == ""

    # Cờ invalid cho Population_Density
    df["density_invalid"] = pd.to_numeric(df["Population_Density"], errors="coerce").isna()

    # Cờ missing
    df["density_missing"] = df["Population_Density"].isna()

    # Cờ âm
    df["density_negative"] = pd.to_numeric(df["Population_Density"], errors="coerce") < 0

    # Cờ invalid cho Daytime_Population
    df["daytime_pop_invalid"] = pd.to_numeric(df["Daytime_Population"], errors="coerce").isna()

    # Cờ missing
    df["daytime_missing"] = df["Daytime_Population"].isna()

    # Cờ âm
    df["daytime_negative"] = pd.to_numeric(df["Daytime_Population"], errors="coerce") < 0

    # Chuyển các cột sang numeric nhưng KHÔNG fillna
    df["Population_Density"] = pd.to_numeric(df["Population_Density"], errors="coerce")
    df["Daytime_Population"] = pd.to_numeric(df["Daytime_Population"], errors="coerce")

    # Lưu file vào 'processed'
    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(
        os.path.join(processed_path, 'population.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    print("Successfully cleaned up - population")

# Block để chạy thử nghiệm độc lập
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
    PROCESSED_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed")

    run_cleaning(RAW_PATH, PROCESSED_PATH)