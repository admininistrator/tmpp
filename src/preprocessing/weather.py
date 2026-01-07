import pandas as pd
import os


def run_cleaning(raw_path, processed_path):

    try:
        df = pd.read_csv(os.path.join(raw_path, 'Weather.csv'))
        print("Successfully loaded Weather.csv")
    except FileNotFoundError:
        print("    LỖI: Không tìm thấy 'Weather.csv' trong 'raw'")
        return
    except Exception as e:
        print(f"    LỖI khi đọc 'Weather.csv': {e}")
        return

    # =============== FLAG DATE ===============
    df["date_invalid"] = False

    df["Parsed_Date"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y",
        errors="coerce"
    )

    # =============== FILTER DATE (day 8-11) ===============
    # Giữ lại các dòng có Parsed_Date hợp lệ và ngày trong tháng từ 8 đến 11
    df = df[df["Parsed_Date"].notna() & df["Parsed_Date"].dt.day.between(8, 11)].copy()

    df.loc[df["Parsed_Date"].isna(), "date_invalid"] = True

    # =============== FLAG WEATHER ===============
    df["Weather"] = df["Weather"].astype(str).str.strip().str.upper()

    df["weather_empty"] = df["Weather"].str.strip() == ""
    df["weather_invalid"] = df["Weather"].str.contains(r"[^A-Z]", na=True)

    # =============== FLAG WIND SPEED ===============
    wind_numeric = pd.to_numeric(
        df["Average_Wind_Speed"],
        errors="coerce"
    )

    df["wind_speed_invalid"] = wind_numeric.isna()
    df["wind_speed_missing"] = df["Average_Wind_Speed"].isna()
    df["wind_speed_negative"] = wind_numeric < 0

    # Gắn lại numeric nhưng KHÔNG fillna
    df["Average_Wind_Speed"] = wind_numeric

    # =============== FLAG WIND DIRECTION ===============
    valid_dirs = {
        "N", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    }

    df["Wind_Direction"] = df["Wind_Direction"].astype(str).str.strip().str.upper()

    df["wind_dir_invalid"] = ~df["Wind_Direction"].isin(valid_dirs)

    # Lưu file vào 'processed'
    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(
        os.path.join(processed_path, 'weather.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    print("Successfully cleaned up - weather")

# Block để chạy thử nghiệm độc lập
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
    PROCESSED_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed")

    run_cleaning(RAW_PATH, PROCESSED_PATH)