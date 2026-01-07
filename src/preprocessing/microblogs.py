import pandas as pd
import re
import html
import os


def run_cleaning(raw_path, processed_path):

    # Đọc file từ 'raw'
    try:
        df = pd.read_csv(os.path.join(raw_path, 'microblogs (8-11).csv'))
        print("Successfully loaded microblogs (8-11).csv")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy 'microblogs (8-11).csv' trong 'raw'")
        return

    # Tăng giới hạn hiển thị
    pd.set_option('display.max_rows', None)

    # Gắn cờ
    for col in df.columns:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Chuẩn hóa thời gian
    df['Created_at'] = pd.to_datetime(
        df['Created_at'],
        format='%m/%d/%Y %H:%M',
        errors='coerce'
    )

    # Tách tọa độ thành 2 cột
    df[['Latitude', 'Longitude']] = df['Location'].str.split(' ', expand=True)

    df.drop(columns=['Location'], inplace=True)

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # Làm sạch text kết hợp gắn cờ
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = html.unescape(text)
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^a-z0-9\s\<\>]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["text"] = df["text"].astype(str).apply(clean_text)

    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

    # Lưu file vào 'processed'
    os.makedirs(processed_path, exist_ok=True)  # Đảm bảo thư mục tồn tại
    df.to_csv(
        os.path.join(processed_path, 'microblogs.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    print("Successfully cleaned up - microblogs")

if __name__ == "__main__":
    # Tạo đường dẫn tương đối để tự test
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
    PROCESSED_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed")

    run_cleaning(RAW_PATH, PROCESSED_PATH)