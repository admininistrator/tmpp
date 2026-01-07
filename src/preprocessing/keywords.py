# Tên file: src/preprocessing/keywords.py
import pandas as pd
import re
import os
from typing import List

def normalize_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch từ khóa để phục vụ phân loại.

        Trả về DataFrame với cột `cleaned_keyword` đã chuẩn hóa và lọc các giá trị
        rỗng/không hợp lệ.
        """

    df = df.copy()
    df["is_empty"] = df["Keyword"].isna() | (df["Keyword"].astype(str).str.strip() == "")

    # Chuẩn hóa keyword (không thay thế bản gốc)
    df["cleaned_keyword"] = (
        df["Keyword"]
        .astype(str)
        .str.lower()
        .str.strip()
        .apply(lambda x: re.sub(r"[^a-z0-9]+", "", x))
    )

    # Cờ từ khóa invalid sau khi làm sạch → chuỗi rỗng
    df["is_invalid"] = df["cleaned_keyword"] == ""

    # Cờ từ khóa quá ngắn (<3)
    df["is_short"] = df["cleaned_keyword"].str.len() < 3

    # Cờ từ khóa thuộc danh sách loại trừ
    exclude = ["amp", "lt", "gt"]
    df["is_excluded"] = df["cleaned_keyword"].isin(exclude)

    # Không xoá dữ liệu — chỉ gắn cờ
    # Nhưng vẫn giữ cleaned_keyword để dùng cho NLP
    valid_mask = (~df["is_empty"]) & (~df["is_invalid"]) & (~df["is_short"]) & (~df["is_excluded"])
    return df.loc[valid_mask, ["cleaned_keyword"]].rename(columns={"cleaned_keyword": "Keyword"})


def _split_keywords(keywords: List[str], symptom_reference: List[str]) -> pd.DataFrame:
    """Tách danh sách keyword thành 2 cột symptom_keyword/other_keyword.

    - Các keyword khớp `symptom_reference` được đưa vào cột `symptom_keyword`.
    - Phần còn lại đưa vào cột `other_keyword`.
    """

    symptom_set = set(symptom_reference)
    symptom_keywords = sorted({kw for kw in keywords if kw in symptom_set})
    other_keywords = sorted({kw for kw in keywords if kw not in symptom_set})

    max_len = max(len(symptom_keywords), len(other_keywords)) or 0
    symptom_col = symptom_keywords + [pd.NA] * (max_len - len(symptom_keywords))
    other_col = other_keywords + [pd.NA] * (max_len - len(other_keywords))

    return pd.DataFrame({
        "symptom_keyword": symptom_col,
        "other_keyword": other_col
    })


def run_cleaning(raw_path, processed_path):
    # Đọc file từ 'raw'
    try:
        df = pd.read_csv(os.path.join(raw_path, "keywords.csv"), header=None, names=["ID", "Keyword"])
        print("Successfully loaded keywords")
    except FileNotFoundError:
        print("    LỖI: Không tìm thấy 'keywords.csv' trong 'raw'")
        return
    except Exception as e:
        print(f"    LỖI khi đọc 'keywords.csv': {e}")
        return

    normalized_df = normalize_keywords(df)

    # Danh sách symptom gốc từ yêu cầu bài toán
    symptom_reference = [
        "stomach", "flu", "fever", "caught", "bad", "medicine", "feel", "chills",
        "sick", "hospital"
    ]

    cleaned_keywords = normalized_df["Keyword"].dropna().astype(str).tolist()
    classified_df = _split_keywords(cleaned_keywords, symptom_reference)

    # Lưu file vào 'processed'
    os.makedirs(processed_path, exist_ok=True)
    classified_df.to_csv(
        os.path.join(processed_path, 'keywords.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    print("Successfully cleaned up - keywords")


# Block để chạy thử nghiệm độc lập
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
    PROCESSED_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed")

    print(f"Chạy test cleaning cho keywords...")
    print(f"Raw path: {RAW_PATH}")
    print(f"Processed path: {PROCESSED_PATH}")
    run_cleaning(RAW_PATH, PROCESSED_PATH)