# File: src/preprocessing/compiler.py
import pandas as pd
import numpy as np
import os
import re
from datetime import timedelta

def _find_word(text, keywords_set):
    found = []
    if not isinstance(text, str):
        return found
    words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
    found_keywords = words_in_text.intersection(keywords_set)
    return list(found_keywords)


def _to_file(df, folder_path, file_name):
    full_path = os.path.join(folder_path, file_name)
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')


def merge_weather_data(microblogs, weather):
    # ... (Giữ nguyên logic merge của bạn) ...
    weather_columns = ['Weather', 'Average_Wind_Speed', 'Wind_Direction']
    weather_df = weather.copy()
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    weather_df = weather_df.sort_values('Date').set_index('Date')
    # Fix nhỏ: dùng 'T' thay vì mặc định để tránh warning ở pandas mới
    weather_upsampled = weather_df.resample('T').asfreq()
    weather_upsampled[weather_columns] = weather_upsampled[weather_columns].ffill()
    weather_upsampled = weather_upsampled.dropna(subset=weather_columns)
    weather_upsampled = weather_upsampled.reset_index()

    microblogs_df = microblogs.copy()
    microblogs_df['Date'] = pd.to_datetime(microblogs_df['Created_at'])
    microblogs_df = microblogs_df.sort_values('Date')
    microblogs_df = microblogs_df.dropna(subset=['Date'])

    combined_df = pd.merge_asof(
        microblogs_df, weather_upsampled, on='Date', direction='backward'
    )
    # Logic lọc cột trùng
    original_cols = list(microblogs.columns)
    final_cols = original_cols + weather_columns
    final_df_cols_unique = []
    for col in final_cols:
        if col not in final_df_cols_unique: final_df_cols_unique.append(col)
    final_cols_exist = [col for col in final_df_cols_unique if col in combined_df.columns]
    return combined_df[final_cols_exist]


def analyze_keyword_stats(microblogs_with_weather, keywords_df):
    # Lấy danh sách keyword từ file processed/keywords.csv (2 cột symptom/other)
    symptom_list = keywords_df['symptom_keyword'].dropna().astype(str).tolist()
    other_keywords = keywords_df['other_keyword'].dropna().astype(str).tolist()

    all_keywords_set = set(symptom_list + other_keywords)

    keyword_df = pd.DataFrame({
        'Keyword': symptom_list + other_keywords,
        'keyword_type': ['symptom'] * len(symptom_list) + ['other'] * len(other_keywords)
    })
    keyword_to_type_map = pd.Series(keyword_df.keyword_type.values, index=keyword_df.Keyword).to_dict()

    stats_df = microblogs_with_weather.copy()
    stats_df['Created_at'] = pd.to_datetime(stats_df['Created_at'])
    stats_df['hour'] = stats_df['Created_at'].dt.hour
    stats_df['date'] = stats_df['Created_at'].dt.date
    stats_df['keyword_founds'] = stats_df['text'].apply(lambda x: _find_word(x, all_keywords_set))

    df_exploded = stats_df.explode('keyword_founds').dropna(subset=['keyword_founds'])
    df_exploded = df_exploded.rename(columns={'keyword_founds': 'keyword'})
    df_exploded['keyword_type'] = df_exploded['keyword'].map(keyword_to_type_map)

    keyword_counts = df_exploded.groupby(['date', 'hour', 'keyword']).size().reset_index(name='count')

    safe_mode = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    weather_stats = df_exploded.groupby(['date', 'hour', 'keyword']).agg(
        weather=('Weather', safe_mode),
        wind_direction=('Wind_Direction', safe_mode)
    ).reset_index()

    final_stats = pd.merge(keyword_counts, weather_stats, on=['date', 'hour', 'keyword'])
    return final_stats, keyword_df, df_exploded


def map_keyword_hourly_locations(df_exploded):
    return df_exploded[['date', 'hour', 'keyword', 'Latitude', 'Longitude']]


def _save_hourly_files(df_in, folder_path, file_name_base, logic_type):
    # ... (Giữ nguyên logic save hourly của bạn) ...
    grouped_by_time = df_in.groupby(['date', 'hour'])
    for (date_obj, hour), group_df in grouped_by_time:
        HH = str(hour).zfill(2)
        date_obj = pd.to_datetime(date_obj)
        DD_MM = date_obj.strftime('%d_%m')
        file_name_out = f"{file_name_base}_{HH}_{DD_MM}.csv"
        output_path = os.path.join(folder_path, file_name_out)

        if logic_type == 'stats':
            output_df = group_df[['keyword', 'count', 'weather', 'wind_direction']]
        elif logic_type == 'location_mapping':
            output_df = group_df[['Latitude', 'Longitude', 'keyword']].rename(
                columns={'Latitude': 'location_lat', 'Longitude': 'location_lon'})
        else:
            output_df = group_df
        os.makedirs(folder_path, exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')


def save_analysis_results(keywords, final_stats, location_stats, path):
    _to_file(keywords, path, 'keywords_classified.csv')
    _to_file(final_stats, path, 'final_stats.csv')
    _to_file(location_stats, path, 'location_stats.csv')
    stat_path = os.path.join(path, "stat_hourly")
    location_path = os.path.join(path, "keyword_location_mapping_hourly")
    _save_hourly_files(final_stats, stat_path, 'stat_hourly', logic_type='stats')
    _save_hourly_files(location_stats, location_path, 'keyword_location_mapping_hourly', logic_type='location_mapping')


# ... (Giữ nguyên toàn bộ code cũ từ đầu file đến hết hàm save_analysis_results) ...

# ============================================================
# PHẦN THÊM MỚI: CẦU NỐI CHO MODEL (QUAN TRỌNG)
# ============================================================

def compile_data(config_data: dict) -> pd.DataFrame:
    """
    Hàm chuẩn bị dữ liệu cho Model Training.
    Đọc final_stats.csv -> Lọc từ khóa bệnh -> Gộp theo giờ -> Trả về (ds, y)
    """
    processed_file = config_data.get('final_stats')

    # Kiểm tra file tồn tại
    if not os.path.exists(processed_file):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {processed_file}.\nHãy chạy lệnh: python -m src.main --etl")

    print(f"Loading data for model from: {processed_file}")
    df = pd.read_csv(processed_file)

    # 1. Lọc từ khóa mục tiêu (chỉ lấy bệnh, bỏ rác)
    target_keywords = config_data.get('target_keywords', [])
    if target_keywords:
        print(f"   -> Filtering {len(target_keywords)} target keywords...")
        df = df[df['keyword'].isin(target_keywords)]
    else:
        print("Cảnh báo: Không có target_keywords trong settings.yaml. Sẽ lấy tất cả từ khóa!")

    # 2. Tạo cột thời gian 'ds' (Date + Hour)
    # final_stats.csv có cột 'date' (str) và 'hour' (int)
    print("   -> Aggregating time series...")
    df['ds'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

    # 3. Tính tổng số lượng (y) của TẤT CẢ các bệnh trong giờ đó
    # Group theo giờ -> Tính tổng count
    df_agg = df.groupby('ds')['count'].sum().reset_index()
    df_agg = df_agg.rename(columns={'count': 'y'})

    # 4. Fill những giờ thiếu bằng 0 (để chuỗi thời gian liên tục)
    df_agg = df_agg.sort_values('ds').set_index('ds')
    df_agg = df_agg.asfreq('h').fillna(0).reset_index()

    print(f"Data compiled! Shape: {df_agg.shape}")
    print(df_agg.tail())  # In thử vài dòng cuối

    return df_agg