import pytest
import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import hàm nội bộ (cần test hàm ẩn _find_word)
from src.preprocessing import compiler


def test_find_word_logic():
    """Test hàm tìm từ khóa trong câu """
    text = "I feel sick and have a fever"
    keywords = {"sick", "fever", "flu"}

    found = compiler._find_word(text, keywords)

    assert "sick" in found
    assert "fever" in found
    assert "flu" not in found
    assert isinstance(found, list)


def test_find_word_empty():
    """Test với text rỗng hoặc None"""
    keywords = {"sick"}
    assert compiler._find_word("", keywords) == []
    assert compiler._find_word(None, keywords) == []


def test_merge_weather_logic():
    """Test logic ghép thời tiết vào bài post (theo thời gian)"""
    #  Tạo data giả: Post lúc 10:05
    microblogs = pd.DataFrame({
        'Created_at': ['2024-01-01 10:05:00'],
        'text': ['Sick']
    })

    # Tạo data thời tiết: Đo lúc 10:00 và 11:00
    weather = pd.DataFrame({
        'Date': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'Weather': ['Sunny', 'Rainy'],
        'Average_Wind_Speed': [5, 10],
        'Wind_Direction': ['NE', 'SW']
    })

    # Merge
    result = compiler.merge_weather_data(microblogs, weather)

    # Kiểm tra: merge_asof backward nên 10:05 phải lấy thời tiết lúc 10:00
    assert not result.empty
    assert result.iloc[0]['Weather'] == 'Sunny'
    assert result.iloc[0]['Average_Wind_Speed'] == 5