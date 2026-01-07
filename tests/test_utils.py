import pytest
import os
import sys
import yaml

# Setup đường dẫn
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config_loader import load_config


def test_load_valid_config(tmp_path):
    """Test đọc file config hợp lệ"""
    # Tạo file yaml giả trong thư mục tạm
    d = tmp_path / "settings.yaml"
    data = {"data": {"raw": "path/to/raw"}, "model": {"epochs": 10}}
    d.write_text(yaml.dump(data), encoding='utf-8')

    # Load thử
    config = load_config(str(d))

    # Kiểm tra
    assert config['data']['raw'] == "path/to/raw"
    assert config['model']['epochs'] == 10


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("duong/dan/khong/ton/tai.yaml")


def test_load_malformed_yaml(tmp_path):
    d = tmp_path / "bad.yaml"
    d.write_text("data: [thieu ngoac dong", encoding='utf-8')

    with pytest.raises(yaml.YAMLError):
        load_config(str(d))