import json
import os

DATA_FILE = 'financial_data.json'

def load_data():
    """加载本地 JSON 数据，如果不存在则返回空字典"""
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_data(data):
    """保存数据到本地 JSON"""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)