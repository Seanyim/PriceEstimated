"""
分组管理优化 v2.2 — 自动化测试
测试 detect_region_from_ticker / get_companies_in_category / get_companies_not_in_category
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.core.db import (
    detect_region_from_ticker,
    init_db, save_company_meta, auto_assign_company_to_region_category,
    get_companies_in_category, get_companies_not_in_category,
    get_all_categories, add_company_to_category, remove_company_from_category,
    delete_company
)

# ===== test_detect_region_from_ticker =====

def test_detect_us():
    assert detect_region_from_ticker("AAPL") == "US"
    assert detect_region_from_ticker("MSFT") == "US"
    assert detect_region_from_ticker("GOOG") == "US"

def test_detect_cn():
    assert detect_region_from_ticker("600519.SS") == "CN"
    assert detect_region_from_ticker("000858.SZ") == "CN"

def test_detect_hk():
    assert detect_region_from_ticker("9988.HK") == "HK"
    assert detect_region_from_ticker("0700.HK") == "HK"

def test_detect_jp():
    assert detect_region_from_ticker("6758.T") == "JP"

def test_detect_tw():
    assert detect_region_from_ticker("2330.TW") == "TW"

def test_detect_case_insensitive():
    assert detect_region_from_ticker("600519.ss") == "CN"
    assert detect_region_from_ticker("9988.hk") == "HK"

def test_detect_empty_and_whitespace():
    assert detect_region_from_ticker("") == "US"
    assert detect_region_from_ticker("  AAPL  ") == "US"


# ===== test_get_companies_in/not_in_category =====

import sqlite3
import tempfile
import modules.core.db as db_module

def _setup_test_db():
    """创建临时测试数据库"""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    # 替换 DB_PATH 为临时路径
    db_module.DB_PATH = tmp.name
    db_module.DB_DIR = os.path.dirname(tmp.name)
    init_db()
    return tmp.name

def _cleanup_test_db(path):
    try:
        os.unlink(path)
    except:
        pass

def test_get_companies_in_category():
    db_path = _setup_test_db()
    try:
        # 添加测试公司
        save_company_meta("AAPL", "Apple", "Billion", "US")
        save_company_meta("MSFT", "Microsoft", "Billion", "US")
        auto_assign_company_to_region_category("AAPL", "US")
        auto_assign_company_to_region_category("MSFT", "US")
        
        # 找到美股分组 ID
        cats = get_all_categories()
        us_cat = next((c for c in cats if "美股" in c["name"]), None)
        assert us_cat is not None, "应存在美股分组"
        
        members = get_companies_in_category(us_cat["id"])
        tickers = [m["ticker"] for m in members]
        assert "AAPL" in tickers
        assert "MSFT" in tickers
    finally:
        _cleanup_test_db(db_path)

def test_get_companies_not_in_category():
    db_path = _setup_test_db()
    try:
        # 添加测试公司
        save_company_meta("AAPL", "Apple", "Billion", "US")
        save_company_meta("MSFT", "Microsoft", "Billion", "US")
        save_company_meta("9988.HK", "Alibaba", "Billion", "HK")
        
        # 只将 AAPL 分配到美股
        auto_assign_company_to_region_category("AAPL", "US")
        auto_assign_company_to_region_category("9988.HK", "HK")
        
        cats = get_all_categories()
        us_cat = next((c for c in cats if "美股" in c["name"]), None)
        assert us_cat is not None
        
        # MSFT 未分配到美股（只用了 auto_assign 但 MSFT 未调用），9988.HK 在港股
        not_in = get_companies_not_in_category(us_cat["id"])
        not_in_tickers = [m["ticker"] for m in not_in]
        assert "9988.HK" in not_in_tickers
        assert "AAPL" not in not_in_tickers
    finally:
        _cleanup_test_db(db_path)

def test_smoke():
    assert True
