import sqlite3
import pandas as pd
import os
import streamlit as st
from modules.config import FINANCIAL_METRICS

DB_DIR = "data"
DB_FILE = "financial_data.db"
DB_PATH = os.path.join(DB_DIR, DB_FILE)

def init_db():
    """初始化数据库：创建独立的财务表和市场表"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # 1. 公司基础信息表
        c.execute('''CREATE TABLE IF NOT EXISTS companies (
                        ticker TEXT PRIMARY KEY,
                        name TEXT,
                        unit TEXT DEFAULT 'Billion',
                        last_market_cap REAL,
                        last_eps_ttm REAL,
                        last_update TEXT
                    )''')
        
        # 2. 财务数据表 (手动录入)
        metric_cols = [f"{m['id']} REAL" for m in FINANCIAL_METRICS]
        cols_sql = ", ".join(metric_cols)
        
        c.execute(f'''CREATE TABLE IF NOT EXISTS financial_records (
                        ticker TEXT,
                        year INTEGER,
                        period TEXT,
                        report_date TEXT,
                        {cols_sql},
                        PRIMARY KEY (ticker, year, period)
                    )''')

        # 3. [升级] 市场行情表 (增加市值、PE等字段)
        # 注意：如果表已存在但缺列，SQLite 不会自动加列。
        # 生产环境通常需要 Migration，这里我们用简单的 "CREATE IF NOT EXISTS"
        # 如果您之前已经运行过，建议删除旧 db 文件重建，或者手动 alter table。
        c.execute('''CREATE TABLE IF NOT EXISTS market_daily (
                        ticker TEXT,
                        date TEXT,
                        close REAL,
                        volume REAL,
                        market_cap REAL,
                        pe_ttm REAL,
                        pe_static REAL,
                        eps_ttm REAL,
                        PRIMARY KEY (ticker, date)
                    )''')
        
        conn.commit()
    except Exception as e:
        st.error(f"DB Init Error: {e}")
    finally:
        conn.close()

# --- 财务数据操作 ---

def get_financial_records(ticker):
    """获取某公司的所有财务记录"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # 按发布日期排序，这对每日 PE 计算至关重要
        query = "SELECT * FROM financial_records WHERE ticker = ? ORDER BY report_date ASC"
        df = pd.read_sql(query, conn, params=(ticker,))
        return df.to_dict('records')
    except:
        return []
    finally:
        conn.close()

def save_financial_record(record):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    clean_record = {k: v for k, v in record.items() if v is not None}
    cols = ", ".join(clean_record.keys())
    placeholders = ", ".join(["?"] * len(clean_record))
    values = tuple(clean_record.values())
    sql = f"INSERT OR REPLACE INTO financial_records ({cols}) VALUES ({placeholders})"
    try:
        c.execute(sql, values)
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Save Error: {e}")
        return False
    finally:
        conn.close()

# --- [升级] 市场数据操作 ---

def save_market_history(ticker, df_history):
    """
    保存包含 PE/市值 的全量市场数据
    df_history 需包含: Close, Volume, market_cap, pe_ttm, pe_static, eps_ttm
    """
    if df_history.empty: return
    conn = sqlite3.connect(DB_PATH)
    
    data = []
    # 确保 DataFrame 有我们需要的列，没有则补 None
    req_cols = ['Close', 'Volume', 'market_cap', 'pe_ttm', 'pe_static', 'eps_ttm']
    for c in req_cols:
        if c not in df_history.columns:
            df_history[c] = None

    for date, row in df_history.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        data.append((
            ticker, 
            date_str, 
            row['Close'], 
            row['Volume'],
            row['market_cap'],
            row['pe_ttm'],
            row['pe_static'],
            row['eps_ttm']
        ))
        
    try:
        c = conn.cursor()
        c.executemany('''INSERT OR REPLACE INTO market_daily 
                         (ticker, date, close, volume, market_cap, pe_ttm, pe_static, eps_ttm) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', data)
        conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        conn.close()

def get_market_history(ticker):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM market_daily WHERE ticker = ? ORDER BY date ASC", 
                         conn, params=(ticker,), parse_dates=['date'])
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

# --- 公司元数据操作 ---
def update_company_snapshot(ticker, market_cap, eps_ttm):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""UPDATE companies 
                     SET last_market_cap = ?, last_eps_ttm = ?, last_update = date('now') 
                     WHERE ticker = ?""", (market_cap, eps_ttm, ticker))
        if c.rowcount == 0:
            c.execute("INSERT INTO companies (ticker, last_market_cap, last_eps_ttm) VALUES (?, ?, ?)", 
                      (ticker, market_cap, eps_ttm))
        conn.commit()
    finally:
        conn.close()

def get_company_meta(ticker):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM companies WHERE ticker = ?", (ticker,))
    row = c.fetchone()
    conn.close()
    if row:
        col_names = [d[0] for d in c.description]
        return dict(zip(col_names, row))
    return {}

def save_company_meta(ticker, name, unit):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO companies (ticker, name, unit) VALUES (?, ?, ?)
                 ON CONFLICT(ticker) DO UPDATE SET name=excluded.name, unit=excluded.unit""", 
              (ticker, name, unit))
    conn.commit()
    conn.close()

def get_all_tickers():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT ticker FROM companies")
    tickers = [row[0] for row in c.fetchall()]
    conn.close()
    return tickers