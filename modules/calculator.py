import pandas as pd
import numpy as np
from modules.config import GROWTH_METRIC_KEYS, ALL_METRIC_KEYS

# 数据库存储的周期映射 (新版 SQLite 使用小写 year/period)
PERIOD_SORT_MAP = {"Q1": 1, "H1": 2, "Q9": 3, "FY": 4}

def process_financial_data(df_raw):
    """
    核心计算引擎：将数据库中的累计数据 (Cumulative) 转换为单季度数据 (Single Quarter)
    输入: df_raw (包含 year, period, Revenue 等累计值)
    输出: df_cum (原始累计), df_single (计算后的单季 Q1-Q4)
    """
    if df_raw.empty:
        return df_raw, df_raw

    df = df_raw.copy()
    
    # 1. 基础清理与排序 (使用小写字段名适配 SQLite)
    if 'period' in df.columns:
        df['Sort_Key'] = df['period'].map(PERIOD_SORT_MAP)
        df = df.sort_values(by=['year', 'Sort_Key'])
    
    # df_cum 就是原始数据
    df_cum = df.copy()
    
    # --- 核心逻辑: 累计转单季 ---
    single_records = []
    
    if 'year' in df.columns:
        years = df['year'].unique()
        
        for year in years:
            year_data = df[df['year'] == year].set_index('period')
            
            def get_val(p, m):
                return year_data.loc[p, m] if p in year_data.index else np.nan

            # 准备4个季度的容器
            q_data = {
                'Q1': {'period': 'Q1', 'year': year},
                'Q2': {'period': 'Q2', 'year': year},
                'Q3': {'period': 'Q3', 'year': year},
                'Q4': {'period': 'Q4', 'year': year}
            }
            
            # 填充日期 (若存在)
            if 'report_date' in df.columns:
                if 'Q1' in year_data.index: q_data['Q1']['report_date'] = year_data.loc['Q1', 'report_date']
                if 'H1' in year_data.index: q_data['Q2']['report_date'] = year_data.loc['H1', 'report_date']
                if 'Q9' in year_data.index: q_data['Q3']['report_date'] = year_data.loc['Q9', 'report_date']
                if 'FY' in year_data.index: q_data['Q4']['report_date'] = year_data.loc['FY', 'report_date']

            for metric in ALL_METRIC_KEYS:
                # 兼容性检查：确保列存在
                if metric not in df.columns: continue

                val_q1 = get_val('Q1', metric)
                val_h1 = get_val('H1', metric)
                val_q9 = get_val('Q9', metric)
                val_fy = get_val('FY', metric)
                
                if metric in GROWTH_METRIC_KEYS:
                    # 流量指标 (营收/利润): 做减法
                    q_data['Q1'][metric] = val_q1
                    
                    # Q2 = H1 - Q1
                    if pd.notna(val_h1) and pd.notna(val_q1):
                        q_data['Q2'][metric] = val_h1 - val_q1
                    else:
                        q_data['Q2'][metric] = np.nan
                        
                    # Q3 = Q9 - H1
                    if pd.notna(val_q9) and pd.notna(val_h1):
                        q_data['Q3'][metric] = val_q9 - val_h1
                    else:
                        q_data['Q3'][metric] = np.nan
                        
                    # Q4 = FY - Q9
                    if pd.notna(val_fy) and pd.notna(val_q9):
                        q_data['Q4'][metric] = val_fy - val_q9
                    else:
                        q_data['Q4'][metric] = np.nan
                else:
                    # 存量指标 (债务/现金): 直接取期末值
                    q_data['Q1'][metric] = val_q1
                    q_data['Q2'][metric] = val_h1
                    q_data['Q3'][metric] = val_q9
                    q_data['Q4'][metric] = val_fy

            # 收集有效数据
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                # 只要有一个关键指标非空，就认为该季度有效
                if any(pd.notna(q_data[q].get(m)) for m in GROWTH_METRIC_KEYS if m in df.columns):
                    single_records.append(q_data[q])

    # 转换为 DataFrame
    df_single = pd.DataFrame(single_records)
    
    # 计算 TTM 和 YoY
    if not df_single.empty:
        # 排序
        df_single['Sort_Key'] = df_single['period'].apply(lambda x: int(x[1]) if isinstance(x, str) and len(x)>1 else 0)
        df_single = df_single.sort_values(by=['year', 'Sort_Key']).reset_index(drop=True)
        
        for metric in GROWTH_METRIC_KEYS:
            if metric not in df_single.columns: continue
            
            # TTM (滚动4季求和)
            df_single[f"{metric}_TTM"] = df_single[metric].rolling(4, min_periods=1).sum()
            
            # YoY (同比去年)
            prev_val = df_single[metric].shift(4)
            df_single[f"{metric}_YoY"] = (df_single[metric] - prev_val) / prev_val.abs()
            
            # TTM YoY
            prev_ttm = df_single[f"{metric}_TTM"].shift(4)
            df_single[f"{metric}_TTM_YoY"] = (df_single[f"{metric}_TTM"] - prev_ttm) / prev_ttm.abs()

    return df_cum, df_single