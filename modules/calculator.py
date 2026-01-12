import pandas as pd
import numpy as np
from modules.config import GROWTH_METRIC_KEYS, ALL_METRIC_KEYS

PERIOD_MAP_SORT = {"Q1": 1, "H1": 2, "Q9": 3, "FY": 4}
PERIOD_MAP_DISPLAY = {"Q1": "Q1", "H1": "Q2", "Q9": "Q3", "FY": "Q4"}

def process_financial_data(df):
    if df.empty:
        return df, df

    df = df.copy()
    df['Sort_Key'] = df['Period'].map(PERIOD_MAP_SORT)
    df = df.sort_values(by=['Year', 'Sort_Key']).reset_index(drop=True)
    
    df_single = df.copy()
    df_single['Quarter_Name'] = df_single['Period'].map(PERIOD_MAP_DISPLAY)

    # 1. 处理需要拆解单季度的增长指标 (Revenue, EPS, Profit...)
    target_metrics = GROWTH_METRIC_KEYS 
    valid_metrics = [m for m in target_metrics if m in df.columns]

    for metric in valid_metrics:
        # A. 单季值拆解
        df_single = _calculate_single_quarter_value(df_single, metric)
        # B. 累计 YoY
        df = _calculate_yoy(df, metric, is_single=False)
        # C. 单季 YoY
        df_single = _calculate_yoy(df_single, f"{metric}_Single", is_single=True)
        # D. TTM 计算 (关键)
        ttm_col = f"{metric}_TTM"
        df_single[ttm_col] = df_single[f"{metric}_Single"].rolling(window=4).sum()
        # E. TTM YoY (用于判断增长趋势的持续性 - Prompt V2 核心)
        df_single = _calculate_yoy(df_single, ttm_col, is_single=True)
    
    # 2. 处理存量/非累计指标 (Close_Price, Market_Cap, Debt)
    # 这些不需要 diff，单季度值 = 报告期值
    non_growth_metrics = [m for m in ALL_METRIC_KEYS if m not in GROWTH_METRIC_KEYS and m in df.columns]
    
    for metric in non_growth_metrics:
        # 直接复制
        df_single[f"{metric}_Single"] = df_single[metric]
        # TTM 对存量数据通常取"最新值"，而不是求和
        df_single[f"{metric}_TTM"] = df_single[metric]

    return df, df_single

# ... (内部函数 _calculate_single_quarter_value, _calculate_yoy, _calculate_qoq 保持不变，见上文) ...
def _calculate_single_quarter_value(df, metric_name):
    target_col = f"{metric_name}_Single"
    df[target_col] = df.groupby('Year')[metric_name].diff()
    mask_q1 = df['Period'] == 'Q1'
    df.loc[mask_q1, target_col] = df.loc[mask_q1, metric_name]
    return df

def _calculate_yoy(df, col_name, is_single=False):
    prev_year_df = df.copy()
    prev_year_df['Year'] = prev_year_df['Year'] + 1
    join_keys = ['Year', 'Quarter_Name'] if is_single else ['Year', 'Period']
    prev_year_subset = prev_year_df[join_keys + [col_name]]
    merged = pd.merge(df, prev_year_subset, on=join_keys, how='left', suffixes=('', '_PrevYear'))
    prev_val = merged[f'{col_name}_PrevYear']
    curr_val = merged[col_name]
    growth_col = f"{col_name}_YoY"
    df[growth_col] = (curr_val - prev_val) / prev_val.abs()
    return df
