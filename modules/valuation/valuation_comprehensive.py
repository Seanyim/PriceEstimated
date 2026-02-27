# modules/valuation/valuation_comprehensive.py
# 综合分析模块 v2.3
# 聚合 EV/EBITDA、增长率透视、Monte Carlo、ROIC/ROA/ROE

import streamlit as st
import pandas as pd
from modules.core.calculator import process_financial_data
from modules.core.db import get_company_meta
from modules.valuation.valuation_advanced import (
    safe_get,
    _render_ev_ebitda,
    _render_growth_analysis,
    _render_monte_carlo,
    _render_profitability_analysis
)


def render_comprehensive_tab(df_raw, unit_label, wacc, rf):
    """综合分析 Tab — 含 EV/EBITDA、增长率透视、Monte Carlo、ROIC/ROA/ROE"""
    st.subheader("💹 综合分析")
    
    if df_raw.empty:
        st.warning("请先录入财务数据")
        return
    
    # 获取基础数据
    _, df_single = process_financial_data(df_raw)
    if df_single.empty:
        st.warning("财务数据不足")
        return
    
    latest = df_single.iloc[-1]
    ticker = df_raw.iloc[0]['ticker']
    meta = get_company_meta(ticker)
    
    # 子 Tab
    sub_tabs = st.tabs([
        "💹 EV/EBITDA",
        "📈 增长率透视",
        "🎲 Monte Carlo",
        "📉 ROIC/ROA/ROE"
    ])
    
    with sub_tabs[0]:
        _render_ev_ebitda(df_single, latest, meta, unit_label)
    
    with sub_tabs[1]:
        _render_growth_analysis(df_single, unit_label)
    
    with sub_tabs[2]:
        _render_monte_carlo(df_single, latest, meta, wacc, unit_label)
    
    with sub_tabs[3]:
        _render_profitability_analysis(df_single, unit_label)
