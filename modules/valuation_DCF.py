import streamlit as st
import pandas as pd
from modules.calculator import process_financial_data

def render_valuation_DCF_tab(df_raw, wacc, rf, unit_label):
    st.subheader("🚀 DCF 现金流折现 (SQLite 版)")
    
    if df_raw.empty: return
    
    # 1. 自动计算基准数据
    _, df_single = process_financial_data(df_raw)
    
    if df_single.empty or 'FCF_TTM' not in df_single.columns:
        st.warning("缺少 FCF 数据，请录入自由现金流")
        return
        
    latest = df_single.iloc[-1]
    base_fcf = latest.get('FCF_TTM', 0)
    
    # 2. 参数输入
    c1, c2, c3 = st.columns(3)
    init_fcf = c1.number_input("基准 FCF (TTM)", value=float(base_fcf))
    growth_rate = c2.number_input("前5年增长率 (%)", value=10.0) / 100
    perp_rate = c3.number_input("永续增长率 (%)", value=2.5) / 100
    
    if wacc <= perp_rate:
        st.error("WACC 必须大于永续增长率")
        return
        
    # 3. 计算
    flows = []
    curr = init_fcf
    total_pv = 0
    
    st.write("未来现金流预测:")
    cols = st.columns(5)
    for i in range(1, 6):
        curr = curr * (1 + growth_rate)
        pv = curr / ((1 + wacc) ** i)
        total_pv += pv
        cols[i-1].metric(f"Y{i}", f"{curr:.2f}", f"PV: {pv:.2f}")
        flows.append(curr)
        
    # 终值
    term_val = flows[-1] * (1 + perp_rate) / (wacc - perp_rate)
    term_pv = term_val / ((1 + wacc) ** 5)
    
    enterprise_value = total_pv + term_pv
    
    st.divider()
    st.metric("企业价值 (EV)", f"{enterprise_value:,.2f} {unit_label}")
    st.caption(f"阶段1现值: {total_pv:,.2f} + 终值现值: {term_pv:,.2f}")