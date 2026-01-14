import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.calculator import process_financial_data
from modules.db import get_market_history

def render_valuation_PE_tab(df_raw, unit_label):
    st.subheader("📊 PE 估值模型 (SQLite 版)")
    
    if df_raw.empty:
        st.warning("暂无财务数据")
        return

    # 1. 获取单季数据 (为了获得 EPS TTM)
    _, df_single = process_financial_data(df_raw)
    
    if df_single.empty or 'EPS_TTM' not in df_single.columns:
        st.warning("无法计算 EPS TTM，请检查是否录入了利润/EPS数据")
        return

    # 2. 结合股价历史
    # 从 df_raw 中提取 ticker (假设是同一家公司)
    ticker = df_raw.iloc[0]['ticker']
    df_price = get_market_history(ticker) # 获取每日股价
    
    if df_price.empty:
        st.info("⚠️ 暂无历史股价数据，请在数据录入页面点击【开始同步】。")
        return

    # 3. 匹配股价与财报 (以财报日期为准，找最近的股价)
    # 确保 report_date 是 datetime
    df_single['report_date'] = pd.to_datetime(df_single['report_date'])
    df_price['date'] = pd.to_datetime(df_price['date'])
    
    # 排序
    df_price = df_price.sort_values('date')
    df_single = df_single.sort_values('report_date')
    
    # 使用 merge_asof 模糊匹配最近的股价
    df_merge = pd.merge_asof(
        df_single, 
        df_price, 
        left_on='report_date', 
        right_on='date', 
        direction='backward'
    )
    
    # 计算历史 PE
    df_merge['PE_TTM'] = df_merge['close'] / df_merge['EPS_TTM']
    
    # 过滤异常值
    valid_pe = df_merge[(df_merge['PE_TTM'] > 0) & (df_merge['PE_TTM'] < 200)]
    
    if valid_pe.empty:
        st.warning("有效 PE 数据不足 (需 EPS>0 且有对应股价)")
        return
        
    # 4. 统计分析
    pe_median = valid_pe['PE_TTM'].median()
    pe_20 = valid_pe['PE_TTM'].quantile(0.2)
    pe_80 = valid_pe['PE_TTM'].quantile(0.8)
    
    latest = valid_pe.iloc[-1]
    current_pe = latest['PE_TTM']
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前 PE (TTM)", f"{current_pe:.2f}")
    c2.metric("中位 PE", f"{pe_median:.2f}")
    c3.metric("低估区 (P20)", f"{pe_20:.2f}")
    c4.metric("高估区 (P80)", f"{pe_80:.2f}")
    
    # 5. 绘制 PE Band
    st.markdown("#### 📉 PE Band 通道图")
    fig = go.Figure()
    
    # 真实股价
    fig.add_trace(go.Scatter(x=valid_pe['report_date'], y=valid_pe['close'], name="股价", line=dict(color='black', width=2)))
    
    # 理论股价线
    fig.add_trace(go.Scatter(x=valid_pe['report_date'], y=valid_pe['EPS_TTM']*pe_80, name=f"高估 ({pe_80:.1f}x)", line=dict(dash='dot', color='red')))
    fig.add_trace(go.Scatter(x=valid_pe['report_date'], y=valid_pe['EPS_TTM']*pe_median, name=f"中枢 ({pe_median:.1f}x)", line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=valid_pe['report_date'], y=valid_pe['EPS_TTM']*pe_20, name=f"低估 ({pe_20:.1f}x)", line=dict(dash='dot', color='green')))
    
    st.plotly_chart(fig, use_container_width=True)