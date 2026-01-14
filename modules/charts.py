import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.calculator import process_financial_data
from modules.config import FINANCIAL_METRICS

def format_large_number(num):
    if pd.isna(num) or num is None: return "-"
    abs_num = abs(num)
    if abs_num >= 1e9: return f"{num/1e9:.2f}B"
    if abs_num >= 1e6: return f"{num/1e6:.2f}M"
    return f"{num:,.2f}"

def render_charts_tab(df_raw, unit_label="Raw"):
    st.subheader("📊 全维财务趋势分析")
    
    if df_raw.empty:
        st.warning("暂无数据，请先录入财务信息。")
        return

    # 1. 调用新的计算引擎
    df_cum, df_single = process_financial_data(df_raw)

    # 2. 控件
    c1, c2 = st.columns(2)
    with c1:
        # 筛选出当前数据中存在的列
        available_metrics = [m for m in FINANCIAL_METRICS if m['id'] in df_raw.columns]
        if not available_metrics:
            st.error("数据列缺失")
            return
            
        selected_metric = st.selectbox(
            "选择财务指标", 
            available_metrics, 
            format_func=lambda x: f"{x['label']}"
        )
        metric_key = selected_metric['id']
        
    with c2:
        view_mode = st.radio("视角", ["单季度 (Q1-Q4)", "TTM (滚动年化)", "累计原始值 (Q1/H1/Q9/FY)"], horizontal=True)

    # 3. 准备数据
    plot_data = pd.DataFrame()
    val_col = ""
    
    if view_mode == "单季度 (Q1-Q4)":
        plot_data = df_single.copy()
        val_col = metric_key
        # 添加 YoY 列名
        yoy_col = f"{metric_key}_YoY"
        
    elif view_mode == "TTM (滚动年化)":
        plot_data = df_single.copy()
        val_col = f"{metric_key}_TTM"
        yoy_col = f"{metric_key}_TTM_YoY"
        
    else: # 累计原始值
        plot_data = df_cum.copy()
        val_col = metric_key
        yoy_col = None

    if plot_data.empty:
        st.info("数据不足以生成图表")
        return

    # 4. 绘图
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 构造X轴 (year + period)
    plot_data['x_label'] = plot_data['year'].astype(str) + " " + plot_data['period']
    
    x = plot_data['x_label']
    y = plot_data.get(val_col, [])
    
    # 柱状图/面积图
    if view_mode == "TTM (滚动年化)":
        fig.add_trace(go.Scatter(x=x, y=y, name=f"{selected_metric['label']} (TTM)", fill='tozeroy'), secondary_y=False)
    else:
        fig.add_trace(go.Bar(x=x, y=y, name=selected_metric['label'], text=y.apply(format_large_number), textposition='auto'), secondary_y=False)

    # 增长率曲线
    if yoy_col and yoy_col in plot_data.columns:
        fig.add_trace(go.Scatter(x=x, y=plot_data[yoy_col], name="YoY 增长率", line=dict(color='orange')), secondary_y=True)
        fig.update_yaxes(title_text="增长率", tickformat=".1%", secondary_y=True)

    fig.update_layout(title=f"{selected_metric['label']} 趋势", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. 数据表
    with st.expander("查看详细数据"):
        cols = ['year', 'period', val_col]
        if yoy_col and yoy_col in plot_data.columns: cols.append(yoy_col)
        st.dataframe(plot_data[cols], use_container_width=True)