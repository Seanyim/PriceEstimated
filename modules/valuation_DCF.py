# modules/valuation_DCF.py
import streamlit as st
import pandas as pd
from modules.calculator import process_financial_data

def render_valuation_DCF_tab(df, wacc, unit_label):
    prefix = "dcf"
    st.subheader("自动 DCF 估值模型")
    
    if df.empty:
        st.warning("暂无数据")
        return

    # 1. 数据处理 (calculator 现在会生成 TTM 数据)
    df_cum, df_single = process_financial_data(df)
    df_single = df_single.sort_values(by=['Year', 'Sort_Key'])
    
    if len(df_single) < 4:
        st.warning("⚠️ 数据不足 4 个季度，无法计算 TTM 年化增长率，将使用单季数据代替，建议补充完整数据。")
    
    latest_single = df_single.iloc[-1]

    # --- [关键修改点] 获取增长率 ---
    # 优先顺序：TTM 同比增长 > 单季同比 > 默认 10%
    # TTM Growth 代表了"过去一年完整的增长能力"，平滑了季节性
    
    ttm_growth = latest_single.get('Profit_TTM_YoY', None) # calculator 新增的字段
    single_growth = latest_single.get('Profit_Single_YoY', 0.10)
    
    if pd.notna(ttm_growth):
        default_growth = ttm_growth
        growth_source_label = "TTM (滚动年化) 增长率"
    else:
        default_growth = single_growth
        growth_source_label = "单季度 (季报) 增长率 [数据不足TTM]"

    # TTM 利润 (基准 FCF)
    if pd.notna(latest_single.get('Profit_TTM')):
        ttm_profit = latest_single['Profit_TTM']
        fcf_hint = "TTM 净利润 (滚动4季总和)"
    else:
        # 降级处理
        ttm_profit = latest_single['Profit_Single'] * 4 
        fcf_hint = "单季净利润 x 4 (估算)"

    # --- 界面部分 ---
    st.markdown(f"### 1. 现金流与增长假设 (单位: {unit_label})")
    
    col1, col2 = st.columns(2)
    
    cf_start = col1.number_input(
        f"基准自由现金流 (Base FCF)", 
        value=float(ttm_profit), 
        format="%.2f",
        help=f"默认加载: {fcf_hint}",
        key=f"{prefix}_fcf_start"
    )

    g_rate = col2.number_input(
        "未来 5 年预期增长率 (%)",
        value=float(default_growth * 100),
        step=0.5,
        format="%.1f",
        # 在 help 中提示用户当前增长率的来源
        help=f"系统自动抓取: {growth_source_label} ({default_growth:.1%})",
        key=f"{prefix}_g_rate"
    ) / 100.0

    st.info(f"💡 折现率 WACC: **{wacc*100:.2f}%** | 增长率参考: **{growth_source_label}**")

    # ... (后续计算逻辑保持不变) ...
    with st.expander("高级设置 (永续增长率)"):
        g_stable = st.slider(
            "永续增长率",
            0.0, 0.05, 0.025,
            step=0.001,
            key=f"{prefix}_g_stable"
        )
    
    # 检查 WACC > g_stable
    if wacc <= g_stable:
        st.error("❌ WACC 必须大于永续增长率")
        return

    # 计算
    cash_flows = []
    for i in range(1, 6):
        cf = cf_start * ((1 + g_rate) ** i)
        pv = cf / ((1 + wacc) ** i)
        cash_flows.append(pv)
        
    sum_pv = sum(cash_flows)
    
    # 终值
    cf_5 = cf_start * ((1 + g_rate) ** 5)
    tv = (cf_5 * (1 + g_stable)) / (wacc - g_stable)
    pv_tv = tv / ((1 + wacc) ** 5)
    
    total = sum_pv + pv_tv
    
    c_res1, c_res2 = st.columns(2)
    c_res1.metric("预测期现值", f"{sum_pv:.2f}")
    c_res2.metric("终值折现", f"{pv_tv:.2f}")
    
    st.divider()
    st.metric("🚀 DCF 估值", f"{total:.2f} {unit_label}")