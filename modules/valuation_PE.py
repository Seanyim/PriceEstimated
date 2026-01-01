import streamlit as st
import pandas as pd
import numpy as np
from modules.calculator import process_financial_data

def render_valuation_PE_tab(df, unit_label):
    st.subheader("多维 PE 估值模型 (TTM / Static / Dynamic / PEG)")
    
    if df.empty:
        st.warning("暂无数据，请先在数据录入页添加财务数据。")
        return

    # --- 1. 数据预处理 ---
    # 调用 calculator 模块，获取清洗后的累计数据(cum)和单季度数据(single)
    # df_single 中包含了拆分好的 'EPS_Single' 和自动计算的 'EPS_Single_YoY'
    df_cum, df_single = process_financial_data(df)
    
    # 确保数据按时间正序排列
    df_single = df_single.sort_values(by=['Year', 'Sort_Key'])
    df_cum = df_cum.sort_values(by=['Year', 'Sort_Key'])

    # --- 2. 关键指标计算 ---
    
    # A. 静态 EPS (Static EPS) - 取最近一个完整财年 (FY) 的 EPS
    last_fy_data = df_cum[df_cum['Period'] == 'FY']
    if not last_fy_data.empty:
        static_eps = last_fy_data.iloc[-1]['EPS']
        static_year = int(last_fy_data.iloc[-1]['Year'])
    else:
        static_eps = 0.0
        static_year = "-"

    # B. 滚动 EPS (TTM EPS) - 最近 4 个单季度的 EPS 之和
    # 只有当数据量 >= 4 时计算才有意义
    if len(df_single) >= 4:
        ttm_eps = df_single['EPS_Single'].tail(4).sum()
        ttm_label = "过去4季度"
    else:
        # 数据不足时降级为使用静态 EPS 或当前累计
        ttm_eps = df_single['EPS_Single'].sum() 
        ttm_label = "数据不足4季(仅统计现有)"

    # C. 获取增长率参考值 (Reference Growth Rate)
    # 优先取最近单季度的 EPS 同比增长率
    latest_single = df_single.iloc[-1]
    ref_growth = latest_single.get('EPS_Single_YoY', 0.0)
    
    # 如果单季增长率无效(如NaN)，尝试取累计增长率
    if pd.isna(ref_growth) or ref_growth == 0:
        latest_cum = df_cum.iloc[-1]
        ref_growth = latest_cum.get('EPS_YoY', 0.0)

    # --- 3. 界面交互 ---

    # 输入：股价
    col_input, _ = st.columns([1, 2])
    with col_input:
        current_price = st.number_input("当前股价", min_value=0.0, value=100.0, step=0.1)

    st.markdown("---")

    # --- 4. 四大估值指标展示 ---
    col1, col2, col3, col4 = st.columns(4)

    # [1] 静态市盈率 (Static PE)
    with col1:
        st.markdown("##### 🏛️ 静态 PE (Static)")
        st.caption(f"基准: {static_year} FY EPS = {static_eps:.2f}")
        
        if static_eps > 0:
            static_pe = current_price / static_eps
            st.metric("Static PE", f"{static_pe:.2f}x")
        else:
            st.metric("Static PE", "N/A", help="EPS <= 0 或无FY数据")

    # [2] 滚动市盈率 (TTM PE) - 市场最常用
    with col2:
        st.markdown("##### 🔄 滚动 PE (TTM)")
        st.caption(f"基准: {ttm_label} EPS = {ttm_eps:.2f}")
        
        if ttm_eps > 0:
            ttm_pe = current_price / ttm_eps
            st.metric("TTM PE", f"{ttm_pe:.2f}x")
        else:
            st.metric("TTM PE", "N/A", help="TTM EPS <= 0")

    # [3] 动态市盈率 (Forward PE)
    with col3:
        st.markdown("##### 🔮 动态 PE (Forward)")
        # 允许用户调整预期增长率，默认使用历史计算出的增长率
        default_g = float(ref_growth * 100) if not pd.isna(ref_growth) else 10.0
        expected_g = st.number_input("预期增速(%)", value=default_g, step=1.0, format="%.1f") / 100.0
        
        # 估算下一年 EPS = TTM EPS * (1 + g) 
        # (注：也可以基于静态EPS估算，这里采用TTM更贴近现状)
        base_eps = ttm_eps if ttm_eps > 0 else static_eps
        forward_eps = base_eps * (1 + expected_g)
        
        st.caption(f"预估 Next EPS: {forward_eps:.2f}")
        
        if forward_eps > 0:
            forward_pe = current_price / forward_eps
            st.metric("Forward PE", f"{forward_pe:.2f}x")
        else:
            st.metric("Forward PE", "N/A")

    # [4] PEG 估值
    with col4:
        st.markdown("##### ⚖️ PEG 比率")
        # PEG = TTM PE / (预期增长率 * 100)
        # 也就是：你为了这 1% 的增长支付了多少倍的 PE
        
        calc_g_val = expected_g * 100 # 使用用户刚才确认的预期增长率
        
        st.caption(f"计算基准: TTM PE / G({calc_g_val:.1f})")
        
        if ttm_eps > 0 and calc_g_val > 0:
            # 重新计算当前的 TTM PE
            pe_now = current_price / ttm_eps
            peg = pe_now / calc_g_val
            
            st.metric("PEG Ratio", f"{peg:.2f}")
            
            if peg < 0.8:
                st.success("低估 (<0.8)")
            elif peg > 2.0:
                st.error("高估 (>2.0)")
            else:
                st.info("合理区间")
        else:
            st.metric("PEG", "N/A", help="PE或增长率为负，PEG失效")

    # --- 5. 辅助数据表 ---
    with st.expander("查看计算详情 (单季EPS与TTM构成)"):
        # 展示最近4个季度的构成
        if len(df_single) > 0:
            st.write("最近 4 个季度数据 (用于计算 TTM):")
            cols = ['Year', 'Quarter_Name', 'EPS_Single', 'EPS_Single_YoY']
            # 取最后4行并反转，方便查看最新的
            display_df = df_single[cols].tail(4).iloc[::-1].copy()
            st.dataframe(display_df.style.format({
                "EPS_Single": "{:.3f}", 
                "EPS_Single_YoY": "{:.2%}"
            }))