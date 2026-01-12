import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.calculator import process_financial_data

def render_valuation_PE_tab(df, unit_label):
    st.markdown("### 🧬 PE 状态变量分析系统 (State Variable System)")
    st.caption("基于 Prompt V2: PE 不是数值标签，而是由 [价格-增长-历史] 共同定义的动态状态。")
    
    if df.empty:
        st.warning("暂无数据。")
        return

    # --- 1. 数据驱动引擎 (Data Engine) ---
    # 获取清洗后的数据 (含 TTM 和 YoY)
    df_cum, df_single = process_financial_data(df)
    
    # 确保排序
    df_single = df_single.sort_values(by=['Year', 'Sort_Key'])
    
    # 核心变量提取
    latest_record = df_single.iloc[-1]
    
    # A. 提取 EPS (TTM) - 严格定义: Sum of last 4 reported quarters
    if len(df_single) >= 4:
        ttm_eps = df_single['EPS_Single'].tail(4).sum()
        ttm_label = "TTM (近4季)"
    else:
        ttm_eps = df_single['EPS_Single'].sum()
        ttm_label = "TTM (数据不足,仅统计现有)"
        
    # B. 提取增长率 (Growth State)
    # 优先使用 TTM 的同比增长，因为它熨平了季节性
    # 如果没有 TTM YoY (例如数据太少), 退化为 单季 YoY
    if 'EPS_TTM_YoY' in df_single.columns and not pd.isna(latest_record.get('EPS_TTM_YoY')):
        growth_rate = latest_record['EPS_TTM_YoY']
        growth_source = "EPS TTM YoY"
    else:
        growth_rate = latest_record.get('EPS_Single_YoY', 0.0)
        growth_source = "EPS Single YoY"

    # C. 构建历史 PE 上下文 (Historical Context)
    # Prompt 要求: Price_t = ClosePrice_of_Financial_Report_Month
    # 我们需要在 df_single 中计算历史每个时间点的 PE
    has_price = 'Close_Price_Single' in df_single.columns # Process data 会把 Close_Price 复制到 Single
    
    historical_pes = []
    if has_price:
        # 计算历史每一期的 PE (TTM)
        # 注意：每一期的 PE = 当期收盘价 / 当期 TTM EPS
        for i in range(len(df_single)):
            # 只有当 TTM 窗口足够 (比如 >=4) 且 EPS > 0 时，历史 PE 才有意义
            # 这里为了尽可能展示数据，放宽到有 TTM 数据即可
            p = df_single.iloc[i].get('Close_Price_Single', 0)
            e = df_single.iloc[i].get('EPS_TTM', 0) # calculator.py 需要确保计算了 EPS_TTM
            if p > 0 and e > 0:
                historical_pes.append(p / e)
    
    hist_pe_series = pd.Series(historical_pes)
    
    # --- 2. 用户交互与当前状态输入 ---
    
    col_input, col_info = st.columns([1, 2])
    with col_input:
        # 允许用户输入当前价格来模拟 "Now" 的状态，或者默认使用最近财报价格
        default_price = float(latest_record.get('Close_Price_Single', 100.0))
        if default_price == 0: default_price = 100.0
        
        current_price = st.number_input("当前价格 (Price_t)", value=default_price, step=0.1)
    
    with col_info:
        # 显示当前的基础状态
        st.info(f"""
        **基础状态数据**:
        * **EPS ({ttm_label})**: {ttm_eps:.3f}
        * **增速 ({growth_source})**: {growth_rate:.2%}
        * **有效历史 PE 样本数**: {len(historical_pes)} 个
        """)

    st.markdown("---")

    # --- 3. 核心逻辑：状态判定 (State Determination) ---

    # [逻辑分支 1] EPS <= 0: 亏损状态处理
    if ttm_eps <= 0:
        st.error("⚠️ 当前处于 [亏损/早期] 状态 (EPS TTM ≤ 0)")
        st.markdown("""
        **根据 Prompt V2 约束，禁止计算数值 PE。**
        
        **请关注以下状态变量：**
        1.  **亏损收窄速度**: 检查净利润 QoQ 是否为正。
        2.  **盈亏平衡点**: 预计何时转正？
        3.  **PS (市销率)**: 建议切换到 PS 估值模型。
        """)
        # 提前结束，不展示 PE 仪表盘
        return

    # [逻辑分支 2] 正常盈利状态
    current_pe = current_price / ttm_eps
    
    # 3.1 计算 PEG 联动状态
    # PEG = PE / (Growth * 100)
    # 保护: 如果增长率为负或0，PEG 无意义
    if growth_rate > 0:
        peg = current_pe / (growth_rate * 100)
    else:
        peg = None

    # 3.2 判定历史位置
    pe_rank_str = "无历史数据"
    if not hist_pe_series.empty:
        pe_median = hist_pe_series.median()
        pe_min = hist_pe_series.min()
        pe_max = hist_pe_series.max()
        
        # 简单的分位判断
        if current_pe < hist_pe_series.quantile(0.2):
            pe_pos = "极低 (Low)"
            color = "green"
        elif current_pe < hist_pe_series.quantile(0.8):
            pe_pos = "中枢 (Neutral)"
            color = "blue"
        else:
            pe_pos = "极高 (High)"
            color = "red"
    else:
        pe_median = 0
        pe_pos = "未知 (Unknown)"
        color = "gray"

    # --- 4. 状态仪表盘 (State Dashboard) ---
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("PE (TTM) 状态值", f"{current_pe:.2f}x", delta_color="off")
        st.caption(f"历史中位数: {pe_median:.2f}x")
    
    with c2:
        if peg:
            status = "低估" if peg < 1 else ("高估" if peg > 2 else "合理")
            st.metric("PEG 联动状态", f"{peg:.2f}", f"{status}")
        else:
            st.metric("PEG 联动状态", "无效", "负增长/零增长")
        st.caption(f"对应增速: {growth_rate:.1%}")

    with c3:
        st.markdown(f"**历史区间位置**")
        st.markdown(f":{color}[**{pe_pos}**]")
        if not hist_pe_series.empty:
            st.caption(f"Range: [{pe_min:.1f}x - {pe_max:.1f}x]")

    # --- 5. 综合结论输出 (Agent Output) ---
    st.markdown("### 📝 估值状态结论 (Agent Output)")
    
    conclusion = ""
    if peg and peg < 0.8 and growth_rate > 0.2:
        conclusion = "**[强力买入区 - GARP]**: PE 相对低估，且伴随高增长，PEG < 0.8。属于典型的 '戴维斯双击' 潜力区。"
    elif peg and peg > 2.0:
        conclusion = "**[风险泡沫区]**: 估值 (PE) 显著高于增长 (G) 的支撑能力。除非有极其确定的加速增长预期，否则需警惕均值回归。"
    elif growth_rate < 0:
        conclusion = "**[价值陷阱警示]**: PE 可能看起来很低，但 EPS 在负增长。这是 '周期性下行' 或 '基本面恶化' 的特征，由于分母变小，未来 PE 会被动升高。"
    elif abs(current_pe - pe_median) / pe_median < 0.15:
        conclusion = "**[合理定价区]**: 当前 PE 处于历史中枢附近，且 PEG 在合理范围。未来回报主要取决于 EPS 的实质增长。"
    else:
        conclusion = "**[观察区]**: 状态特征不明显，建议结合宏观利率环境进一步判断。"

    st.success(conclusion)

    # --- 6. 可视化：PE Band (历史 PE 通道) ---
    if not hist_pe_series.empty and has_price:
        st.subheader("📉 历史 PE 通道 (Valuation Band)")
        
        # 构造绘图数据
        df_chart = df_single.copy()
        # 过滤掉 EPS <= 0 的点
        df_chart = df_chart[df_chart['EPS_TTM'] > 0]
        
        if not df_chart.empty:
            df_chart['Date_Label'] = df_chart['Year'].astype(str) + " " + df_chart['Quarter_Name']
            
            fig = go.Figure()

            # 实际价格线
            fig.add_trace(go.Scatter(
                x=df_chart['Date_Label'], y=df_chart['Close_Price_Single'],
                mode='lines+markers', name='实际股价 (Price)',
                line=dict(color='black', width=3)
            ))
            
            # 理论价格线 (基于历史 PE 分位 * 当期 EPS)
            # P_implied = PE_benchmark * EPS_TTM
            pe_20 = hist_pe_series.quantile(0.2)
            pe_50 = hist_pe_series.quantile(0.5)
            pe_80 = hist_pe_series.quantile(0.8)
            
            fig.add_trace(go.Scatter(
                x=df_chart['Date_Label'], y=df_chart['EPS_TTM'] * pe_80,
                mode='lines', name=f'高估线 (PE={pe_80:.1f}x)',
                line=dict(color='rgba(255, 0, 0, 0.3)', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_chart['Date_Label'], y=df_chart['EPS_TTM'] * pe_50,
                mode='lines', name=f'中枢线 (PE={pe_50:.1f}x)',
                line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_chart['Date_Label'], y=df_chart['EPS_TTM'] * pe_20,
                mode='lines', name=f'低估线 (PE={pe_20:.1f}x)',
                line=dict(color='rgba(0, 255, 0, 0.3)', dash='dash')
            ))

            fig.update_layout(title="股价 vs 估值锚点 (基于历史 PE 区间)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("注：虚线代表若股价按照历史 PE (P20/P50/P80) 交易时的理论价格。")
    
    else:
        st.info("需要更多包含 'Close_Price' 和正收益的数据点来生成 PE 通道图。")
