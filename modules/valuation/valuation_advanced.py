# modules/valuation_advanced.py
# 高级估值模型模块
# v1.1 - 修复 None 值处理

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.core.calculator import process_financial_data
from modules.core.db import get_company_meta, get_market_history
from modules.core.risk_free_rate import get_risk_free_rate
from modules.data.industry_data import get_industry_benchmarks

def safe_get(row, key, default=0):
    """安全获取 DataFrame 行的值，处理 None 和 NaN"""
    val = row.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def render_advanced_valuation_tab(df_raw, unit_label, wacc, rf):
    """渲染高级估值模型 Tab"""
    st.subheader("🔬 高级估值模型")
    
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
        "🔄 DCF 倒推",
        "📊 PEG 倒推", 
        "💹 EV/EBITDA",
        "📈 增长率透视",
        "🎲 Monte Carlo",
        "📉 ROIC/ROA/ROE"
    ])
    
    with sub_tabs[0]:
        _render_dcf_reverse(df_single, latest, meta, wacc, rf, unit_label, df_raw)
    
    with sub_tabs[1]:
        _render_peg_analysis(df_single, latest, meta, unit_label)
    
    with sub_tabs[2]:
        _render_ev_ebitda(df_single, latest, meta, unit_label)
    
    with sub_tabs[3]:
        _render_growth_analysis(df_single, unit_label)
    
    with sub_tabs[4]:
        _render_monte_carlo(df_single, latest, meta, wacc, unit_label)
    
    with sub_tabs[5]:
        _render_profitability_analysis(df_single, unit_label)


def _render_dcf_reverse(df_single, latest, meta, wacc, rf, unit_label, df_raw=None):
    """DCF 倒推 - 从当前市值倒推市场隐含增长率 (v2.1)"""
    st.markdown("#### 🔄 DCF 倒推分析 (Reverse DCF)")
    st.caption("基于当前市值，倒推市场对未来5年的隐含增长率预期。")
    
    market_cap = meta.get('last_market_cap', 0)
    if market_cap == 0:
        st.warning("⚠️ 缺少市值数据，无法进行倒推")
        return

    # --- FCF 基准选择 (与 DCF 模块对齐) ---
    base_fcf = 0
    fcf_source = "Unknown"
    
    # 获取需要的数据
    val_ttm = latest.get('FreeCashFlow_TTM', 0)
    
    # 尝试找最新 FY
    df_fy = pd.DataFrame()
    latest_fy_year = 0
    if df_raw is not None and not df_raw.empty:
        df_fy = df_raw[df_raw['period'] == 'FY'].sort_values('year')
        latest_fy_year = df_fy.iloc[-1]['year'] if not df_fy.empty else 0
    
    val_fy = df_fy.iloc[-1].get('FreeCashFlow', 0) if not df_fy.empty else 0
    
    # 补全逻辑
    if val_ttm == 0:
        o = latest.get('OperatingCashFlow_TTM', 0)
        c = abs(latest.get('CapEx', 0))
        if o != 0: val_ttm = o - c
        
    if val_fy == 0 and not df_fy.empty:
        o = df_fy.iloc[-1].get('OperatingCashFlow', 0)
        c = abs(df_fy.iloc[-1].get('CapEx', 0))
        if o != 0: val_fy = o - c

    # 判断是否使用 TTM
    use_ttm = True
    if df_raw is not None:
        last_record_year = latest.get('year', 0)
        # 如果季度数据比FY新，倾向于TTM
        if last_record_year > latest_fy_year and val_ttm != 0:
            use_ttm = True
        elif val_fy != 0:
            use_ttm = False
            
    if use_ttm and val_ttm != 0:
        base_fcf = val_ttm
        fcf_source = "FCF TTM"
    elif val_fy != 0:
        base_fcf = val_fy
        fcf_source = f"FCF FY{latest_fy_year}"
    else:
        base_fcf = val_ttm # Final fallback
    
    if base_fcf == 0:
        st.warning("⚠️ 需要 FCF 数据")
        return
        
    # 展示核心参数
    c1, c2, c3 = st.columns(3)
    c1.metric("当前市值", f"{market_cap/1e9:.2f}B")
    c2.metric(f"基准 FCF ({fcf_source})", f"{base_fcf:,.2f} {unit_label}")
    c3.metric("WACC", f"{wacc:.1%}")
    
    # 输入: 永续增长率 (v2.1 改为 unrestricted number input)
    perp_rate_input = st.number_input(
        "永续增长率假设 (%)", 
        value=2.50,
        step=0.01,
        format="%.2f",
        key="adv_dcf_perp_growth",
        help="支持任意数值手动输入"
    )
    perp_rate = perp_rate_input / 100
    
    if wacc <= perp_rate:
        st.error(f"❌ WACC ({wacc:.1%}) 必须大于永续增长率")
        return

    # --- 倒推计算 ---
    # Goal: Find g such that DCF(g) = Market Cap
    # DCF = Sum(FCF_i / (1+w)^i) + TV / (1+w)^5
    
    fcf_dollars = base_fcf * 1e9 if base_fcf < 10000 else base_fcf
    
    def calculate_ev(g):
        total_pv = 0
        curr = fcf_dollars
        for i in range(1, 6):
            curr = curr * (1 + g)
            total_pv += curr / ((1 + wacc) ** i)
        
        # Terminal
        term_val = curr * (1 + perp_rate) / (wacc - perp_rate)
        term_pv = term_val / ((1 + wacc) ** 5)
        return total_pv + term_pv

    # 二分查找
    low = -0.5
    high = 1.0 # 100% Growth
    implied_g = None
    
    for _ in range(100):
        mid = (low + high) / 2
        ev = calculate_ev(mid)
        if abs(ev - market_cap) < market_cap * 0.0001:
            implied_g = mid
            break
        if ev < market_cap:
            low = mid
        else:
            high = mid
            
    implied_g = (low + high) / 2
    
    st.divider()
    
    # 结果展示
    st.markdown(f"#### 💡 市场隐含增长率: **{implied_g:.1%}**")
    st.caption(f"即：为支撑当前 {market_cap/1e9:.1f}B 市值，市场预期未来 5 年 FCF 需保持 {implied_g:.1%} 的复合增长。")
    
    # FCF 拆解展示
    st.markdown("**📅 隐含 FCF 路径分解**")
    
    proj_data = []
    curr = fcf_dollars
    for i in range(1, 6):
        prev = curr
        curr = curr * (1 + implied_g)
        change = curr - prev
        proj_data.append({
            "年份": f"Y{i}",
            f"FCF 预测 ({unit_label})": f"{curr/1e9:.2f}B" if base_fcf < 10000 else f"{curr:.2f}",
            "YoY": f"{implied_g:.1%}",
            "折现因子": f"{1/((1+wacc)**i):.3f}"
        })
    
    st.dataframe(pd.DataFrame(proj_data), use_container_width=True, hide_index=True)
    
    # 敏感性分析
    st.markdown("**🎯 敏感性分析: WACC vs 永续增长率 → 隐含增长率**")
    
    wacc_opts = [wacc-0.01, wacc-0.005, wacc, wacc+0.005, wacc+0.01]
    perp_opts = [perp_rate-0.01, perp_rate-0.005, perp_rate, perp_rate+0.005, perp_rate+0.01]
    
    mtx = []
    for p in perp_opts:
        row = []
        for w in wacc_opts:
            if w <= p:
                row.append(None)
                continue
            # Solve for g
            l, h = -0.5, 1.0
            for _ in range(20):
                m = (l+h)/2
                # calc EV with this w, p, m
                c = fcf_dollars
                tp = 0
                for i in range(1,6):
                    c *= (1+m)
                    tp += c/((1+w)**i)
                tv = c*(1+p)/(w-p)
                tp += tv/((1+w)**5)
                if tp < market_cap: l = m
                else: h = m
            row.append((l+h)/2 * 100)
        mtx.append(row)
        
    fig = go.Figure(data=go.Heatmap(
        z=mtx,
        x=[f"WACC {w:.1%}" for w in wacc_opts],
        y=[f"g_perp {p:.1%}" for p in perp_opts],
        colorscale='RdYlGn',
        texttemplate="%{z:.1f}%",
        colorbar=dict(title="隐含5年增长率(%)")
    ))
    fig.update_layout(height=350, title="敏感性矩阵")
    st.plotly_chart(fig, use_container_width=True)

def _render_peg_analysis(df_single, latest, meta, unit_label):
    """PEG 倒推分析"""
    st.markdown("#### 📊 PEG 倒推分析")
    st.caption("基于 PEG=1 反推合理股价")
    
    eps_ttm = safe_get(latest, 'EPS_TTM', 0)
    
    # 从财务数据计算增长率
    cagr = 0.15  # 默认值
    growth_source = "默认"
    
    # 优先使用 EPS_TTM_YoY
    if 'EPS_TTM_YoY' in df_single.columns:
        latest_yoy = safe_get(latest, 'EPS_TTM_YoY', None)
        if latest_yoy is not None and latest_yoy > 0:
            cagr = latest_yoy
            growth_source = "EPS TTM 同比"
    
    # 备选：使用历史 EPS 计算 CAGR
    if growth_source == "默认" and 'EPS_TTM' in df_single.columns and len(df_single) >= 5:
        eps_series = df_single['EPS_TTM'].dropna()
        if len(eps_series) >= 5:
            eps_old = eps_series.iloc[-5]
            eps_new = eps_series.iloc[-1]
            if eps_old > 0 and eps_new > 0:
                cagr = (eps_new / eps_old) ** (1/4) - 1
                growth_source = "EPS 4年 CAGR"
    
    # 获取最新股价
    ticker = df_single.iloc[0].get('ticker', '') if len(df_single) > 0 else ''
    df_price = get_market_history(ticker) if ticker else pd.DataFrame()
    
    current_price = 0
    if not df_price.empty:
        current_price = df_price.iloc[-1].get('close', 0) or 0
    
    # 数据验证
    if eps_ttm <= 0:
        st.warning("⚠️ EPS TTM 数据无效或为负数，无法计算 PEG")
        st.info(f"当前 EPS TTM: {eps_ttm}")
        return
    
    if current_price <= 0:
        st.warning("⚠️ 缺少股价数据，请先同步市场数据")
        return
    
    # 计算 PE 和 PEG
    current_pe = current_price / eps_ttm
    growth_pct = cagr * 100  # 转为百分比
    current_peg = current_pe / growth_pct if growth_pct > 0 else float('inf')
    
    # ===== 费雪利率修正 PEG (Fisher Adjusted PEG) =====
    # 费雪提出：考虑到利率环境，PEG 应调整为 PEG / (无风险利率 * 2)
    # 当利率较高时，相同的 PEG 代表更高的估值
    # Fisher Adjusted PEG = PE / (G + 2*rf) 其中 G 为增长率%，rf 为无风险利率%
    from modules.core.risk_free_rate import get_risk_free_rate
    
    rf_rate = get_risk_free_rate(use_cache=True)
    rf_pct = rf_rate * 100  # 转为百分比
    
    # Fisher 修正公式: 合理 PE = 增长率 + 2*无风险利率
    fisher_denominator = growth_pct + 2 * rf_pct
    fisher_peg = current_pe / fisher_denominator if fisher_denominator > 0 else float('inf')
    
    # ===== 完整计算过程展示 =====
    st.markdown("##### 📐 计算过程")
    
    with st.expander("🔍 查看详细计算", expanded=False):
        st.markdown(f"""
**1. 基础数据:**
- 最新股价: **${current_price:.2f}**
- EPS TTM: **${eps_ttm:.2f}**
- 增长率 (G): **{growth_pct:.2f}%** (来源: {growth_source})
- 无风险利率 (rf): **{rf_pct:.2f}%**

**2. 传统 PEG 计算:**
PE = {current_pe:.2f}, PEG = {current_peg:.2f}

**3. 费雪利率修正 PEG:**
Fisher PEG = PE / (G + 2×rf) = {current_pe:.2f} / ({growth_pct:.2f} + 2×{rf_pct:.2f}) = {fisher_peg:.2f}
        """)
    
    # 用户输入
    st.markdown("##### ⚙️ 参数调整")
    col1, col2 = st.columns(2)
    growth_input = col1.number_input("预期 EPS 增长率 (%)", value=float(growth_pct), step=1.0, min_value=0.1)
    target_peg = col2.number_input("目标 PEG (传统=1, 费雪修正<1)", value=1.0, step=0.1, min_value=0.1)
    
    # 计算合理价格
    fair_pe = target_peg * growth_input
    fair_price = fair_pe * eps_ttm
    upside = (fair_price / current_price - 1) * 100 if current_price > 0 else 0
    
    # 费雪修正合理价格
    fisher_fair_pe = growth_input + 2 * rf_pct
    fisher_fair_price = fisher_fair_pe * eps_ttm
    fisher_upside = (fisher_fair_price / current_price - 1) * 100 if current_price > 0 else 0
    
    # ===== 估值指标展示 =====
    st.markdown("##### 📊 估值指标")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("当前 PE", f"{current_pe:.1f}")
    m2.metric("传统 PEG", f"{current_peg:.2f}" if current_peg < 100 else "∞", 
              help="<1 低估")
    m3.metric("费雪修正 PEG", f"{fisher_peg:.2f}" if fisher_peg < 100 else "∞",
              help="考虑利率后 <1 低估")
    
    m4, m5, m6 = st.columns(3)
    m4.metric("合理股价 (PEG=1)", f"${fair_price:.2f}", f"{upside:+.1f}%")
    m5.metric("费雪合理股价", f"${fisher_fair_price:.2f}", f"{fisher_upside:+.1f}%")
    m6.metric("合理 PE (费雪)", f"{fisher_fair_pe:.1f}")
    
    # 估值判断
    if current_peg < 1:
        st.success("✅ 传统 PEG < 1，根据 Peter Lynch 标准可能被低估")
    elif current_peg > 2:
        st.warning("⚠️ PEG > 2，估值偏高")
    
    if fisher_peg < 1:
        st.success("✅ 费雪修正 PEG < 1，考虑利率环境后仍被低估")

    st.markdown("---")
    st.markdown("#### 📐 PEG 倒推可视化")
    
    # 可视化：增长率 vs 合理 PE (Implied PE)
    
    growth_range = np.arange(5, 50, 1)
    
    # 传统 PEG=1 时的合理 PE = G
    fair_pe_traditional = growth_range * 1.0 
    
    # 费雪 PEG=1 时的合理 PE = G + 2*rf
    fair_pe_fisher = growth_range + 2 * rf_pct
    
    fig = go.Figure()
    
    # 费雪合理 PE 线
    fig.add_trace(go.Scatter(
        x=growth_range, y=fair_pe_fisher, mode='lines', name='Fisher 合理 PE (PEG=1)',
        line=dict(color='green', width=3)
    ))
    
    # 传统合理 PE 线
    fig.add_trace(go.Scatter(
        x=growth_range, y=fair_pe_traditional, mode='lines', name='传统合理 PE (PEG=1)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # 当前 PE 线
    fig.add_hline(y=current_pe, line_dash="dash", line_color="orange", annotation_text=f"当前 PE {current_pe:.1f}")
    
    # 标记当前增长率点
    # 找到当前 PE 在 Fisher 线上对应的增长率 (反推)
    # PE = G_implied + 2*rf  => G_implied = PE - 2*rf
    implied_growth_fisher = current_pe - 2 * rf_pct
    
    if implied_growth_fisher > 0:
        fig.add_trace(go.Scatter(
            x=[implied_growth_fisher], y=[current_pe], mode='markers', 
            name=f"市场隐含增长率 {implied_growth_fisher:.1f}%",
            marker=dict(size=12, color='red', symbol='x')
        ))
    
    fig.update_layout(
        title=f"PEG 倒推：当前股价隐含增长率约 {implied_growth_fisher:.1f}% (Fisher Model)",
        xaxis_title="预期增长率 (%)",
        yaxis_title="合理 PE 倍数",
        height=400,
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if implied_growth_fisher < growth_pct:
        st.success(f"✅ 市场隐含增长率 ({implied_growth_fisher:.1f}%) < 实际/预期增长率 ({growth_pct:.1f}%)，意味着当前价格未充分计入增长预期 (低估)")
    else:
        st.warning(f"⚠️ 市场隐含增长率 ({implied_growth_fisher:.1f}%) > 实际/预期增长率 ({growth_pct:.1f}%)，意味着当前价格透支了过高的增长预期 (高估)")
    
    # v2.1: PE/PEG 敏感性分析热力图
    st.markdown("#### 🎯 PE/PEG 敏感性分析: 增长率 vs 目标PEG → 合理股价")
    
    growth_sens = [max(5, growth_pct - 10), max(5, growth_pct - 5), growth_pct, growth_pct + 5, growth_pct + 10]
    peg_sens = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    price_matrix = []
    upside_matrix = []
    for g in growth_sens:
        prices_row = []
        upside_row = []
        for p in peg_sens:
            fp = p * g * eps_ttm
            prices_row.append(fp)
            up = (fp / current_price - 1) * 100 if current_price > 0 else 0
            upside_row.append(up)
        price_matrix.append(prices_row)
        upside_matrix.append(upside_row)
    
    # 合理股价热力图
    fig_peg_sense = go.Figure(data=go.Heatmap(
        z=price_matrix,
        x=[f"PEG={p}" for p in peg_sens],
        y=[f"G={g:.0f}%" for g in growth_sens],
        colorscale='RdYlGn',
        texttemplate="$%{z:.0f}",
        colorbar=dict(title="合理股价 ($)")
    ))
    fig_peg_sense.update_layout(
        title=f"敏感性: 增长率/PEG → 合理股价 (当前 ${current_price:.0f})",
        xaxis_title="目标 PEG", yaxis_title="增长率 (%)", height=350
    )
    st.plotly_chart(fig_peg_sense, use_container_width=True)
    
    # 涨跌幅热力图
    fig_upside = go.Figure(data=go.Heatmap(
        z=upside_matrix,
        x=[f"PEG={p}" for p in peg_sens],
        y=[f"G={g:.0f}%" for g in growth_sens],
        colorscale='RdYlGn', zmid=0,
        texttemplate="%{z:+.0f}%",
        colorbar=dict(title="潜在涨跌幅 (%)")
    ))
    fig_upside.update_layout(
        title=f"敏感性: 增长率/PEG → 潜在涨跌幅 (vs 当前 ${current_price:.0f})",
        xaxis_title="目标 PEG", yaxis_title="增长率 (%)", height=350
    )
    st.plotly_chart(fig_upside, use_container_width=True)
    
    # 结论摘要
    all_ups = [v for row in upside_matrix for v in row]
    st.info(f"📊 **敏感性结论**: 合理股价区间 **${min(p for row in price_matrix for p in row):.0f} ~ ${max(p for row in price_matrix for p in row):.0f}**，涨跌幅区间 **{min(all_ups):+.0f}% ~ {max(all_ups):+.0f}%**")


def _render_ev_ebitda(df_single, latest, meta, unit_label):
    """EV/EBITDA 分析 (含行业对比)"""
    st.markdown("#### 💹 EV/EBITDA 分析")
    
    # 获取参数
    market_cap = meta.get('last_market_cap', 0)
    debt = safe_get(latest, 'TotalDebt', 0)
    if debt == 0: debt = safe_get(latest, 'LongTermDebt', 0)
    cash = safe_get(latest, 'CashAndEquivalents', 0)
    if cash == 0: cash = safe_get(latest, 'CashEndOfPeriod', 0)
    
    # EBITDA
    ebitda = safe_get(latest, 'EBITDA_TTM', 0)
    if ebitda == 0: ebitda = safe_get(latest, 'OperatingProfit_TTM', 0)
    if ebitda == 0: ebitda = safe_get(latest, 'OperatingProfit', 0)
    if ebitda == 0: 
        gp = safe_get(latest, 'GrossProfit_TTM', 0) or safe_get(latest, 'GrossProfit', 0)
        opex = safe_get(latest, 'OperatingExpenses_TTM', 0) or safe_get(latest, 'OperatingExpenses', 0)
        ebitda = gp - opex
    
    if market_cap == 0 or ebitda == 0:
        st.warning("⚠️ 缺少市值或 EBITDA 数据")
        return
        
    # 计算 EV (Scaling)
    if ebitda < 10000 and ebitda != 0: 
        scale_input = 1e9
    else:
        scale_input = 1.0
        
    ebitda_dollars = ebitda * scale_input
    debt_dollars = debt * scale_input
    cash_dollars = cash * scale_input
    
    ev = market_cap + debt_dollars - cash_dollars
    ev_ebitda = ev / ebitda_dollars if ebitda_dollars > 0 else 0
    
    # 行业对比 (自动 + 手动)
    # 优先使用数据库中存储的真实 Sector
    meta_sector = meta.get('sector', 'Unknown')
    # 如果数据库没有，尝试尝试从 meta 中获取 (兼容旧逻辑)
    if meta_sector == 'Unknown' or not meta_sector:
         meta_sector = 'Technology' # 默认回退
         
    st.info(f"所属行业识别: {meta_sector}")
    
    bench = get_industry_benchmarks(meta_sector)
    industry_median = bench.get('ev_ebitda', 15.0)
    
    col1, col2 = st.columns(2)
    input_sector_median = col1.number_input("行业中位数 (手动调整)", value=float(industry_median))
    
    # 展示
    m1, m2, m3 = st.columns(3)
    m1.metric("EV/EBITDA (公司)", f"{ev_ebitda:.1f}x")
    m2.metric(f"EV/EBITDA (行业)", f"{input_sector_median:.1f}x")
    diff_pct = (ev_ebitda / input_sector_median - 1) * 100
    m3.metric("相对溢价", f"{diff_pct:+.1f}%", delta_color="inverse") # 越低越好，所以inverse
    
    # 可视化对比
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['EV/EBITDA'], x=[ev_ebitda], orientation='h', name='公司 (Current)', marker_color='blue',
        text=f"{ev_ebitda:.1f}x", textposition='auto'
    ))
    fig.add_trace(go.Bar(
        y=['EV/EBITDA'], x=[input_sector_median], orientation='h', name=f'行业中位 ({meta_sector})', marker_color='gray',
        text=f"{input_sector_median:.1f}x", textposition='auto'
    ))
    fig.update_layout(
        title="公司 EV/EBITDA vs 行业中位数 (越低越好)", 
        height=250, 
        barmode='group',
        xaxis_title="倍数 (x)",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # v2.1: 历史 EV/EBITDA 趋势
    st.markdown("**📈 历史 EV/EBITDA 趋势**")
    hist_data = []
    for _, row in df_single.iterrows():
        ebitda_val = safe_get(row, 'EBITDA_TTM', 0) or safe_get(row, 'OperatingProfit_TTM', 0) or safe_get(row, 'OperatingProfit', 0)
        if ebitda_val > 0:
            ebitda_d = ebitda_val * scale_input
            hist_ev = market_cap + (safe_get(row, 'TotalDebt', 0) or safe_get(row, 'LongTermDebt', 0)) * scale_input - (safe_get(row, 'CashAndEquivalents', 0) or safe_get(row, 'CashEndOfPeriod', 0)) * scale_input
            hist_data.append({"period": f"{row.get('year','')}{row.get('period','')}", "ev_ebitda": hist_ev / ebitda_d})
    
    if len(hist_data) >= 2:
        df_hist = pd.DataFrame(hist_data)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df_hist['period'], y=df_hist['ev_ebitda'], mode='lines+markers', name='EV/EBITDA', line=dict(color='#3B82F6', width=2)))
        fig_hist.add_hline(y=input_sector_median, line_dash="dash", line_color="gray", annotation_text=f"行业中位 {input_sector_median:.1f}x")
        fig_hist.update_layout(title="EV/EBITDA 历史趋势", xaxis_title="期间", yaxis_title="EV/EBITDA (x)", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # v2.1: 隐含合理市值计算
    implied_mc = input_sector_median * ebitda_dollars - debt_dollars + cash_dollars
    implied_diff = (implied_mc / market_cap - 1) * 100 if market_cap > 0 else 0
    st.metric("隐含合理市值 (行业中位EV/EBITDA)", f"{implied_mc/1e9:.1f}B", f"{implied_diff:+.1f}% vs 当前市值")
    
    # v2.1: 敏感性分析
    st.markdown("**🎯 敏感性: EV/EBITDA 倍数 vs EBITDA 变动 → 隐含市值 (B)**")
    mult_range = [ev_ebitda*0.7, ev_ebitda*0.85, ev_ebitda, input_sector_median, ev_ebitda*1.15, ev_ebitda*1.3]
    ebitda_chg = [-20, -10, 0, 10, 20]
    
    mc_matrix = []
    for chg in ebitda_chg:
        row_vals = []
        for m in mult_range:
            adj_ebitda = ebitda_dollars * (1 + chg/100)
            implied = (m * adj_ebitda - debt_dollars + cash_dollars) / 1e9
            row_vals.append(implied)
        mc_matrix.append(row_vals)
    
    fig_s = go.Figure(data=go.Heatmap(
        z=mc_matrix,
        x=[f"{m:.1f}x" for m in mult_range],
        y=[f"EBITDA {c:+d}%" for c in ebitda_chg],
        colorscale='RdYlGn', texttemplate="%{z:.0f}B",
        colorbar=dict(title="隐含市值(B)")
    ))
    fig_s.update_layout(title="EV/EBITDA 敏感性分析", xaxis_title="EV/EBITDA 倍数", yaxis_title="EBITDA 变动", height=350)
    st.plotly_chart(fig_s, use_container_width=True)


def _render_growth_analysis(df_single, unit_label):
    """增长率透视 (全方位: 营收/利润/现金流/债务)"""
    st.markdown("#### 📈 增长率透视 (Growth Perspective)")
    
    if len(df_single) < 4:
        st.warning("数据不足，无法计算增长趋势")
        return
        
    metrics = {
        '业务规模': [('TotalRevenue_TTM', '营收'), ('GrossProfit_TTM', '毛利')],
        '盈利能力': [('NetIncome_TTM', '净利'), ('EPS_TTM', 'EPS')],
        '现金流': [('OperatingCashFlow_TTM', 'OCF'), ('FreeCashFlow_TTM', 'FCF')],
        '资产负债': [('TotalAssets', '总资产'), ('TotalDebt', '总债务'), ('TotalEquity', '股东权益')]
    }
    
    # 汇总数据
    rows = []
    analysis_points = []
    
    for category, items in metrics.items():
        for col, name in items:
            if col in df_single.columns:
                s = df_single[col].dropna()
                if len(s) >= 4:
                    val_new = s.iloc[-1]
                    
                    cagr = 0
                    if len(s) >= 5: 
                        val_old_4y = s.iloc[-5] 
                        if val_old_4y != 0 and val_new != 0:
                            # 能够处理负数的简单CAGR逻辑 (取绝对值计算幅度，保留方向符号)
                            cagr = (abs(val_new) / abs(val_old_4y))**(1/4) - 1
                            if val_new < 0 and val_old_4y > 0: cagr = -abs(cagr)
                            elif val_new > 0 and val_old_4y < 0: cagr = abs(cagr)
                            elif val_new < 0 and val_old_4y < 0: 
                                if val_new > val_old_4y: cagr = abs(cagr) # 亏损收窄
                                else: cagr = -abs(cagr) # 亏损扩大
                    
                    # QoQ
                    qoq = 0
                    if len(s) >= 2 and s.iloc[-2] != 0:
                        qoq = (s.iloc[-1] / s.iloc[-2] - 1)
                    
                    # 记录用于分析
                    if category == '业务规模' and name == '营收':
                        analysis_points.append(f"营收 4年复合增速为 {cagr:.1%}")
                    if category == '盈利能力' and name == '净利':
                        analysis_points.append(f"净利润 4年复合增速为 {cagr:.1%}")
                    
                    rows.append({
                        "类别": category,
                        "指标": name,
                        "最新值": f"{val_new/1e9:.2f}B" if abs(val_new)>1e6 else f"{val_new:.2f}",
                        "QoQ": f"{qoq:+.1%}",
                        "CAGR (4Y)": f"{cagr:+.1%}",
                        "_cagr_raw": cagr
                    })
    
    if rows:
        st.dataframe(pd.DataFrame(rows).drop(columns=['_cagr_raw']), use_container_width=True)
        
    # === 自动文本分析 ===
    st.markdown("##### 📝 增长趋势分析")
    if analysis_points:
        summary = "、".join(analysis_points) + "。"
        
        # 查找主要矛盾
        df_rows = pd.DataFrame(rows)
        rev_growth = df_rows[df_rows['指标']=='营收']['_cagr_raw'].values
        prof_growth = df_rows[df_rows['指标']=='净利']['_cagr_raw'].values
        rev_g = rev_growth[0] if len(rev_growth)>0 else 0
        prof_g = prof_growth[0] if len(prof_growth)>0 else 0
        
        if prof_g > rev_g + 0.05:
            summary += " 净利增速显著快于营收，显示**盈利能力提升**或成本控制有效。"
        elif prof_g < rev_g - 0.05:
            summary += " 净利增速落后于营收，可能面临**毛利下滑**或费用增加压力。"
        else:
            summary += " 营收与利润虽然同步增长，经营质量维持稳定。"
            
        st.info(summary)

    # === 可视化: 历史趋势折线图 ===
    st.markdown("##### 📅 核心指标趋势 (5年)")
    
    metric_keys = ['TotalRevenue_TTM', 'NetIncome_TTM', 'FreeCashFlow_TTM']
    labels = ['营收', '净利', 'FCF']
    colors = ['#3B82F6', '#10B981', '#F59E0B']
    
    fig_ts = go.Figure()
    
    has_data = False
    for k, label, color in zip(metric_keys, labels, colors):
        if k in df_single.columns:
            s_plot = df_single.dropna(subset=[k]).tail(20) # 5年 (4*5=20个季度)
            if not s_plot.empty:
                fig_ts.add_trace(go.Scatter(
                    x=s_plot['report_date'], y=s_plot[k], name=label,
                    mode='lines', line=dict(color=color, width=2)
                ))
                has_data = True
    
    if has_data:
        fig_ts.update_layout(title="核心财务指标趋势 (TTM)", height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ts, use_container_width=True)
        
    # === 增长率对比 (Bar) ===
    df_chart = pd.DataFrame(rows)
    if not df_chart.empty:
        fig = go.Figure()
        df_chart['cagr_val'] = df_chart['_cagr_raw'] * 100
        
        colors_map = {'业务规模': 'blue', '盈利能力': 'green', '现金流': 'orange', '资产负债': 'red'}
        
        for cat in metrics.keys():
            df_sub = df_chart[df_chart['类别'] == cat]
            if not df_sub.empty:
                fig.add_trace(go.Bar(
                    x=df_sub['指标'], y=df_sub['cagr_val'],
                    name=cat, marker_color=colors_map.get(cat, 'gray'),
                    text=[f"{v:.1f}%" for v in df_sub['cagr_val']],
                    textposition='auto'
                ))
                
        fig.update_layout(title="各维度复合增长率对比 (4Y CAGR)", yaxis_title="CAGR (%)", height=300, legend=dict(orientation="h", y=1.2))
        st.plotly_chart(fig, use_container_width=True)




def _render_monte_carlo(df_single, latest, meta, wacc, unit_label):
    """Monte Carlo 模拟 (v2.1 - 多指标选择)"""
    st.markdown("#### 🎲 Monte Carlo 模拟")
    st.caption("使用概率分布模拟估值区间")
    
    fcf = safe_get(latest, 'FreeCashFlow_TTM', 0)
    if fcf == 0:
        fcf = safe_get(latest, 'FreeCashFlow', 0)
    
    if fcf == 0:
        st.warning("需要 FCF 数据")
        return
    
    if fcf < 10000:
        fcf_dollars = fcf * 1e9
    else:
        fcf_dollars = fcf
    
    market_cap = meta.get('last_market_cap', 0)
    
    # v2.1: 用户可选增长率指标
    metric_options = {
        "FCF 同比增长": "FreeCashFlow_TTM_YoY",
        "营收同比增长": "TotalRevenue_TTM_YoY",
        "EPS 同比增长": "EPS_TTM_YoY",
        "OCF 同比增长": "OperatingCashFlow_TTM_YoY"
    }
    
    selected_metric = st.selectbox("📊 选择增长率指标", list(metric_options.keys()), 
                                    help="选择用于模拟的增长率数据源")
    metric_col = metric_options[selected_metric]
    
    # 自动计算历史增长率均值和标准差
    hist_growth_mean = 0.10
    hist_growth_std = 0.05
    source_msg = "默认值 (无足够历史数据)"
    
    if metric_col in df_single.columns:
        growth_series = df_single[metric_col].dropna()
        growth_series = growth_series[(growth_series > -0.5) & (growth_series < 1.0)]
        if len(growth_series) >= 4:
            hist_growth_mean = growth_series.mean()
            hist_growth_std = growth_series.std()
            source_msg = f"✅ 基于 {len(growth_series)} 个季度 {selected_metric} 数据 (Mean={hist_growth_mean:.1%}, Std={hist_growth_std:.1%})"
    
    st.info(f"💡 参数推断: {source_msg}")
    
    # 参数设置
    col1, col2, col3 = st.columns(3)
    growth_mean = col1.number_input("增长率均值 (%)", value=float(hist_growth_mean * 100)) / 100
    growth_std = col2.number_input("增长率标准差 (%)", value=float(hist_growth_std * 100)) / 100
    n_sims = col3.number_input("模拟次数", value=1000, step=100)
    
    if st.button("🎲 运行模拟"):
        np.random.seed(42)
        evs = []
        
        for _ in range(int(n_sims)):
            # 随机增长率 (正态分布)
            growth = np.random.normal(growth_mean, growth_std)
            # 限制范围避免极端值破坏模拟结果
            growth = max(-0.3, min(0.6, growth))
            
            # 计算 EV
            curr = fcf_dollars
            total_pv = 0
            for i in range(1, 6):
                curr = curr * (1 + growth)
                pv = curr / ((1 + wacc) ** i)
                total_pv += pv
            
            term_val = curr * 1.025 / (wacc - 0.025)
            term_pv = term_val / ((1 + wacc) ** 5)
            evs.append((total_pv + term_pv) / 1e9)
        
        evs = np.array(evs)
        
        # 显示结果
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("P10 (保守)", f"{np.percentile(evs, 10):.1f}B")
        col2.metric("P50 (中性)", f"{np.percentile(evs, 50):.1f}B")
        col3.metric("P90 (乐观)", f"{np.percentile(evs, 90):.1f}B")
        col4.metric("平均值", f"{np.mean(evs):.1f}B")
        
        # 与当前市值对比
        upside_p50 = (np.percentile(evs, 50) * 1e9 / market_cap - 1) * 100 if market_cap > 0 else 0
        
        # 结论文本分析
        st.markdown("##### 📝 模拟结果分析")
        if upside_p50 > 15:
            st.success(f"📈 **结论**: Monte Carlo 模拟中位数 (P50) 显示潜在上涨空间 {upside_p50:+.1f}%。即使在较保守情境 (P10) 下，估值为 {np.percentile(evs, 10):.1f}B。")
        elif upside_p50 < -15:
            st.error(f"📉 **结论**: 模拟结果显示当前价格可能高估 (溢价 {abs(upside_p50):.1f}%)。建议关注增长率假设的合理性。")
        else:
            st.info(f"⚖️ **结论**: 模拟结果支持当前估值合理性，差异在正常波动范围内 ({upside_p50:+.1f}%)。")
        
        # 分布图
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=evs, nbinsx=50, name='EV 分布概率', 
            marker_color='rgba(100, 149, 237, 0.7)', opacity=0.7
        ))
        
        # 垂直辅助线
        fig.add_vline(x=market_cap/1e9, line_dash="dash", line_color="orange", 
                      annotation_text=f"当前市值 {market_cap/1e9:.1f}B")
        
        fig.add_vline(x=np.percentile(evs, 50), line_dash="solid", line_color="green",
                     annotation_text="P50 (中位)")
                     
        fig.update_layout(
            title=f"企业价值概率分布 (基于 {int(n_sims)} 次随机模拟)", 
            xaxis_title="企业价值 (Billion USD)",
            yaxis_title="频次",
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_profitability_analysis(df_single, unit_label):
    """ROIC/ROA/ROE 分析 (含行业对比)"""
    st.markdown("#### 📉 盈利能力透视 (ROIC/ROA/ROE)")
    
    if len(df_single) < 2:
        st.warning("数据不足")
        return
    
    latest = df_single.iloc[-1]
    
    # 行业对比
    ticker = df_single.iloc[0].get('ticker', '')
    meta = get_company_meta(ticker)
    sector = meta.get('sector', 'General')
    st.info(f"所属行业: **{sector}** | Ticker: {ticker}")
    
    bench = get_industry_benchmarks(sector)
    
    # 辅助函数：安全获取数值
    def safe_val(row, key):
        val = row.get(key, 0)
        return val if val is not None and not (isinstance(val, float) and np.isnan(val)) else 0
    
    # 计算指标
    net_income = safe_val(latest, 'NetIncome_TTM')
    total_assets = safe_val(latest, 'TotalAssets')
    total_equity = safe_val(latest, 'TotalEquity')
    total_debt = safe_val(latest, 'TotalDebt')
    invested_capital = total_equity + total_debt
    
    roa = (net_income / total_assets * 100) if total_assets > 0 else 0
    roe = (net_income / total_equity * 100) if total_equity > 0 else 0
    roic = (net_income / invested_capital * 100) if invested_capital > 0 else 0
    
    # 行业基准
    ind_roe = bench.get('roe', 15.0)
    ind_roa = bench.get('roa', 5.0)
    ind_roic = bench.get('roic', 10.0)
    
    # 指标卡片
    c1, c2, c3 = st.columns(3)
    c1.metric("ROE (净资产回报)", f"{roe:.1f}%", f"行业 {ind_roe}%", delta_color="normal")
    c2.metric("ROA (总资产回报)", f"{roa:.1f}%", f"行业 {ind_roa}%", delta_color="normal")
    c3.metric("ROIC (投入资本回报)", f"{roic:.1f}%", f"行业 {ind_roic}%", delta_color="normal")
    
    # 杜邦分析
    revenue = safe_val(latest, 'TotalRevenue_TTM')
    npm = (net_income / revenue * 100) if revenue > 0 else 0
    asset_turnover = revenue / total_assets if total_assets > 0 else 0
    equity_multiplier = total_assets / total_equity if total_equity > 0 else 0
    
    st.info(f"💡 杜邦拆解: ROE {roe:.1f}% ≈ 净利率 {npm:.1f}% × 资产周转率 {asset_turnover:.2f} × 权益乘数 {equity_multiplier:.2f}")
    
    # 可视化: 公司 vs 行业
    metric_names = ['ROE', 'ROA', 'ROIC']
    company_vals = [roe, roa, roic]
    industry_vals = [ind_roe, ind_roa, ind_roic]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metric_names, y=company_vals, name='公司', marker_color='#3B82F6', text=[f"{v:.1f}%" for v in company_vals], textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=metric_names, y=industry_vals, name=f'行业 ({sector})', marker_color='#9CA3AF', text=[f"{v:.1f}%" for v in industry_vals], textposition='auto'
    ))
    
    fig.update_layout(
        title="盈利能力对比: 公司 vs 行业",
        yaxis_title="百分比 (%)",
        barmode='group',
        height=300,
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 历史趋势图
    st.markdown("##### 📅 历史趋势")
    fig2 = go.Figure()
    
    # 添加 ROE 趋势
    if 'NetIncome_TTM' in df_single.columns and 'TotalEquity' in df_single.columns:
        df_plot = df_single.dropna(subset=['NetIncome_TTM', 'TotalEquity']).tail(12)
        if not df_plot.empty:
            roe_series = df_plot['NetIncome_TTM'] / df_plot['TotalEquity'] * 100
            fig2.add_trace(go.Scatter(
                x=df_plot['report_date'], y=roe_series, mode='lines+markers', name='ROE 历史',
                line=dict(width=2)
            ))
            
    # 添加 ROIC 趋势
    if 'NetIncome_TTM' in df_single.columns and 'TotalDebt' in df_single.columns:
        df_plot_roic = df_single.dropna(subset=['NetIncome_TTM', 'TotalEquity', 'TotalDebt']).tail(12)
        if not df_plot_roic.empty:
            capital = df_plot_roic['TotalEquity'] + df_plot_roic['TotalDebt']
            roic_series = df_plot_roic['NetIncome_TTM'] / capital * 100
            fig2.add_trace(go.Scatter(
                x=df_plot_roic['report_date'], y=roic_series, mode='lines+markers', name='ROIC 历史',
                line=dict(dash='dash', width=2)
            ))
            
    fig2.update_layout(title="盈利能力历史趋势 (ROE vs ROIC)", yaxis_title="百分比 (%)", height=300, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)
    
    # v2.1: ROIC vs WACC 价值创造分析
    st.markdown("##### 💎 价值创造能力 (ROIC vs WACC)")
    
    # 获取 WACC (从 session state)
    wacc_pct = st.session_state.get('wacc', 0.10) * 100 if 'wacc' in st.session_state else 10.0
    wacc_input = st.number_input("WACC (%)", value=float(wacc_pct), step=0.5, key="roic_wacc")
    
    spread = roic - wacc_input
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("ROIC", f"{roic:.1f}%")
    sc2.metric("WACC", f"{wacc_input:.1f}%")
    sc3.metric("超额收益 (ROIC-WACC)", f"{spread:+.1f}%", "创造价值 ✅" if spread > 0 else "毁灭价值 ❌")
    
    if spread > 3:
        st.success(f"🌟 **强价值创造**: ROIC 超过 WACC {spread:.1f}个百分点，公司每投入1元资本产生超越资本成本的回报。")
    elif spread > 0:
        st.info(f"✅ **正向价值创造**: ROIC 略高于 WACC {spread:.1f}个百分点，但需关注可持续性。")
    else:
        st.error(f"❌ **价值毁灭**: ROIC 低于 WACC {abs(spread):.1f}个百分点, 投入的资本回报低于资本成本。")
    
    # v2.1: 杠杆可持续性分析
    st.markdown("##### ⚖️ 杠杆可持续性")
    de_ratio = total_debt / total_equity if total_equity > 0 else 0
    interest_exp = abs(safe_val(latest, 'InterestExpense_TTM') or safe_val(latest, 'InterestExpense') or 0)
    op_income = safe_val(latest, 'OperatingProfit_TTM') or safe_val(latest, 'OperatingProfit') or 0
    interest_coverage = op_income / interest_exp if interest_exp > 0 else float('inf')
    
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("D/E 比率", f"{de_ratio:.2f}", "健康" if de_ratio < 1 else "偏高", delta_color="inverse")
    lc2.metric("利息覆盖倍数", f"{interest_coverage:.1f}x" if interest_coverage < 100 else "N/A", "安全" if interest_coverage > 3 else "风险")
    lc3.metric("权益乘数", f"{equity_multiplier:.2f}", "适度杠杆" if equity_multiplier < 3 else "高杠杆")
    
    if de_ratio > 1.5 and roe > ind_roe:
        st.warning("⚠️ **注意**: 高 ROE 可能主要由高杠杆驱动（D/E > 1.5x），盈利质量需关注资产周转率和净利率趋势。")
    
    # v2.1: 估值弹性分析
    st.markdown("##### 📊 估值弹性分析 (ROIC 变动 → 隐含价值变化)")
    
    roic_range = [roic*0.7, roic*0.85, roic, roic*1.15, roic*1.3]
    if invested_capital > 0 and net_income > 0:
        fig_elast = go.Figure()
        implied_premiums = [(r - wacc_input) / wacc_input * 100 for r in roic_range]
        fig_elast.add_trace(go.Bar(
            x=[f"ROIC={r:.1f}%" for r in roic_range],
            y=implied_premiums,
            marker_color=['#EF4444' if p < 0 else '#10B981' for p in implied_premiums],
            text=[f"{p:+.0f}%" for p in implied_premiums],
            textposition='auto'
        ))
        fig_elast.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_elast.update_layout(title="ROIC 变动 → 超额收益变化", xaxis_title="ROIC 情景", yaxis_title="超额收益/WACC (%)", height=300)
        st.plotly_chart(fig_elast, use_container_width=True)
