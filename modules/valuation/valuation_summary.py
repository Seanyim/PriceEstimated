# modules/valuation/valuation_summary.py
# 估值总结模块 v2.5.2
# 交互式可视化仪表盘 — 集成大师分析 + 敏感性分析 + 综合信号
# 保留原有 Markdown 报告下载功能 (非破坏性)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
from modules.core.calculator import process_financial_data
from modules.core.db import get_company_meta, get_market_history
from modules.data.industry_data import get_industry_benchmarks
from modules.valuation.valuation_advanced import safe_get
from modules.valuation.master_analysis import (
    compute_master_scores, MASTER_DEFINITIONS
)
from modules.valuation.qg_pro_model import compute_qg_pro_score


# ============================================================
# 1. 核心数据提取器
# ============================================================

def _extract_valuation_data(ticker, df_raw, unit_label, wacc, rf):
    """
    从财务数据和市场数据中提取所有估值相关指标。
    返回一个字典，供各可视化区块使用。
    """
    meta = get_company_meta(ticker)
    company_name = meta.get('name', ticker)
    region = meta.get('region', 'Unknown')
    sector = meta.get('sector', 'Unknown')
    market_cap = meta.get('last_market_cap', 0)

    _, df_single = process_financial_data(df_raw)
    df_price = get_market_history(ticker)

    current_price = 0
    if not df_price.empty:
        current_price = df_price.iloc[-1].get('close', 0) or 0

    eps_val = 0
    pe_ttm = 0
    latest = None

    if not df_single.empty:
        latest = df_single.iloc[-1]
        eps_val = safe_get(latest, 'EPS_TTM', 0)
        if eps_val > 0 and current_price > 0:
            pe_ttm = current_price / eps_val

    return {
        'meta': meta,
        'company_name': company_name,
        'region': region,
        'sector': sector,
        'market_cap': market_cap,
        'df_single': df_single,
        'df_price': df_price,
        'current_price': current_price,
        'eps_val': eps_val,
        'pe_ttm': pe_ttm,
        'latest': latest,
        'wacc': wacc,
        'rf': rf,
        'unit_label': unit_label,
    }


# ============================================================
# 2. 正推估值计算引擎 (复用 dashboard 逻辑)
# ============================================================

def _compute_forward_valuations(data: dict) -> Dict:
    """计算各模型正推估值结果"""
    results = {}
    latest = data['latest']
    if latest is None:
        return results

    eps_ttm = data['eps_val']
    pe_ttm = data['pe_ttm']
    current_price = data['current_price']
    market_cap = data['market_cap']
    wacc = data['wacc']
    unit_label = data['unit_label']
    df_single = data['df_single']
    df_price = data['df_price']

    # FCF 数据
    fcf_ttm = safe_get(latest, 'FreeCashFlow_TTM', 0)
    if fcf_ttm == 0:
        ocf = safe_get(latest, 'OperatingCashFlow_TTM', 0)
        capex = abs(safe_get(latest, 'CapEx', 0))
        if ocf != 0:
            fcf_ttm = ocf - capex

    # 增长率
    eps_yoy = safe_get(latest, 'EPS_TTM_YoY', None)
    rev_yoy = safe_get(latest, 'TotalRevenue_TTM_YoY', None)
    growth_rate = 0.10
    if eps_yoy is not None and eps_yoy > 0:
        growth_rate = eps_yoy
    elif rev_yoy is not None and rev_yoy > 0:
        growth_rate = rev_yoy

    growth_pct = growth_rate * 100

    # --- PE Band 估值 ---
    if pe_ttm > 0 and eps_ttm > 0:
        df_single_sorted = df_single.sort_values('report_date')
        df_price_sorted = df_price.sort_values('date') if not df_price.empty else pd.DataFrame()

        if not df_price_sorted.empty and 'report_date' in df_single_sorted.columns:
            df_single_sorted['report_date'] = pd.to_datetime(df_single_sorted['report_date'])
            df_price_sorted['date'] = pd.to_datetime(df_price_sorted['date'])

            df_m = pd.merge_asof(
                df_single_sorted, df_price_sorted,
                left_on='report_date', right_on='date', direction='backward'
            )
            df_m['PE_TTM'] = df_m['close'] / df_m['EPS_TTM']
            valid_pe = df_m[(df_m['PE_TTM'] > 0) & (df_m['PE_TTM'] < 200)]

            if not valid_pe.empty:
                pe_20 = valid_pe['PE_TTM'].quantile(0.20)
                pe_50 = valid_pe['PE_TTM'].quantile(0.50)
                pe_80 = valid_pe['PE_TTM'].quantile(0.80)

                results['PE Band (20%)'] = {
                    'fair_price': pe_20 * eps_ttm,
                    'method': f"PE {pe_20:.1f}x × EPS {eps_ttm:.2f}",
                    'model': 'PE'
                }
                results['PE Band (50%)'] = {
                    'fair_price': pe_50 * eps_ttm,
                    'method': f"PE {pe_50:.1f}x × EPS {eps_ttm:.2f}",
                    'model': 'PE'
                }
                results['PE Band (80%)'] = {
                    'fair_price': pe_80 * eps_ttm,
                    'method': f"PE {pe_80:.1f}x × EPS {eps_ttm:.2f}",
                    'model': 'PE'
                }

    # --- PEG=1 合理价格 ---
    if eps_ttm > 0 and growth_pct > 0:
        peg1_fair = growth_pct * eps_ttm
        results['PEG=1 合理价'] = {
            'fair_price': peg1_fair,
            'method': f"PE={growth_pct:.0f}x × EPS {eps_ttm:.2f}",
            'model': 'PEG'
        }

    # --- DCF 正推 ---
    perp_rate = 0.025
    if fcf_ttm != 0 and wacc > perp_rate:
        g_rate = min(growth_rate, 0.50)
        curr_fcf = fcf_ttm
        total_pv = 0
        for i in range(1, 6):
            curr_fcf *= (1 + g_rate)
            total_pv += curr_fcf / ((1 + wacc) ** i)

        term_val = curr_fcf * (1 + perp_rate) / (wacc - perp_rate)
        term_pv = term_val / ((1 + wacc) ** 5)
        dcf_ev = total_pv + term_pv

        if market_cap > 0 and current_price > 0:
            shares = market_cap / current_price
            dcf_ev_d = dcf_ev * 1e9 if dcf_ev < 10000 else dcf_ev
            dcf_per_share = dcf_ev_d / shares
            results['DCF 内在价值'] = {
                'fair_price': dcf_per_share,
                'method': f"FCF={fcf_ttm:.1f}{unit_label}, g={g_rate:.1%}, WACC={wacc:.1%}",
                'model': 'DCF'
            }

    # --- EV/EBITDA ---
    ebitda = safe_get(latest, 'EBITDA_TTM', 0) or safe_get(latest, 'OperatingProfit_TTM', 0)
    if ebitda > 0 and market_cap > 0:
        bench = get_industry_benchmarks(data['sector'])
        ind_ev = bench.get('ev_ebitda', 15.0)
        scale = 1e9 if ebitda < 10000 else 1.0
        ebitda_d = ebitda * scale
        debt_d = (safe_get(latest, 'TotalDebt', 0) or safe_get(latest, 'LongTermDebt', 0)) * scale
        cash_d = (safe_get(latest, 'CashAndEquivalents', 0) or safe_get(latest, 'CashEndOfPeriod', 0)) * scale
        implied_mc = ind_ev * ebitda_d - debt_d + cash_d
        if current_price > 0:
            shares = market_cap / current_price
            ev_per_share = implied_mc / shares
            results['EV/EBITDA 行业对标'] = {
                'fair_price': ev_per_share,
                'method': f"行业中位 {ind_ev:.1f}x × EBITDA {ebitda:.1f}",
                'model': 'EV/EBITDA'
            }

    return results


# ============================================================
# 3. 可视化区块
# ============================================================

def _render_executive_metrics(data: dict, forward_results: dict, master_scores: Optional[dict]):
    """区块 1: 核心指标卡片"""
    current_price = data['current_price']
    pe_ttm = data['pe_ttm']

    # 计算综合内在价值
    intrinsic_value = 0
    margin = 0
    if forward_results and current_price > 0:
        all_fairs = [info['fair_price'] for info in forward_results.values()]
        intrinsic_value = np.median(all_fairs)
        margin = (intrinsic_value / current_price - 1) * 100

    # 大师综合分 (如果可用)
    master_avg = None
    if master_scores:
        available = [master_scores[k]['score'] for k in master_scores if master_scores[k]['available']]
        if available:
            master_avg = np.mean(available)

    # 评级
    if margin > 20:
        rating = "⭐⭐⭐⭐⭐ 严重低估"
    elif margin > 5:
        rating = "⭐⭐⭐⭐ 轻度低估"
    elif margin > -10:
        rating = "⭐⭐⭐ 合理估值"
    elif margin > -25:
        rating = "⭐⭐ 轻度高估"
    else:
        rating = "⭐ 显著高估"

    st.markdown("### 🎯 核心估值指标")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("当前股价", f"${current_price:.2f}" if current_price > 0 else "N/A")
    c2.metric(
        "综合内在价值",
        f"${intrinsic_value:.2f}" if intrinsic_value > 0 else "N/A"
    )
    c3.metric(
        "安全边际",
        f"{margin:+.1f}%" if forward_results else "N/A",
        "低估" if margin > 0 else "高估",
        delta_color="normal" if margin > 0 else "inverse"
    )
    c4.metric("PE (TTM)", f"{pe_ttm:.1f}x" if pe_ttm > 0 else "N/A")
    c5.metric(
        "大师综合评分",
        f"{master_avg:.0f}/100" if master_avg is not None else "N/A",
        "优" if master_avg and master_avg >= 70 else ("中" if master_avg and master_avg >= 40 else "弱") if master_avg else None
    )

    # 评级横幅
    if margin > 10:
        st.success(f"📊 **综合评级: {rating}** — 安全边际 {margin:+.1f}%，多模型估值中位数 ${intrinsic_value:.2f}")
    elif margin < -10:
        st.error(f"📊 **综合评级: {rating}** — 溢价 {abs(margin):.1f}%，需关注增长率是否支撑当前估值")
    else:
        st.info(f"📊 **综合评级: {rating}** — 当前定价基本合理")

    return intrinsic_value, margin


def _render_valuation_range_chart(data: dict, forward_results: dict, intrinsic_value: float):
    """区块 2: 估值区间对比图"""
    current_price = data['current_price']

    if not forward_results or current_price <= 0:
        return

    st.markdown("### 📊 各模型估值对比")

    names = list(forward_results.keys())
    prices = [info['fair_price'] for info in forward_results.values()]
    colors = ['#10B981' if fp > current_price * 1.05 else '#EF4444' if fp < current_price * 0.95 else '#F59E0B'
              for fp in prices]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names, y=prices,
        marker_color=colors,
        text=[f"${fp:.0f}" for fp in prices],
        textposition='auto',
        name="合理股价"
    ))

    # 当前股价线
    fig.add_hline(
        y=current_price, line_dash="dash", line_color="orange", line_width=2,
        annotation_text=f"当前 ${current_price:.0f}"
    )

    # 内在价值线
    if intrinsic_value > 0:
        fig.add_hline(
            y=intrinsic_value, line_dash="solid", line_color="#10B981", line_width=2,
            annotation_text=f"中位内在价值 ${intrinsic_value:.0f}"
        )

    fig.update_layout(
        yaxis_title="股价 ($)",
        height=380,
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_master_mini_panel(master_scores: Optional[dict]):
    """区块 3: 大师分析迷你面板 (雷达图 + 风格评分)"""
    if not master_scores:
        st.info("💡 请先查看「大师分析」Tab 生成评分数据")
        return

    st.markdown("### 🧠 大师分析概览")

    master_order = ["Buffett", "Munger", "Lynch", "Graham", "Greenblatt",
                    "Fisher", "Templeton", "Dalio", "Soros"]

    # 两栏布局：左雷达 右评分
    col_radar, col_scores = st.columns([3, 2])

    with col_radar:
        categories = []
        values = []
        for key in master_order:
            if key in master_scores:
                defn = MASTER_DEFINITIONS[key]
                categories.append(f"{defn['icon']} {defn['name_cn']}")
                val = master_scores[key]['score']
                values.append(float(val) if val is not None and not np.isnan(val) else 50.0)

        if values:
            avg_score = np.mean(values)
            if avg_score >= 70:
                fill_color, line_color = "rgba(46, 204, 113, 0.25)", "#2ECC71"
            elif avg_score >= 40:
                fill_color, line_color = "rgba(241, 196, 15, 0.25)", "#F1C40F"
            else:
                fill_color, line_color = "rgba(231, 76, 60, 0.25)", "#E74C3C"

            cat_closed = categories + [categories[0]]
            val_closed = values + [values[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=val_closed, theta=cat_closed,
                fill='toself', fillcolor=fill_color,
                line=dict(color=line_color, width=2),
                marker=dict(size=6, color=line_color),
                hovertemplate="%{theta}<br>评分: %{r:.0f}<extra></extra>"
            ))
            fig.add_trace(go.Scatterpolar(
                r=[60] * len(cat_closed), theta=cat_closed,
                line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dash"),
                hoverinfo='skip', showlegend=False
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    tickvals=[20, 40, 60, 80, 100]),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=False,
                margin=dict(l=60, r=60, t=20, b=20),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_scores:
        # 四维风格评分
        value_masters = ["Buffett", "Munger", "Graham", "Greenblatt", "Templeton"]
        growth_masters = ["Lynch", "Fisher"]
        trend_masters = ["Soros"]
        defense_masters = ["Dalio"]

        def _dim_avg(keys):
            vals = [master_scores[k]['score'] for k in keys if k in master_scores and master_scores[k]['available']]
            return np.mean(vals) if vals else 50.0

        value_avg = _dim_avg(value_masters)
        growth_avg = _dim_avg(growth_masters)
        trend_avg = _dim_avg(trend_masters)
        defense_avg = _dim_avg(defense_masters)

        st.metric("🏰 价值维度", f"{value_avg:.0f}/100",
                  help="Buffett + Munger + Graham + Greenblatt + Templeton")
        st.metric("🚀 成长维度", f"{growth_avg:.0f}/100",
                  help="Lynch + Fisher")
        st.metric("⚡ 趋势维度", f"{trend_avg:.0f}/100",
                  help="Soros")
        st.metric("🛡️ 防御维度", f"{defense_avg:.0f}/100",
                  help="Dalio")

        # 主导风格判断
        style_map = {
            "价值投资": value_avg,
            "成长投资": growth_avg,
            "趋势动量": trend_avg,
            "防御宏观": defense_avg,
        }
        dominant = max(style_map, key=style_map.get)
        st.caption(f"🧭 主导风格: **{dominant}** ({style_map[dominant]:.0f})")


def _render_dcf_sensitivity(data: dict):
    """区块 4: DCF 敏感性热力图 — WACC vs 增长率 → 每股内在价值"""
    latest = data['latest']
    if latest is None:
        return

    wacc = data['wacc']
    current_price = data['current_price']
    market_cap = data['market_cap']
    unit_label = data['unit_label']

    fcf = safe_get(latest, 'FreeCashFlow_TTM', 0)
    if fcf == 0:
        ocf = safe_get(latest, 'OperatingCashFlow_TTM', 0)
        capex = abs(safe_get(latest, 'CapEx', 0))
        if ocf != 0:
            fcf = ocf - capex

    if fcf <= 0 or market_cap <= 0 or current_price <= 0:
        return

    st.markdown("### 🎯 DCF 敏感性分析")
    st.caption("WACC 与增长率变动对每股内在价值的影响")

    shares = market_cap / current_price
    fcf_dollars = fcf * 1e9 if fcf < 10000 else fcf
    perp_rate = 0.025

    # 参数范围
    wacc_range = [max(0.04, wacc - 0.02), max(0.05, wacc - 0.01), wacc,
                  wacc + 0.01, wacc + 0.02]
    growth_range = [0.03, 0.06, 0.10, 0.15, 0.20, 0.25]

    matrix = []
    for g in growth_range:
        row = []
        for w in wacc_range:
            if w <= perp_rate:
                row.append(None)
                continue
            curr = fcf_dollars
            total_pv = 0
            for i in range(1, 6):
                curr *= (1 + g)
                total_pv += curr / ((1 + w) ** i)
            tv = curr * (1 + perp_rate) / (w - perp_rate)
            total_pv += tv / ((1 + w) ** 5)
            per_share = total_pv / shares
            row.append(per_share)
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"WACC {w:.1%}" for w in wacc_range],
        y=[f"g={g:.0%}" for g in growth_range],
        colorscale='RdYlGn',
        texttemplate="$%{z:.0f}",
        colorbar=dict(title="每股价值($)")
    ))

    if current_price > 0:
        fig.update_layout(
            title=f"DCF 敏感性矩阵 (当前 ${current_price:.0f})",
        )

    fig.update_layout(
        xaxis_title="WACC", yaxis_title="5年增长率",
        height=380, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pe_sensitivity(data: dict):
    """区块 5: PE 敏感性热力图 — PE 倍数 vs EPS → 合理股价"""
    eps_val = data['eps_val']
    pe_ttm = data['pe_ttm']
    current_price = data['current_price']

    if eps_val <= 0 or pe_ttm <= 0 or current_price <= 0:
        return

    st.markdown("### 📈 PE 敏感性分析")
    st.caption("PE 倍数与 EPS 变动对合理股价的影响")

    # 参数范围
    pe_range = [max(5, pe_ttm * 0.5), pe_ttm * 0.75, pe_ttm, pe_ttm * 1.25, pe_ttm * 1.5]
    eps_changes = [-20, -10, 0, 10, 20, 30]

    matrix = []
    for eps_chg in eps_changes:
        row = []
        adj_eps = eps_val * (1 + eps_chg / 100)
        for pe in pe_range:
            fair = pe * adj_eps
            row.append(fair)
        matrix.append(row)

    # 涨跌幅矩阵
    upside_matrix = []
    for row in matrix:
        upside_matrix.append([(v / current_price - 1) * 100 if current_price > 0 else 0 for v in row])

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"PE {pe:.0f}x" for pe in pe_range],
            y=[f"EPS {c:+d}%" for c in eps_changes],
            colorscale='RdYlGn',
            texttemplate="$%{z:.0f}",
            colorbar=dict(title="合理股价($)")
        ))
        fig.update_layout(
            title="合理股价矩阵",
            xaxis_title="PE 倍数", yaxis_title="EPS 变动",
            height=350, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(data=go.Heatmap(
            z=upside_matrix,
            x=[f"PE {pe:.0f}x" for pe in pe_range],
            y=[f"EPS {c:+d}%" for c in eps_changes],
            colorscale='RdYlGn', zmid=0,
            texttemplate="%{z:+.0f}%",
            colorbar=dict(title="潜在涨跌幅(%)")
        ))
        fig2.update_layout(
            title=f"涨跌幅矩阵 (vs ${current_price:.0f})",
            xaxis_title="PE 倍数", yaxis_title="EPS 变动",
            height=350, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_signal_panel(data: dict, forward_results: dict,
                         intrinsic_value: float, margin: float,
                         master_scores: Optional[dict],
                         qg_scores: Optional[dict] = None):
    """区块 6: 综合信号面板"""
    current_price = data['current_price']
    latest = data['latest']

    if not forward_results or current_price <= 0 or latest is None:
        return

    st.markdown("### 💡 综合信号面板")

    # 信号1: 正推模型共识
    bullish = sum(1 for info in forward_results.values() if info['fair_price'] > current_price * 1.1)
    bearish = sum(1 for info in forward_results.values() if info['fair_price'] < current_price * 0.9)
    total = len(forward_results)
    bull_ratio = bullish / total if total > 0 else 0

    # 信号2: 盈利质量
    ni = safe_get(latest, 'NetIncome_TTM', 0)
    ocf = safe_get(latest, 'OperatingCashFlow_TTM', 0)
    quality_score = 50
    if ni > 0 and ocf > 0:
        quality_ratio = min(ocf / ni, 2.0)
        quality_score = min(quality_ratio * 50, 100)

    # 信号3: 大师综合
    master_avg = None
    if master_scores:
        available = [master_scores[k]['score'] for k in master_scores if master_scores[k]['available']]
        if available:
            master_avg = np.mean(available)
            
    # 信号4: QG-Pro
    qg_score = None
    if qg_scores and qg_scores.get('available'):
        qg_score = qg_scores.get('score')

    signal_data = {
        "维度": ["模型共识", "安全边际", "盈利质量", "大师评分", "QG-Pro 质量", "综合评级"],
        "结果": [],
        "信号": [],
    }

    # 模型共识
    signal_data["结果"].append(f"{bullish}/{total} 看涨")
    signal_data["信号"].append(
        "✅ 多数看涨" if bull_ratio >= 0.6 else
        "⚠️ 多数看跌" if bull_ratio <= 0.3 else
        "📊 信号分歧"
    )

    # 安全边际
    signal_data["结果"].append(f"{margin:+.1f}%")
    signal_data["信号"].append(
        "✅ 充足安全边际" if margin > 20 else
        "📊 轻微低估" if margin > 0 else
        "📊 轻微高估" if margin > -15 else
        "⚠️ 显著高估"
    )

    # 盈利质量
    signal_data["结果"].append(f"得分 {quality_score:.0f}/100")
    signal_data["信号"].append(
        "✅ 优良" if quality_score >= 70 else
        "📊 一般" if quality_score >= 40 else
        "⚠️ 较差"
    )

    # 大师评分
    if master_avg is not None:
        signal_data["结果"].append(f"{master_avg:.0f}/100")
        signal_data["信号"].append(
            "✅ 大师认可" if master_avg >= 65 else
            "📊 评价中性" if master_avg >= 40 else
            "⚠️ 多项偏弱"
        )
    else:
        signal_data["结果"].append("N/A")
        signal_data["信号"].append("—")

    # QG-Pro 质量
    if qg_score is not None:
        signal_data["结果"].append(f"{qg_score:.0f}/100")
        signal_data["信号"].append(
            "✅ 增长优质" if qg_score >= 70 else
            "📊 质量一般" if qg_score >= 40 else
            "⚠️ 增长恶化"
        )
    else:
        signal_data["结果"].append("N/A")
        signal_data["信号"].append("—")

    # 综合评级
    scores = [
        50 + margin if forward_results else 50,
        quality_score,
        master_avg if master_avg else 50,
        qg_score if qg_score else 50,
    ]
    total_score = np.mean(scores)
    if total_score >= 70:
        overall = "⭐⭐⭐⭐⭐ 强烈看好"
    elif total_score >= 55:
        overall = "⭐⭐⭐⭐ 看好"
    elif total_score >= 45:
        overall = "⭐⭐⭐ 中性"
    elif total_score >= 35:
        overall = "⭐⭐ 谨慎"
    else:
        overall = "⭐ 回避"

    signal_data["结果"].append(f"综合 {total_score:.0f}/100")
    signal_data["信号"].append(overall)

    st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)


def _render_qg_pro_panel(qg_scores: Optional[dict]):
    """区块 3.5: QG-Pro 机构实盘级多因子质量模型"""
    if not qg_scores:
        return

    st.markdown("### 🎯 QG-Pro 增长质量模型 (机构级)")
    st.caption("综合评估盈利质量、下行风险及基本面加速动能 (基于绝对阈值评价)")

    score = qg_scores.get('score', 50)
    factors = qg_scores.get('factors', {})
    available = qg_scores.get('available', False)
    dim_scores = qg_scores.get('dim_scores', {})

    if not available:
        st.info("⚠️ 数据不足，无法计算 QG-Pro 因子，采用默认中性分数 (50)")

    col1, col2 = st.columns([2, 2])

    with col1:
        if score >= 70:
            badge = "优异"
            delta_color = "normal"
        elif score >= 40:
            badge = "一般"
            delta_color = "off"
        else:
            badge = "偏弱"
            delta_color = "inverse"

        st.metric(
            label="QG-Pro 综合得分",
            value=f"{score:.0f}/100",
            delta=badge,
            delta_color=delta_color
        )
        
        st.markdown("**底层驱动因子**:")
        for fname, fval in factors.items():
            if not fname.startswith("⚠️"):
                st.write(f"- **{fname}**: {fval}")

    with col2:
        if dim_scores:
            categories = ['增长加速 (G_adj)', '下行控制 (S_down)', '防雷安全 (D_risk)', '盈余质量 (CF_quality)']
            values = [
                dim_scores.get('G_adj', 50),
                dim_scores.get('S_down', 50),
                dim_scores.get('D_risk', 50),
                dim_scores.get('CF_quality', 50)
            ]
            
            # 闭合曲线
            categories = categories + [categories[0]]
            values = values + [values[0]]
            
            if score >= 70:
                fill_color, line_color = "rgba(46, 204, 113, 0.25)", "#2ECC71"
            elif score >= 40:
                fill_color, line_color = "rgba(241, 196, 15, 0.25)", "#F1C40F"
            else:
                fill_color, line_color = "rgba(231, 76, 60, 0.25)", "#E74C3C"
                
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories,
                fill='toself', fillcolor=fill_color,
                line=dict(color=line_color, width=2),
                marker=dict(size=6, color=line_color),
                hovertemplate="%{theta}<br>得分: %{r:.0f}/100<extra></extra>"
            ))
            fig.add_trace(go.Scatterpolar(
                r=[60]*5, theta=categories,
                line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dash"),
                hoverinfo='skip', showlegend=False
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickvals=[20,40,60,80,100]),
                    bgcolor="rgba(0,0,0,0)"
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=20, b=20),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)



# ============================================================
# 4. 原版 Markdown 报告 (保留，非破坏性)
# ============================================================

def _build_summary_markdown(ticker, df_raw, unit_label, wacc, rf):
    """
    基于财务数据和估值参数，生成结构化的估值总结 Markdown 文本。
    v2.5 — 增加内在价值分析、增长可持续性、盈利质量、资本结构评估
    """
    meta = get_company_meta(ticker)
    company_name = meta.get('name', ticker)
    region = meta.get('region', 'Unknown')
    sector = meta.get('sector', 'Unknown')
    market_cap = meta.get('last_market_cap', 0)

    _, df_single = process_financial_data(df_raw)

    # 获取市场数据
    df_price = get_market_history(ticker)
    current_price = 0
    pe_ttm = 0
    eps_val = 0

    if not df_price.empty:
        current_price = df_price.iloc[-1].get('close', 0) or 0

    if not df_single.empty and current_price > 0:
        eps_val = safe_get(df_single.iloc[-1], 'EPS_TTM', 0)
        if eps_val > 0:
            pe_ttm = current_price / eps_val

    lines = []

    # 0. 执行摘要
    lines.append(f"# {company_name} ({ticker}) — 估值总结报告\n")
    lines.append(f"**地区**: {region} | **行业**: {sector} | **数据单位**: {unit_label}\n")
    lines.append("## 📋 执行摘要\n")

    exec_points = []
    if pe_ttm > 0:
        if pe_ttm < 15:
            exec_points.append(f"- **估值状态**: PE {pe_ttm:.1f}x，处于偏低水平")
        elif pe_ttm > 35:
            exec_points.append(f"- **估值状态**: PE {pe_ttm:.1f}x，处于偏高水平")
        else:
            exec_points.append(f"- **估值状态**: PE {pe_ttm:.1f}x，处于中等水平")

    if not df_single.empty:
        rev_yoy = safe_get(df_single.iloc[-1], 'TotalRevenue_TTM_YoY', None)
        if rev_yoy is not None:
            exec_points.append(f"- **增长动力**: 营收同比 {rev_yoy:.1%}")

    if exec_points:
        lines.extend(exec_points)
    else:
        lines.append("- 数据不足，无法生成执行摘要")
    lines.append("")

    # 1. 基本面概览
    lines.append("## 1. 基本面概览\n")
    if not df_single.empty:
        latest = df_single.iloc[-1]
        rev = safe_get(latest, 'TotalRevenue_TTM', 0)
        ni = safe_get(latest, 'NetIncome_TTM', 0)
        eps = safe_get(latest, 'EPS_TTM', 0)
        fcf = safe_get(latest, 'FreeCashFlow_TTM', 0)

        lines.append("| 指标 | 最新 TTM 值 |")
        lines.append("|------|-----------|")
        if rev: lines.append(f"| 营收 | {rev:,.2f} {unit_label} |")
        if ni: lines.append(f"| 净利润 | {ni:,.2f} {unit_label} |")
        if eps: lines.append(f"| EPS | {eps:.2f} |")
        if fcf: lines.append(f"| FCF | {fcf:,.2f} {unit_label} |")
        lines.append("")

    # 2. 估值参数
    lines.append("## 2. 估值参数\n")
    lines.append(f"- **WACC**: {wacc:.2%}")
    lines.append(f"- **Rf**: {rf}%")
    if current_price > 0: lines.append(f"- **当前股价**: ${current_price:.2f}")
    if pe_ttm > 0: lines.append(f"- **PE (TTM)**: {pe_ttm:.1f}x")
    lines.append("")

    # 3. 关键假设与风险
    lines.append("## 3. 关键假设与风险\n")
    lines.append("- 使用历史财务数据外推未来增长率")
    lines.append("- DCF 模型假设永续增长率 2.5%")
    lines.append("- 宏观经济波动可能影响实际增长率")
    lines.append("- 行业竞争格局变化可能影响盈利能力")
    lines.append("")

    lines.append("---")
    lines.append(f"*报告自动生成 (v2.5.2)，仅供参考，不构成投资建议。*")

    return "\n".join(lines)


# ============================================================
# 5. 入口函数
# ============================================================

def render_summary_tab(ticker, df_raw, unit_label, wacc, rf):
    """
    估值总结 — 交互式可视化仪表盘 v2.5.2
    集成大师分析 + 敏感性热力图 + 综合信号面板
    """
    st.subheader(f"📋 估值总结: {ticker}")
    st.caption(
        "综合估值模型、大师分析和敏感性分析的最终汇总报告。"
        "包含核心指标、多模型估值对比、大师评分概览和 DCF/PE 敏感性矩阵。"
    )

    if df_raw.empty:
        st.warning("请先录入财务数据")
        return

    # === 数据提取 ===
    data = _extract_valuation_data(ticker, df_raw, unit_label, wacc, rf)

    if data['latest'] is None:
        st.warning("财务数据处理后为空")
        return

    # === 正推估值 ===
    forward_results = _compute_forward_valuations(data)

    # === 大师评分 (从 session_state 读取，避免重复计算) ===
    master_scores = st.session_state.get('master_scores', None)

    # 如果没有缓存的大师评分 (例如用户直接跳到总结Tab)，则实时计算
    if master_scores is None:
        try:
            _, df_single = process_financial_data(df_raw)
            if not df_single.empty:
                latest = df_single.iloc[-1]
                meta = get_company_meta(ticker)
                df_price = get_market_history(ticker)
                master_scores = compute_master_scores(df_single, latest, meta, df_price)
                st.session_state['master_scores'] = master_scores
        except Exception:
            master_scores = None

    # === QG-Pro 评分 ===
    qg_scores = st.session_state.get('qg_scores', None)
    if qg_scores is None:
        try:
            _, df_single = process_financial_data(df_raw)
            if not df_single.empty:
                latest = df_single.iloc[-1]
                qg_scores = compute_qg_pro_score(df_single, latest)
                st.session_state['qg_scores'] = qg_scores
        except Exception:
            qg_scores = None

    # === 渲染各区块 ===
    intrinsic_value, margin = _render_executive_metrics(data, forward_results, master_scores)

    st.divider()

    _render_valuation_range_chart(data, forward_results, intrinsic_value)

    st.divider()

    _render_master_mini_panel(master_scores)

    st.divider()

    _render_qg_pro_panel(qg_scores)

    st.divider()

    # 敏感性分析 (两栏布局)
    _render_dcf_sensitivity(data)

    st.divider()

    _render_pe_sensitivity(data)

    st.divider()

    # 综合信号面板
    _render_signal_panel(data, forward_results, intrinsic_value, margin, master_scores, qg_scores)

    # === Markdown 报告下载 ===
    st.divider()
    with st.expander("📥 下载 Markdown 报告", expanded=False):
        summary_md = _build_summary_markdown(ticker, df_raw, unit_label, wacc, rf)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📥 下载 Markdown",
                data=summary_md,
                file_name=f"{ticker}_valuation_summary.md",
                mime="text/markdown"
            )
        with col2:
            st.text_area("", summary_md, height=200, label_visibility="collapsed")
