# modules/valuation/valuation_dashboard.py
# 估值整合仪表盘 v2.5
# 正推/倒推动态整合系统 — 各模型相互约束、矫正、验证

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.core.calculator import process_financial_data
from modules.core.db import get_company_meta, get_market_history
from modules.data.industry_data import get_industry_benchmarks
from modules.valuation.valuation_advanced import safe_get


def render_dashboard_tab(ticker, df_raw, unit_label, wacc, rf):
    """估值整合仪表盘 — 正推/倒推动态整合"""
    st.subheader(f"🔀 估值整合仪表盘: {ticker}")
    st.caption("将各估值模型的正推与倒推结果动态整合，实现相互约束、矫正和验证。")
    
    if df_raw.empty:
        st.warning("请先录入财务数据")
        return
    
    # ===== 数据准备 =====
    _, df_single = process_financial_data(df_raw)
    if df_single.empty:
        st.warning("财务数据不足")
        return
    
    latest = df_single.iloc[-1]
    meta = get_company_meta(ticker)
    market_cap = meta.get('last_market_cap', 0)
    sector = meta.get('sector', 'Unknown')
    
    # 获取股价数据
    df_price = get_market_history(ticker)
    current_price = 0
    if not df_price.empty:
        current_price = df_price.iloc[-1].get('close', 0) or 0
    
    # 基础财务指标
    eps_ttm = safe_get(latest, 'EPS_TTM', 0)
    pe_ttm = current_price / eps_ttm if eps_ttm > 0 and current_price > 0 else 0
    
    # FCF 数据
    fcf_ttm = safe_get(latest, 'FreeCashFlow_TTM', 0)
    if fcf_ttm == 0:
        ocf = safe_get(latest, 'OperatingCashFlow_TTM', 0)
        capex = abs(safe_get(latest, 'CapEx', 0))
        if ocf != 0:
            fcf_ttm = ocf - capex
    
    # EBITDA 数据
    ebitda = safe_get(latest, 'EBITDA_TTM', 0) or safe_get(latest, 'OperatingProfit_TTM', 0)
    
    # 增长率数据
    eps_yoy = safe_get(latest, 'EPS_TTM_YoY', None)
    rev_yoy = safe_get(latest, 'TotalRevenue_TTM_YoY', None)
    
    if market_cap == 0 or current_price == 0:
        st.warning("⚠️ 缺少市值或股价数据，整合分析功能受限")
    
    # ========================
    # Section 1: 参数锚定面板
    # ========================
    st.markdown("### ⚙️ 统一参数锚定")
    st.caption("以下参数作为所有模型估值的统一输入，调整后各模型结果自动联动。")
    
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    
    # 统一增长率输入
    growth_default = 10.0
    if eps_yoy is not None and eps_yoy > 0:
        growth_default = eps_yoy * 100
    elif rev_yoy is not None and rev_yoy > 0:
        growth_default = rev_yoy * 100
    
    unified_growth = c_p1.number_input(
        "统一增长率 (%)", value=float(min(growth_default, 50.0)),
        step=0.5, format="%.1f", key="dash_unified_growth",
        help="统一用于 PE/PEG/DCF 模型的增长率假设"
    )
    
    # 永续增长率
    tp_rf = float(rf) if rf else 3.0
    if tp_rf < 0.5:
        tp_rf *= 100
    unified_perp = c_p2.number_input(
        "永续增长率 (%)", value=min(2.5, tp_rf * 0.8),
        step=0.01, format="%.2f", key="dash_perp_growth"
    )
    
    c_p3.metric("WACC", f"{wacc:.2%}")
    c_p4.metric("当前股价", f"${current_price:.2f}" if current_price > 0 else "N/A")
    
    st.divider()
    
    # ==============================
    # Section 2: 正推估值汇总
    # ==============================
    st.markdown("### 📊 正推估值汇总 (Forward Valuation)")
    st.caption("各模型基于统一参数的正推估值结果，所有结果使用相同的增长率假设。")
    
    forward_results = {}
    
    # --- PE Band 估值 ---
    if pe_ttm > 0 and eps_ttm > 0:
        # 计算历史 PE 分位数
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
                
                forward_results['PE Band (20%)'] = {
                    'fair_price': pe_20 * eps_ttm,
                    'method': f"PE {pe_20:.1f}x × EPS {eps_ttm:.2f}",
                    'model': 'PE'
                }
                forward_results['PE Band (50%)'] = {
                    'fair_price': pe_50 * eps_ttm,
                    'method': f"PE {pe_50:.1f}x × EPS {eps_ttm:.2f}",
                    'model': 'PE'
                }
                forward_results['PE Band (80%)'] = {
                    'fair_price': pe_80 * eps_ttm,
                    'method': f"PE {pe_80:.1f}x × EPS {eps_ttm:.2f}",
                    'model': 'PE'
                }
    
    # --- PEG=1 合理价格 ---
    if eps_ttm > 0 and unified_growth > 0:
        peg1_fair_pe = unified_growth  # PEG=1 时 PE = G
        peg1_fair = peg1_fair_pe * eps_ttm
        forward_results['PEG=1 合理价'] = {
            'fair_price': peg1_fair,
            'method': f"PE={unified_growth:.0f}x × EPS {eps_ttm:.2f}",
            'model': 'PEG'
        }
        
        # Fisher 修正
        fisher_pe = unified_growth + 2 * tp_rf
        fisher_fair = fisher_pe * eps_ttm
        forward_results['Fisher 修正价'] = {
            'fair_price': fisher_fair,
            'method': f"PE={fisher_pe:.1f}x (G+2×Rf) × EPS {eps_ttm:.2f}",
            'model': 'PEG'
        }
    
    # --- DCF 正推 ---
    if fcf_ttm != 0 and wacc > unified_perp / 100:
        g_rate = unified_growth / 100
        p_rate = unified_perp / 100
        
        curr_fcf = fcf_ttm
        total_pv = 0
        for i in range(1, 6):
            curr_fcf *= (1 + g_rate)
            total_pv += curr_fcf / ((1 + wacc) ** i)
        
        term_val = curr_fcf * (1 + p_rate) / (wacc - p_rate)
        term_pv = term_val / ((1 + wacc) ** 5)
        dcf_ev = total_pv + term_pv
        
        # 转换为每股价值（简单近似）
        if market_cap > 0 and current_price > 0:
            shares = market_cap / current_price
            dcf_ev_dollars = dcf_ev * 1e9 if dcf_ev < 10000 else dcf_ev
            dcf_per_share = dcf_ev_dollars / shares
            forward_results['DCF 内在价值'] = {
                'fair_price': dcf_per_share,
                'method': f"FCF={fcf_ttm:.1f}{unit_label}, g={unified_growth:.1f}%, WACC={wacc:.1%}",
                'model': 'DCF'
            }
    
    # --- EV/EBITDA 隐含价值 ---
    if ebitda > 0 and market_cap > 0:
        bench = get_industry_benchmarks(sector)
        industry_ev_ebitda = bench.get('ev_ebitda', 15.0)
        
        scale = 1e9 if ebitda < 10000 else 1.0
        ebitda_d = ebitda * scale
        debt = safe_get(latest, 'TotalDebt', 0) or safe_get(latest, 'LongTermDebt', 0)
        cash = safe_get(latest, 'CashAndEquivalents', 0) or safe_get(latest, 'CashEndOfPeriod', 0)
        debt_d = debt * scale
        cash_d = cash * scale
        
        implied_mc = industry_ev_ebitda * ebitda_d - debt_d + cash_d
        if current_price > 0:
            shares = market_cap / current_price
            ev_ebitda_per_share = implied_mc / shares
            forward_results['EV/EBITDA 行业对标'] = {
                'fair_price': ev_ebitda_per_share,
                'method': f"行业中位 {industry_ev_ebitda:.1f}x × EBITDA {ebitda:.1f}",
                'model': 'EV/EBITDA'
            }
    
    # 展示正推汇总表
    if forward_results:
        fwd_data = {
            "估值方法": [],
            "合理股价 ($)": [],
            "vs 当前股价": [],
            "判断": [],
            "计算依据": []
        }
        
        for name, info in forward_results.items():
            fp = info['fair_price']
            fwd_data["估值方法"].append(name)
            fwd_data["合理股价 ($)"].append(f"${fp:.2f}")
            diff = (fp / current_price - 1) * 100 if current_price > 0 else 0
            fwd_data["vs 当前股价"].append(f"{diff:+.1f}%")
            if diff > 10:
                fwd_data["判断"].append("低估 ✅")
            elif diff < -10:
                fwd_data["判断"].append("高估 ⚠️")
            else:
                fwd_data["判断"].append("合理 📊")
            fwd_data["计算依据"].append(info['method'])
        
        st.dataframe(pd.DataFrame(fwd_data), use_container_width=True, hide_index=True)
        
        # 正推估值可视化
        fair_prices = [info['fair_price'] for info in forward_results.values()]
        fair_names = list(forward_results.keys())
        
        fig_fwd = go.Figure()
        colors = ['#3B82F6' if fp > current_price else '#EF4444' for fp in fair_prices]
        
        fig_fwd.add_trace(go.Bar(
            x=fair_names, y=fair_prices,
            marker_color=colors,
            text=[f"${fp:.0f}" for fp in fair_prices],
            textposition='auto', name="合理股价"
        ))
        
        if current_price > 0:
            fig_fwd.add_hline(
                y=current_price, line_dash="dash", line_color="orange",
                annotation_text=f"当前 ${current_price:.0f}"
            )
        
        fig_fwd.update_layout(
            title="各模型正推合理股价对比",
            yaxis_title="股价 ($)", height=350,
            showlegend=False
        )
        st.plotly_chart(fig_fwd, use_container_width=True)
    else:
        st.info("数据不足，无法生成正推估值汇总")
    
    st.divider()
    
    # ==============================
    # Section 3: 倒推隐含参数汇总
    # ==============================
    st.markdown("### 🔄 倒推隐含参数汇总 (Reverse Valuation)")
    st.caption("从当前市值/股价反推各模型隐含的增长率或估值倍数。")
    
    reverse_results = {}
    
    # --- PE 倒推隐含增长率 (Fisher Model) ---
    if pe_ttm > 0:
        implied_g_fisher = pe_ttm - 2 * tp_rf  # PE = G + 2×Rf → G = PE - 2×Rf
        reverse_results['PE/Fisher 隐含增长率'] = {
            'value': implied_g_fisher,
            'unit': '%',
            'method': f"G = PE({pe_ttm:.1f}) - 2×Rf({tp_rf:.1f}%) = {implied_g_fisher:.1f}%",
            'model': 'PE'
        }
        
        # 传统 PEG 隐含增长率
        implied_g_peg = pe_ttm  # PEG=1 时 G = PE
        reverse_results['PEG=1 隐含增长率'] = {
            'value': implied_g_peg,
            'unit': '%',
            'method': f"G = PE({pe_ttm:.1f}) (PEG=1意味着 PE = G%)",
            'model': 'PE'
        }
    
    # --- DCF 倒推隐含增长率 ---
    if fcf_ttm != 0 and market_cap > 0 and wacc > unified_perp / 100:
        p_rate = unified_perp / 100
        fcf_dollars = fcf_ttm * 1e9 if fcf_ttm < 10000 else fcf_ttm
        
        # 二分查找
        low_g, high_g = -0.5, 1.0
        for _ in range(100):
            mid_g = (low_g + high_g) / 2
            c = fcf_dollars
            tp = 0
            for i in range(1, 6):
                c *= (1 + mid_g)
                tp += c / ((1 + wacc) ** i)
            tv = c * (1 + p_rate) / (wacc - p_rate)
            tp += tv / ((1 + wacc) ** 5)
            if abs(tp - market_cap) < market_cap * 0.0001:
                break
            if tp < market_cap:
                low_g = mid_g
            else:
                high_g = mid_g
        
        dcf_implied_g = (low_g + high_g) / 2
        reverse_results['DCF 隐含增长率'] = {
            'value': dcf_implied_g * 100,
            'unit': '%',
            'method': f"FCF={fcf_ttm:.1f}, 支撑市值{market_cap/1e9:.1f}B需年增{dcf_implied_g:.1%}",
            'model': 'DCF'
        }
    
    # --- EV/EBITDA 隐含倍数 ---
    if ebitda > 0 and market_cap > 0:
        scale = 1e9 if ebitda < 10000 else 1.0
        ebitda_d = ebitda * scale
        debt_d = (safe_get(latest, 'TotalDebt', 0) or safe_get(latest, 'LongTermDebt', 0)) * scale
        cash_d = (safe_get(latest, 'CashAndEquivalents', 0) or safe_get(latest, 'CashEndOfPeriod', 0)) * scale
        actual_ev = market_cap + debt_d - cash_d
        actual_ev_ebitda = actual_ev / ebitda_d if ebitda_d > 0 else 0
        
        reverse_results['EV/EBITDA 实际倍数'] = {
            'value': actual_ev_ebitda,
            'unit': 'x',
            'method': f"EV({actual_ev/1e9:.1f}B) / EBITDA({ebitda_d/1e9:.1f}B) = {actual_ev_ebitda:.1f}x",
            'model': 'EV/EBITDA'
        }
    
    # 展示倒推汇总表
    if reverse_results:
        rev_data = {
            "模型": [],
            "隐含值": [],
            "vs 统一假设": [],
            "偏差信号": [],
            "计算依据": []
        }
        
        for name, info in reverse_results.items():
            rev_data["模型"].append(name)
            rev_data["隐含值"].append(f"{info['value']:.1f}{info['unit']}")
            
            # 对比统一假设
            if info['unit'] == '%':
                gap = info['value'] - unified_growth
                rev_data["vs 统一假设"].append(f"{gap:+.1f}%")
                if abs(gap) < 3:
                    rev_data["偏差信号"].append("一致 ✅")
                elif gap > 0:
                    rev_data["偏差信号"].append("市场更乐观 ⬆️")
                else:
                    rev_data["偏差信号"].append("市场更保守 ⬇️")
            else:
                rev_data["vs 统一假设"].append("—")
                rev_data["偏差信号"].append("—")
            
            rev_data["计算依据"].append(info['method'])
        
        st.dataframe(pd.DataFrame(rev_data), use_container_width=True, hide_index=True)
    else:
        st.info("数据不足，无法生成倒推分析")
    
    st.divider()
    
    # ==============================
    # Section 4: 交叉约束矩阵
    # ==============================
    st.markdown("### 🔗 交叉约束矩阵 (Cross-Validation)")
    st.caption("不同模型的估值结果相互验证，识别一致性信号与潜在矛盾。")
    
    # 收集增长率数据点
    growth_points = {}
    for name, info in reverse_results.items():
        if info['unit'] == '%':
            growth_points[name] = info['value']
    growth_points['统一假设'] = unified_growth
    
    if len(growth_points) >= 2:
        # 增长率一致性分析
        st.markdown("#### 📐 增长率一致性检验")
        
        g_values = list(growth_points.values())
        g_names = list(growth_points.keys())
        g_mean = np.mean(g_values)
        g_std = np.std(g_values)
        cv = g_std / abs(g_mean) * 100 if g_mean != 0 else 0  # 变异系数
        
        cols_g = st.columns(len(growth_points))
        for i, (gn, gv) in enumerate(growth_points.items()):
            delta = gv - unified_growth
            cols_g[i].metric(
                gn.replace("隐含增长率", "").strip(),
                f"{gv:.1f}%",
                f"{delta:+.1f}%" if gn != '统一假设' else None,
                delta_color="inverse"
            )
        
        # 一致性评分
        if cv < 15:
            st.success(f"✅ **高度一致** (CV={cv:.0f}%): 各模型增长率偏差小，估值交叉验证通过，结论可信度高。")
        elif cv < 30:
            st.warning(f"🟡 **部分分歧** (CV={cv:.0f}%): 模型间存在中等偏差，建议审视增长率假设合理性。")
        else:
            st.error(f"🔴 **显著矛盾** (CV={cv:.0f}%): 各模型隐含增长率差异大，需重新审视参数假设或关注市场定价是否合理。")
        
        # 增长率对比柱状图
        fig_gc = go.Figure()
        colors_gc = ['#10B981' if abs(v - unified_growth) < 3 else '#F59E0B' if abs(v - unified_growth) < 8 else '#EF4444' 
                     for v in g_values]
        fig_gc.add_trace(go.Bar(
            x=g_names, y=g_values,
            marker_color=colors_gc,
            text=[f"{v:.1f}%" for v in g_values],
            textposition='auto'
        ))
        fig_gc.add_hline(y=unified_growth, line_dash="dash", line_color="blue",
                         annotation_text=f"统一假设 {unified_growth:.1f}%")
        fig_gc.update_layout(
            title="增长率交叉验证", yaxis_title="增长率 (%)", height=300,
            showlegend=False
        )
        st.plotly_chart(fig_gc, use_container_width=True)
    
    st.divider()
    
    # ==============================
    # Section 5: 内在价值综合评估
    # ==============================
    st.markdown("### 💎 内在价值综合评估")
    st.caption("基于所有模型结果的加权平均，计算综合内在价值与安全边际。")
    
    if forward_results and current_price > 0:
        # 模型权重配置
        model_weights = {
            'PE': 0.25,    # PE 模型权重
            'PEG': 0.20,   # PEG 模型权重
            'DCF': 0.35,   # DCF 模型权重（最高，因为基于现金流）
            'EV/EBITDA': 0.20  # EV/EBITDA 模型权重
        }
        
        st.markdown("#### ⚖️ 模型权重配置")
        w_cols = st.columns(4)
        w_pe = w_cols[0].number_input("PE 权重", value=0.25, step=0.05, format="%.2f", key="w_pe")
        w_peg = w_cols[1].number_input("PEG 权重", value=0.20, step=0.05, format="%.2f", key="w_peg")
        w_dcf = w_cols[2].number_input("DCF 权重", value=0.35, step=0.05, format="%.2f", key="w_dcf")
        w_ev = w_cols[3].number_input("EV/EBITDA 权重", value=0.20, step=0.05, format="%.2f", key="w_ev")
        
        model_weights = {'PE': w_pe, 'PEG': w_peg, 'DCF': w_dcf, 'EV/EBITDA': w_ev}
        
        # 归一化权重
        total_w = sum(model_weights.values())
        if total_w > 0:
            model_weights = {k: v / total_w for k, v in model_weights.items()}
        
        # 计算加权平均内在价值
        weighted_sum = 0
        weight_used = 0
        model_contributions = {}
        
        for name, info in forward_results.items():
            model_type = info['model']
            w = model_weights.get(model_type, 0)
            if w > 0:
                # 同一模型可能有多个结果，取中位数
                if model_type not in model_contributions:
                    model_contributions[model_type] = []
                model_contributions[model_type].append(info['fair_price'])
        
        for model_type, prices in model_contributions.items():
            median_price = np.median(prices)
            w = model_weights.get(model_type, 0)
            weighted_sum += median_price * w
            weight_used += w
        
        intrinsic_value = weighted_sum / weight_used if weight_used > 0 else 0
        
        # 安全边际
        margin_of_safety = (intrinsic_value / current_price - 1) * 100 if current_price > 0 else 0
        
        # 信心等级（基于模型一致性）
        all_fairs = [info['fair_price'] for info in forward_results.values()]
        fair_std = np.std(all_fairs) / np.mean(all_fairs) * 100 if np.mean(all_fairs) != 0 else 100
        
        if fair_std < 15:
            confidence = "🟢 高 (模型高度一致)"
            confidence_score = 85
        elif fair_std < 30:
            confidence = "🟡 中 (模型存在分歧)"
            confidence_score = 60
        else:
            confidence = "🔴 低 (模型严重分歧)"
            confidence_score = 35
        
        # 展示核心指标
        st.markdown("#### 🎯 核心结论")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("综合内在价值", f"${intrinsic_value:.2f}")
        r2.metric("当前股价", f"${current_price:.2f}")
        r3.metric(
            "安全边际",
            f"{margin_of_safety:+.1f}%",
            "被低估" if margin_of_safety > 0 else "被高估",
            delta_color="normal" if margin_of_safety > 0 else "inverse"
        )
        r4.metric("估值信心", confidence)
        
        # 估值区间可视化
        st.markdown("#### 📊 估值区间全览")
        
        fig_range = go.Figure()
        
        # 各模型贡献的点
        for name, info in forward_results.items():
            color_map = {'PE': '#3B82F6', 'PEG': '#10B981', 'DCF': '#F59E0B', 'EV/EBITDA': '#8B5CF6'}
            fig_range.add_trace(go.Scatter(
                x=[info['fair_price']], y=[name],
                mode='markers',
                marker=dict(size=12, color=color_map.get(info['model'], 'gray')),
                name=info['model'],
                showlegend=False,
                hovertemplate=f"{name}: ${info['fair_price']:.2f}<extra></extra>"
            ))
        
        # 内在价值线
        fig_range.add_vline(
            x=intrinsic_value, line_dash="solid", line_color="green", line_width=3,
            annotation_text=f"内在价值 ${intrinsic_value:.0f}"
        )
        
        # 当前股价线
        fig_range.add_vline(
            x=current_price, line_dash="dash", line_color="orange", line_width=2,
            annotation_text=f"当前 ${current_price:.0f}"
        )
        
        fig_range.update_layout(
            title=f"估值区间 (内在价值 ${intrinsic_value:.0f} vs 当前 ${current_price:.0f}, 安全边际 {margin_of_safety:+.1f}%)",
            xaxis_title="股价 ($)",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_range, use_container_width=True)
        
        # 雷达图 — 多维度评估
        st.markdown("#### 🕸️ 多维度评估雷达图")
        
        # 构建评估维度
        dimensions = []
        dim_scores = []
        
        # 维度1: 估值吸引力 (安全边际越高越好)
        val_score = min(max(50 + margin_of_safety, 0), 100)
        dimensions.append("估值吸引力")
        dim_scores.append(val_score)
        
        # 维度2: 增长动力
        growth_score = min(max(unified_growth * 3, 0), 100)
        dimensions.append("增长动力")
        dim_scores.append(growth_score)
        
        # 维度3: 盈利质量 (OCF vs NI)
        ni = safe_get(latest, 'NetIncome_TTM', 0)
        ocf = safe_get(latest, 'OperatingCashFlow_TTM', 0)
        if ni > 0 and ocf > 0:
            quality_ratio = min(ocf / ni, 2.0)  # OCF/NI > 1 说明盈利质量高
            quality_score = min(quality_ratio * 50, 100)
        else:
            quality_score = 50
        dimensions.append("盈利质量")
        dim_scores.append(quality_score)
        
        # 维度4: 模型一致性
        dimensions.append("模型一致性")
        dim_scores.append(confidence_score)
        
        # 维度5: 现金流健康度
        if fcf_ttm > 0 and ni > 0:
            fcf_score = min(fcf_ttm / ni * 50, 100) if ni > 0 else 50
        else:
            fcf_score = 30
        dimensions.append("现金流健康")
        dim_scores.append(fcf_score)
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=dim_scores + [dim_scores[0]],  # 闭合
            theta=dimensions + [dimensions[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='#3B82F6', width=2),
            name='当前评估'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="估值综合评估雷达图",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # ==============================
        # Section 6: 综合判断信号
        # ==============================
        st.markdown("### 💡 综合判断信号")
        
        # 统计正推信号
        bullish_count = sum(1 for info in forward_results.values() if info['fair_price'] > current_price * 1.1)
        bearish_count = sum(1 for info in forward_results.values() if info['fair_price'] < current_price * 0.9)
        total_models = len(forward_results)
        
        # 综合判断
        bull_ratio = bullish_count / total_models if total_models > 0 else 0
        
        st.markdown("#### 📋 信号汇总")
        
        signal_data = {
            "维度": ["正推模型共识", "安全边际", "增长率一致性", "盈利质量", "综合评级"],
            "结果": [],
            "信号": []
        }
        
        # 正推共识
        signal_data["结果"].append(f"{bullish_count}/{total_models} 看涨")
        if bull_ratio >= 0.6:
            signal_data["信号"].append("✅ 多数模型看涨")
        elif bull_ratio <= 0.3:
            signal_data["信号"].append("⚠️ 多数模型看跌")
        else:
            signal_data["信号"].append("📊 信号分歧")
        
        # 安全边际
        signal_data["结果"].append(f"{margin_of_safety:+.1f}%")
        if margin_of_safety > 20:
            signal_data["信号"].append("✅ 充足安全边际")
        elif margin_of_safety > 0:
            signal_data["信号"].append("📊 微弱低估")
        elif margin_of_safety > -15:
            signal_data["信号"].append("📊 微弱高估")
        else:
            signal_data["信号"].append("⚠️ 显著高估")
        
        # 增长率一致性
        if len(growth_points) >= 2:
            signal_data["结果"].append(f"CV={cv:.0f}%")
            if cv < 15:
                signal_data["信号"].append("✅ 高度一致")
            elif cv < 30:
                signal_data["信号"].append("🟡 中等分歧")
            else:
                signal_data["信号"].append("⚠️ 严重分歧")
        else:
            signal_data["结果"].append("N/A")
            signal_data["信号"].append("—")
        
        # 盈利质量
        signal_data["结果"].append(f"得分 {quality_score:.0f}/100")
        if quality_score >= 70:
            signal_data["信号"].append("✅ 优良")
        elif quality_score >= 40:
            signal_data["信号"].append("📊 一般")
        else:
            signal_data["信号"].append("⚠️ 较差")
        
        # 综合评级
        total_score = (val_score * 0.3 + growth_score * 0.2 + quality_score * 0.2 + 
                       confidence_score * 0.15 + fcf_score * 0.15)
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
        
        signal_data["结果"].append(f"综合得分 {total_score:.0f}/100")
        signal_data["信号"].append(overall)
        
        st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)
        
        # 最终结论
        st.markdown("---")
        if margin_of_safety > 20 and bull_ratio >= 0.6:
            st.success(f"""
            🟢 **综合结论: 低估**
            
            综合内在价值 **${intrinsic_value:.2f}**，安全边际 **{margin_of_safety:+.1f}%**。
            {bullish_count}/{total_models} 个模型显示低估，估值信心等级: {confidence}。
            """)
        elif margin_of_safety < -15 and bull_ratio <= 0.3:
            st.error(f"""
            🔴 **综合结论: 高估**
            
            综合内在价值 **${intrinsic_value:.2f}**，安全边际 **{margin_of_safety:+.1f}%**。
            {bearish_count}/{total_models} 个模型显示高估，估值信心等级: {confidence}。
            """)
        else:
            st.info(f"""
            📊 **综合结论: 估值中性**
            
            综合内在价值 **${intrinsic_value:.2f}**，安全边际 **{margin_of_safety:+.1f}%**。
            信号存在分歧，建议结合行业趋势和公司基本面做进一步判断。
            """)
    else:
        st.info("数据不足，无法生成综合内在价值评估。请确保已录入足够的财务数据和市场数据。")
