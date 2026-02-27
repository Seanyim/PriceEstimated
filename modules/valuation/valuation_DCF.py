import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.core.calculator import process_financial_data
from modules.core.db import get_company_meta, get_market_history
from modules.valuation.valuation_advanced import _render_dcf_reverse, safe_get

def render_valuation_DCF_tab(df_raw, wacc, rf, unit_label):
    st.subheader("🚀 DCF 现金流折现 v2.1")
    
    if df_raw.empty: return
    
    # 1. 自动计算基准数据
    _, df_single_q = process_financial_data(df_raw) # df_single_q is Q1-Q4 data
    
    if df_single_q.empty:
        st.warning("缺少财务数据")
        return
        
    latest_q = df_single_q.iloc[-1]
    ticker = df_raw.iloc[0]['ticker']
    meta = get_company_meta(ticker)
    market_cap = meta.get('last_market_cap', 0)
    
    # --- FCF 基准选择逻辑 (优化版) ---
    # 规则: 如果存在最新 FY 之后的季度数据 (Q2/Q3 等)，优先使用 TTM
    # 否则使用最新 FY 数据 (避免 Q1 波动过大影响)
    
    df_fy = df_raw[df_raw['period'] == 'FY'].sort_values('year')
    latest_fy_year = df_fy.iloc[-1]['year'] if not df_fy.empty else 0
    
    # 检查是否有更新的季度数据
    has_newer_data = False
    if not df_single_q.empty:
        last_record_year = df_single_q.iloc[-1]['year']
        last_record_period = df_single_q.iloc[-1]['period']
        # 简单判断: 如果单季度数据的最后一年 > FY年份，且 Period 不是 Q1 (即至少有H1/Q2)，或者 Year 大了不止1年
        # 用户需求: "相差半个年度以上" -> 意味着有 Q2, Q3 数据。
        # 这里宽松一点：只要有 FY 之后的数据，且 TTM 有效，就倾向于 TTM，但如果是 Q1 可能波动大。 
        # 用户例子: 2026/Q1, Q2, Q3 -> TTM. 
        if last_record_year > latest_fy_year:
             # 如果仅仅是 Q1，有时 TTM 会受季节性影响，但由用户决定，默认推荐 TTM
             has_newer_data = True
    
    # 默认值逻辑
    base_fcf = 0
    fcf_source = "Unknown"
    
    val_ttm = latest_q.get('FreeCashFlow_TTM', 0)
    val_fy = df_fy.iloc[-1].get('FreeCashFlow', 0) if not df_fy.empty else 0
    
    # 尝试补全 OCF-CapEx
    if val_ttm == 0:
        o = latest_q.get('OperatingCashFlow_TTM', 0)
        c = abs(latest_q.get('CapEx', 0)) # CapEx is usually negative
        if o != 0: val_ttm = o - c
        
    if val_fy == 0 and not df_fy.empty:
        o = df_fy.iloc[-1].get('OperatingCashFlow', 0)
        c = abs(df_fy.iloc[-1].get('CapEx', 0))
        if o != 0: val_fy = o - c

    # 决策
    if has_newer_data and val_ttm != 0:
        base_fcf = val_ttm
        fcf_source = f"FCF TTM (含 {last_record_year} {last_record_period})"
    elif val_fy != 0:
        base_fcf = val_fy
        fcf_source = f"FCF FY {latest_fy_year}"
    else:
        base_fcf = val_ttm # Fallback
        fcf_source = "FCF TTM (Fallback)"
            
    if base_fcf == 0:
        st.warning("缺少 FCF 数据，请录入自由现金流 (FreeCashFlow / OperatingCashFlow)")
        return
    
    # 2. 动态计算多种历史增长率 (v2.1 增强)
    growth_options = {}
    growth_debug_info = {} # Store details for display
    
    # A. 平滑趋势预测 (Log-Linear Regression) - 推荐
    # A. 平滑趋势预测 (Log-Linear Regression) - 推荐
    calc_error = None
    try:
        # **关键修复**: 如果 df_fy 数据不足 (可能是只有单季度数据), 尝试从 Q1-Q4 合成年度数据
        target_df = df_fy
        if len(target_df) < 3 and not df_single_q.empty:
            # Group by year and count quarters
            annual_groups = df_single_q.groupby('year')
            synth_rows = []
            for year, group in annual_groups:
                # 只有当该年有 4 个季度数据时才合成 (或者至少 3 个? 严格点选 4)
                if len(group) == 4:
                    # Sum relevant columns
                    fcf_sum = group['FreeCashFlow'].sum() if 'FreeCashFlow' in group.columns else 0
                    if fcf_sum == 0:
                        o = group['OperatingCashFlow'].sum() if 'OperatingCashFlow' in group.columns else 0
                        c = abs(group['CapEx'].sum()) if 'CapEx' in group.columns else 0
                        if o != 0: fcf_sum = o - c
                    
                    if fcf_sum != 0:
                        synth_rows.append({'year': year, 'FreeCashFlow': fcf_sum})
            
            if len(synth_rows) >= 3:
                target_df = pd.DataFrame(synth_rows).sort_values('year')
        
        # 使用 target_df 进行回归
        if len(target_df) >= 3:
            # 取最近 5 年
            df_trend = target_df.tail(5).copy()
            
            # 1. 尝试 Log-Linear (如果不含负值且数据点足够)
            pos_mask = []
            fcf_values = []
            years = []
            for _, r in df_trend.iterrows():
                # 安全获取数值
                v1 = r.get('FreeCashFlow')
                if pd.isna(v1) or v1 == "": v1 = 0
                
                v = float(v1)
                # Backup logic if column empty but components exist (already handled in synth, but good for raw)
                if v == 0 and 'OperatingCashFlow' in r:
                    o = r.get('OperatingCashFlow', 0)
                    c = abs(r.get('CapEx', 0))
                    if pd.notna(o) and o != 0: v = float(o) - float(c)
                
                fcf_values.append(v)
                years.append(r['year'])
                pos_mask.append(v > 0)
            
            fcf_arr = np.array(fcf_values)
            years_arr = np.array(years)
            pos_count = sum(pos_mask)
            
            calc_type = None
            smooth_pct = 0 # Initialize
            
            # 优先 1: Log-Linear (需要至少3个正值，且正值比例较高)
            if pos_count >= 3:
                # 只取正值点做 Log 回归
                valid_idx = [i for i, x in enumerate(pos_mask) if x]
                y_log = np.log(fcf_arr[valid_idx])
                x_act = years_arr[valid_idx] - years_arr[0]
                
                slope, intercept = np.polyfit(x_act, y_log, 1)
                
                # R2
                y_pred = slope * x_act + intercept
                ss_res = np.sum((y_log - y_pred) ** 2)
                ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                smooth_g = np.exp(slope) - 1
                try_pct = smooth_g * 100
                
                # Log-Linear 依旧保留宽松检查，如果太离谱就降级到 Linear
                if -100 < try_pct < 300:
                    smooth_pct = try_pct
                    calc_type = "log_linear"
                
            # 备选 2: 线性回归 (Linear Regression) - 只有当 Log-Linear 失败或结果极端时才用
            if calc_type is None:
                # 使用所有点 (含负值)
                x_lin = years_arr - years_arr[0]
                slope_lin, intercept_lin = np.polyfit(x_lin, fcf_arr, 1)
                
                # 计算相对增长率: 斜率 / 平均绝对值
                avg_abs_val = np.mean(np.abs(fcf_arr))
                if avg_abs_val != 0:
                    smooth_g = slope_lin / avg_abs_val
                    smooth_pct = smooth_g * 100
                    
                    # R2 for linear
                    y_lin_pred = slope_lin * x_lin + intercept_lin
                    ss_res = np.sum((fcf_arr - y_lin_pred) ** 2)
                    ss_tot = np.sum((fcf_arr - np.mean(fcf_arr)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    calc_type = "linear"
                    slope = slope_lin
                    intercept = intercept_lin
            
            # 最终检查: 只要算出来了，就必须用 (Linear Fallback 不设限，但给提示)
            if calc_type:
                label_prefix = "📈" if smooth_pct > 0 else "📉"
                label = f"{label_prefix} 趋势预测 ({calc_type[:3]}) ({smooth_pct:.1f}%)"
                growth_options[label] = smooth_pct
                growth_debug_info[label] = {
                    "type": calc_type,
                    "years": years,
                    "values": fcf_values,
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r2,
                    "formula": "ln(FCF) = a*t + b (Exp)" if calc_type == "log_linear" else "FCF = a*t + b (Lin)"
                }
            else:
                calc_error = f"Calculation failed (Avg FCF is 0?)"
        else:
            calc_error = f"Insufficient annual data: {len(target_df)} years (need 3). Calc from {len(df_single_q)} quarters failed."
            
    except Exception as e:
        calc_error = f"Error: {str(e)}"
        pass

    # B. FCF CAGR (5年) - 传统
    try:
        if len(df_fy) >= 5:
            # ... (Existing logic for simple CAGR)
            pass 
    except Exception:
        pass
    
    # 复用原有 CAGR 计算逻辑 (简化保留)
    try:
        if len(df_fy) >= 5:
            vals = []
            years = []
            for _, r in df_fy.tail(5).iterrows():
                v = r.get('FreeCashFlow') or (r.get('OperatingCashFlow', 0) - abs(r.get('CapEx', 0)))
                vals.append(v)
                years.append(r['year'])
            if vals[0] > 0 and vals[-1] > 0:
                cagr = (vals[-1]/vals[0])**(1/4) - 1
                label = f"FCF 5Y CAGR ({cagr*100:.1f}%)"
                growth_options[label] = cagr * 100
                growth_debug_info[label] = {
                    "type": "cagr",
                    "start_year": years[0],
                    "end_year": years[-1],
                    "start_val": vals[0],
                    "end_val": vals[-1]
                }
    except: pass

    # Revenue / EPS CAGR 补充
    try:
        if len(df_fy) >= 5:
            rev_s = df_fy.tail(5)['TotalRevenue'].dropna().values
            if len(rev_s) >=2 and rev_s[0]>0 and rev_s[-1]>0:
                g = (rev_s[-1]/rev_s[0])**(1/(len(rev_s)-1)) - 1
                growth_options[f"Revenue 5Y CAGR ({g*100:.1f}%)"] = g*100
    except: pass
    
    # 默认值
    if not growth_options:
        default_label = "默认 (10.0%)"
        growth_options[default_label] = 10.0
        if calc_error:
            growth_debug_info[default_label] = {
                "type": "error",
                "message": calc_error,
                "formula": "Calculation Failed"
            }
    
    # 3. 参数输入
    st.markdown("#### ⚙️ DCF 参数设置")
    c1, c2, c3 = st.columns(3)
    
    init_fcf = c1.number_input(f"基准 FCF ({unit_label})", value=float(base_fcf), 
                                help=f"数据来源: {fcf_source}")
    
    # 默认选中第一个 (通常是趋势预测，如果字典是插入顺序)
    default_opt_idx = 0
    growth_choice = c2.selectbox("增长率来源", list(growth_options.keys()), index=default_opt_idx)
    growth_default = growth_options[growth_choice]
    
    # 优先使用 Session State 中的手动输入值，否则使用默认计算值
    if "dcf_stage1_growth_rate" in st.session_state:
        growth_val_to_use = st.session_state["dcf_stage1_growth_rate"]
    else:
        growth_val_to_use = float(growth_default)

    growth_rate_input = c2.number_input("前5年增长率 (%)", 
                                        value=growth_val_to_use, 
                                        step=0.1, 
                                        format="%.1f",
                                        key="dcf_stage1_growth_rate",
                                        help="支持手动输入覆盖自动计算值")
    growth_rate = growth_rate_input / 100
    
    # 3.3 永续增长率 (移至 C3)
    tp_rf = float(rf) if rf else 3.0
    if tp_rf < 0.5: tp_rf *= 100 
        
    perp_cap = tp_rf
    # 优先使用 Session State 中的手动输入值，否则使用默认计算值
    if "dcf_perp_growth_rate" in st.session_state:
        perp_default = st.session_state["dcf_perp_growth_rate"]
    else:
        perp_default = min(3.0, perp_cap * 0.8) 
    
    perp_rate_input = c3.number_input(
        "永续增长率 (%)", 
        value=float(perp_default),
        min_value=None,
        max_value=None,
        step=0.01,
        format="%.2f",
        key="dcf_perp_growth_rate",
        help="通常为 2.0% ~ 3.0%。支持手动输入任意数值。"
    )
    perp_rate = perp_rate_input / 100
    
    if wacc <= perp_rate:
        st.error(f"❌ WACC ({wacc:.1%}) 必须大于永续增长率 ({perp_rate:.1%})")
        return

    # v2.5: 动态反馈约束 (倒推市场隐含增长率对比)
    if market_cap > 0 and base_fcf > 0:
        # 二分查找计算隐含增长率
        low_g, high_g = -0.5, 1.0
        fcf_dollars = base_fcf * 1e9 if base_fcf < 10000 else base_fcf
        for _ in range(50):
            mid_g = (low_g + high_g) / 2
            c = fcf_dollars
            tp = 0
            for i in range(1, 6):
                c *= (1 + mid_g)
                tp += c / ((1 + wacc) ** i)
            tv = c * (1 + perp_rate) / (wacc - perp_rate)
            tp += tv / ((1 + wacc) ** 5)
            if abs(tp - market_cap) < market_cap * 0.001:
                break
            if tp < market_cap:
                low_g = mid_g
            else:
                high_g = mid_g
        
        implied_g = (low_g + high_g) / 2
        delta_g = implied_g - growth_rate
        implied_g_pct = implied_g * 100
        input_g_pct = growth_rate * 100

        if delta_g > 0.05:
            st.warning(f"⚠️ **预期偏差提示**: 结合当前市值 ({market_cap/1e9:.1f}B)，市场隐含的前5年增长率预期约为 **{implied_g_pct:.1f}%**。您的输入 ({input_g_pct:.1f}%) 显著低于市场预期，如果您的判断正确，该股可能被**高估**。")
        elif delta_g < -0.05:
            st.success(f"🟢 **预期偏差提示**: 结合当前市值 ({market_cap/1e9:.1f}B)，市场隐含的前5年增长率预期约为 **{implied_g_pct:.1f}%**。您的输入 ({input_g_pct:.1f}%) 显著高于市场预期，如果您的判断正确，该股可能被**低估**。")
        else:
            st.info(f"⚖️ **预期偏差提示**: 您的输入 ({input_g_pct:.1f}%) 与当前市值隐含的增长率预期 ({implied_g_pct:.1f}%) 基本一致，估值合理。")

    # --- 增长率计算详情展示 (New) ---
    if growth_choice in growth_debug_info:
        info = growth_debug_info[growth_choice]
        with st.expander("🔢 查看增长率计算过程 (含可视化)", expanded=False):
            
            # 1. 可视化回归拟合图
            st.markdown("#### 1. 趋势拟合可视化")
            years = np.array(info['years'])
            values = np.array(info['values'])
            x_rel = years - years[0]
            
            # 生成拟合线数据
            slope = info['slope']
            intercept = info['intercept']
            
            if info['type'] == 'log_linear':
                # ln(y) = ax + b  => y = e^(ax+b)
                y_fit = np.exp(slope * x_rel + intercept)
                model_name = "Log-Linear (指数回归)"
            else: # linear
                # y = ax + b
                y_fit = slope * x_rel + intercept
                model_name = "Linear (线性回归)"
            
            fig_reg = go.Figure()
            # 实际点
            fig_reg.add_trace(go.Scatter(
                x=years, y=values,
                mode='markers',
                name='实际 FCF',
                marker=dict(color='#3B82F6', size=10),
                text=[f"{v:,.2f}" for v in values]
            ))
            # 拟合线
            fig_reg.add_trace(go.Scatter(
                x=years, y=y_fit,
                mode='lines',
                name=f'趋势线 (R²={info["r_squared"]:.2f})',
                line=dict(color='#F59E0B', width=2, dash='dash')
            ))
            
            fig_reg.update_layout(
                title=f"FCF 增长趋势拟合: {model_name}",
                xaxis_title="年份",
                yaxis_title=f"FCF ({unit_label})",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            
            # 2. 详细计算步骤
            st.markdown("#### 2. 计算步骤分解")
            
            c1_d, c2_d, c3_d = st.columns(3)
            c1_d.metric("拟合斜率 (Slope)", f"{slope:.4f}")
            c2_d.metric("截距 (Intercept)", f"{intercept:,.2f}")
            c3_d.metric("R² (拟合优度)", f"{info['r_squared']:.2f}")

            if info['type'] == 'log_linear':
                st.info(r"""
                **Log-Linear 模型推导过程**:
                1. **数据预处理**: 取 FCF 的自然对数 $y' = \ln(FCF)$
                2. **线性回归**: 对 $y'$ 和 $t$ (年份差) 做回归，得到 $y' = a \cdot t + b$
                3. **还原增长率**: 
                   - **斜率 $a$ (指数因子)**: 代表连续复利增长率 (Continuous Compounding Rate)
                   - **年化增长率 (Annualized Growth)**: $Growth = e^a - 1$
                   - **为何叫 CAGR?**: 它的数学意义等同于这条平滑趋势线的复合年均增长率，但比简单首尾相除更稳健 (考虑了中间所有年份波动)。
                   - 计算: $e^{%.4f} - 1 = %.1f%%$
                """ % (slope, (np.exp(slope)-1)*100))
                
                # Table
                df_dbg = pd.DataFrame({
                    "年份": years,
                    "t (时间差)": x_rel,
                    f"FCF ({unit_label})": [f"{v:,.2f}" for v in values],
                    "ln(FCF)": [f"{np.log(v):.2f}" for v in values],
                    "拟合预测": [f"{v:,.2f}" for v in y_fit]
                })
                st.dataframe(df_dbg, use_container_width=True, hide_index=True)
            
            elif info['type'] == 'linear':
                avg_abs = np.mean(np.abs(values))
                st.info(f"""
                **Linear Regression 模型推导过程**:
                1. **线性回归**: 直接拟合 FCF = $a \cdot t + b$
                2. **计算平均规模**: 历史 FCF 绝对值的平均数 = {avg_abs:,.0f}
                3. **相对增长率**: 
                   - 斜率 $a$ = {slope:,.0f} (每年增加额)
                   - 增长率 = 斜率 / 平均规模 = {slope:,.0f} / {avg_abs:,.0f} = {growth_options[growth_choice]:.1f}%
                """)
                
                df_dbg = pd.DataFrame({
                    "年份": years,
                    "t (时间差)": x_rel,
                    f"FCF ({unit_label})": [f"{v:,.2f}" for v in values],
                    "拟合预测": [f"{v:,.2f}" for v in y_fit]
                })
                st.dataframe(df_dbg, use_container_width=True, hide_index=True)

            elif info['type'] == 'cagr':
                st.markdown(f"**方法**: 复合年均增长率 (CAGR)")
                st.latex(r"CAGR = \left( \frac{V_{end}}{V_{start}} \right)^{\frac{1}{n}} - 1")
                st.write(f"Start: {info['start_year']} ({info['start_val']:,.0f}) → End: {info['end_year']} ({info['end_val']:,.0f})")
                
            elif info['type'] == 'error':
                st.error(f"❌ 自动计算失败原因: {info['message']}")
                st.info("系统无法从历史数据提取有效的趋势 (可能数据点不足、波动过大或包含无效值)，已回退到默认 10%。")
    
    # 4. 计算
    flows = []    # 预测 FCF
    pvs = []      # 折现值 PV
    yoy_rates = []  # 各年 YoY
    curr = init_fcf
    total_pv = 0
    
    for i in range(1, 6):
        prev = curr
        curr = curr * (1 + growth_rate)
        pv = curr / ((1 + wacc) ** i)
        total_pv += pv
        flows.append(curr)
        pvs.append(pv)
        yoy_rate = (curr / prev - 1) * 100 if prev != 0 else 0
        yoy_rates.append(yoy_rate)
    
    # 终值 (Terminal Value)
    term_fcf = flows[-1] * (1 + perp_rate)
    term_val = term_fcf / (wacc - perp_rate)
    term_pv = term_val / ((1 + wacc) ** 5)
    
    enterprise_value = total_pv + term_pv
    
    # === 5. 详细计算过程展示 (v2.1 增强) ===
    with st.expander("📝 查看详细计算过程 (5 Year Projection)", expanded=True):
        # 表格化展示
        calc_data = {
            "年份": [f"Y{i}" for i in range(1, 6)],
            f"FCF 预测 ({unit_label})": [f"{f:,.2f}" for f in flows],
            "YoY 增长率": [f"{y:+.1f}%" for y in yoy_rates],
            f"折现值 PV ({unit_label})": [f"{p:,.2f}" for p in pvs],
            "折现因子": [f"{1/((1+wacc)**i):.4f}" for i in range(1, 6)]
        }
        st.dataframe(pd.DataFrame(calc_data), use_container_width=True, hide_index=True)
        st.caption("ℹ️ **折现因子 (Discount Factor)**: 代表未来 1 元钱在今天的价值。公式: $1 / (1 + WACC)^t$")
        
        # Terminal 详细计算过程
        st.markdown("##### 📐 Terminal Value 计算过程")
        st.markdown(f"""
| 步骤 | 公式 | 计算 | 结果 ({unit_label}) |
|------|------|------|------|
| 1. Y5 FCF | — | — | {flows[-1]:,.2f} |
| 2. 永续 FCF | FCF₅ × (1+g) | {flows[-1]:,.2f} × (1+{perp_rate:.2%}) | {term_fcf:,.2f} |
| 3. 终值 (TV) | FCF₆ / (WACC-g) | {term_fcf:,.2f} / ({wacc:.2%} - {perp_rate:.2%}) | {term_val:,.2f} |
| 4. 终值现值 | TV / (1+WACC)⁵ | {term_val:,.2f} / (1+{wacc:.2%})⁵ | {term_pv:,.2f} |
        """)
    
    st.divider()
    
    # 6. 结果展示 (v2.1 - 含市值对比)
    term_mix = term_pv / enterprise_value if enterprise_value > 0 else 0
    
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric(f"企业价值 (EV)", f"{enterprise_value:,.2f} {unit_label}")
    rc2.metric("阶段1 现值 (1-5Y)", f"{total_pv:,.2f} {unit_label}", f"占比 {total_pv/enterprise_value:.1%}")
    rc3.metric("终值 现值 (Terminal)", f"{term_pv:,.2f} {unit_label}", f"占比 {term_mix:.1%}")
    
    # v2.1 市值对比
    if market_cap > 0:
        # 单位对齐：market_cap 是美元，EV 可能是 Billion
        if enterprise_value < 10000:
            ev_dollars = enterprise_value * 1e9
        else:
            ev_dollars = enterprise_value
        diff_pct = (ev_dollars / market_cap - 1) * 100
        rc4.metric("vs 当前市值", f"{diff_pct:+.1f}%", 
                    "低估" if diff_pct > 0 else "高估",
                    delta_color="normal")
    
    st.info(f"💡 货币单位: {unit_label} | 永续增长率已参考无风险利率 {rf}% 限制 | 增长率来源: {growth_choice}")
    
    # === 7. 可视化展示 (v2.1 增强) ===
    st.markdown("#### 📊 估值构成可视化")
    
    # A. 双轴图：FCF 预测折线 + PV 柱状 + 增长率
    fig_dcf = make_subplots(specs=[[{"secondary_y": True}]])
    
    x_labels = [f"Y{i}" for i in range(1, 6)] + ["Terminal"]
    pv_all = pvs + [term_pv]
    
    # 柱状图：PV 贡献
    fig_dcf.add_trace(go.Bar(
        x=x_labels, 
        y=pv_all,
        text=[f"{v:,.0f}" for v in pv_all],
        textposition='auto',
        marker_color=['#60A5FA']*5 + ['#34D399'],
        name=f"折现值 PV ({unit_label})",
        hovertemplate="%{x}: %{y:,.2f} " + unit_label + "<extra></extra>"
    ), secondary_y=False)
    
    # 折线图：FCF 预测趋势
    fig_dcf.add_trace(go.Scatter(
        x=[f"Y{i}" for i in range(1, 6)],
        y=flows,
        mode='lines+markers+text',
        text=[f"{f:,.0f}" for f in flows],
        textposition='top center',
        name=f"FCF 预测 ({unit_label})",
        line=dict(color='#F59E0B', width=3),
        marker=dict(size=8)
    ), secondary_y=False)
    
    # 增长率折线 (副轴)
    fig_dcf.add_trace(go.Scatter(
        x=[f"Y{i}" for i in range(1, 6)],
        y=yoy_rates,
        mode='lines+markers+text',
        text=[f"{r:.1f}%" for r in yoy_rates],
        textposition='bottom center',
        name="YoY 增长率 (%)",
        line=dict(color='#EF4444', width=2, dash='dot'),
        marker=dict(size=6)
    ), secondary_y=True)
    
    fig_dcf.update_layout(
        title=f"DCF 估值构成 (企业价值: {enterprise_value:,.0f} {unit_label})",
        height=450,
        legend=dict(orientation="h", y=1.15, x=0),
        hovermode="x unified",
        barmode='overlay'
    )
    fig_dcf.update_xaxes(title_text="预测年份")
    fig_dcf.update_yaxes(title_text=f"金额 ({unit_label})", secondary_y=False)
    fig_dcf.update_yaxes(title_text="增长率 (%)", secondary_y=True)
    st.plotly_chart(fig_dcf, use_container_width=True)
    
    # B. 市值对比图 (v2.1 新增)
    if market_cap > 0:
        st.markdown("#### 📊 DCF 企业价值 vs 当前市值")
        
        ev_in_b = enterprise_value if enterprise_value < 10000 else enterprise_value / 1e9
        mc_in_b = market_cap / 1e9
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=["DCF 企业价值", "当前市值"],
            y=[ev_in_b, mc_in_b],
            text=[f"{ev_in_b:,.1f}B", f"{mc_in_b:,.1f}B"],
            textposition='auto',
            marker_color=['#3B82F6', '#EF4444'],
            width=0.4
        ))
        
        diff_val = ev_in_b - mc_in_b
        diff_pct_val = (ev_in_b / mc_in_b - 1) * 100 if mc_in_b > 0 else 0
        
        fig_comp.update_layout(
            title=f"DCF EV vs 市值 (差异: {diff_pct_val:+.1f}%, {diff_val:+.1f}B)",
            yaxis_title=f"金额 (Billion USD)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        if diff_pct_val > 10:
            st.success(f"📈 **结论**: DCF 估值高于当前市值 {diff_pct_val:.1f}%，暗示市场可能低估。")
        elif diff_pct_val < -10:
            st.warning(f"📉 **结论**: DCF 估值低于当前市值 {abs(diff_pct_val):.1f}%，暗示市场可能高估。")
        else:
            st.info(f"⚖️ **结论**: DCF 估值与当前市值差异在 ±10% 以内 ({diff_pct_val:+.1f}%)，估值基本合理。")
    
    # C. 分析报告
    st.markdown("#### 📝 估值分析报告")
    
    analysis_md = f"""
**1. 估值结果**
基于 **DCF 模型**，{ticker} 的推算企业价值 (Enterprise Value) 为 **{enterprise_value:,.2f} {unit_label}**。

**2. 核心假设**
- **基准现金流**: {init_fcf:,.2f} {unit_label} (来源: {fcf_source})
- **折现率 (WACC)**: {wacc*100:.2f}%
- **增长阶段**: 前5年 CAGR 为 {growth_rate*100:.1f}% ({growth_choice})，永续增长率为 {perp_rate*100:.1f}%

**3. 结构分析**
- **前5年增长**: 贡献了 {total_pv:,.2f} {unit_label} ({1-term_mix:.1%}) 的价值
- **永续阶段**: 终值折现后贡献了 {term_pv:,.2f} {unit_label} ({term_mix:.1%}) 的价值
    """
    
    if term_mix > 0.7:
        analysis_md += f"""
> ⚠️ **终值依赖度较高**: 超过 70% 的价值来自于永续阶段 (Terminal Value = {term_mix:.1%})。
> 这意味着估值对 **永续增长率** 和 **WACC** 的微小变化非常敏感，需谨慎评估这些长期假设。
        """
        
    st.markdown(analysis_md)

    # D. 敏感性分析 (v2.1 增强 - 更清晰)
    st.markdown("#### 🎯 敏感性分析 (Enterprise Value)")
    st.caption(f"所有数值单位: {unit_label}")
    
    # 构造矩阵
    wacc_range = [wacc - 0.02, wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01, wacc + 0.02]
    g_range = [perp_rate - 0.01, perp_rate - 0.005, perp_rate, perp_rate + 0.005, perp_rate + 0.01]
    
    # 确保 g < wacc
    valid_g = [g for g in g_range if g >= 0 and g < min(wacc_range)]
    if not valid_g: valid_g = [perp_rate]
    
    res_matrix = []
    for g in valid_g:
        row_vals = []
        for w in wacc_range:
            if w <= g:
                row_vals.append(None)
                continue
            # 重新计算
            curr_s = init_fcf
            pv_5y_sense = 0
            last_flow = 0
            for i in range(1, 6):
                curr_s = curr_s * (1 + growth_rate)
                pv_5y_sense += curr_s / ((1 + w) ** i)
                last_flow = curr_s
            term_val_sense = last_flow * (1 + g) / (w - g)
            term_pv_sense = term_val_sense / ((1 + w) ** 5)
            ev_sense = pv_5y_sense + term_pv_sense
            row_vals.append(ev_sense)
        res_matrix.append(row_vals)
    
    # Heatmap
    fig_sense = go.Figure(data=go.Heatmap(
        z=res_matrix,
        x=[f"WACC {w*100:.1f}%" for w in wacc_range],
        y=[f"g = {g*100:.2f}%" for g in valid_g],
        colorscale='RdYlGn',
        texttemplate="%{z:,.0f}",
        hoverongaps=False,
        colorbar=dict(title=f"EV ({unit_label})")
    ))
    
    fig_sense.update_layout(
        title=f"敏感性分析: WACC vs 永续增长率 → 企业价值 ({unit_label})",
        xaxis_title="WACC (折现率)",
        yaxis_title="永续增长率 (g)",
        height=350
    )
    
    st.plotly_chart(fig_sense, use_container_width=True)
    
    # 敏感性分析结论
    if res_matrix and all(r is not None for row in res_matrix for r in row if r is not None):
        all_vals = [v for row in res_matrix for v in row if v is not None]
        ev_min = min(all_vals)
        ev_max = max(all_vals)
        st.info(f"""
📊 **敏感性分析结论**: 在 WACC 从 {wacc_range[0]*100:.1f}% 到 {wacc_range[-1]*100:.1f}%、永续增长率从 {valid_g[0]*100:.2f}% 到 {valid_g[-1]*100:.2f}% 的范围内：
- 估值区间: **{ev_min:,.0f} ~ {ev_max:,.0f} {unit_label}**
- 基准估值: **{enterprise_value:,.0f} {unit_label}** (WACC={wacc*100:.1f}%, g={perp_rate*100:.1f}%)
- 估值弹性: WACC 每变动 0.5%，企业价值约变动 {abs(res_matrix[len(valid_g)//2][3] - res_matrix[len(valid_g)//2][4]):,.0f} {unit_label} (如果存在的话)
        """)

    # === v2.3: DCF 倒推分析 (从高级模型合并) ===
    st.divider()
    st.markdown("## 🔄 DCF 倒推分析 (Reverse DCF)")
    st.caption("以下内容基于当前市值倒推市场隐含的增长率预期，含敏感性矩阵。")
    _render_dcf_reverse(df_single_q, latest_q, meta, wacc, rf, unit_label, df_raw)
