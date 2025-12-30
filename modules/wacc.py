import streamlit as st

def render_wacc_module(df):
    st.markdown("## 🧮 WACC 假设与计算")

    prefix = "wacc"

    # --- 宏观 ---
    st.markdown("### 🌍 宏观利率")

    rf = st.number_input(
        "无风险利率 Rf (%)",
        value=4.0,
        step=0.1,
        key=f"{prefix}_rf"
    ) / 100
    st.markdown("### 📈 市场风险参数")

    col1, col2, col3 = st.columns(3)

    beta = col1.number_input(
        "Beta",
        value=1.1,
        step=0.05,
        key=f"{prefix}_beta"
    )

    erp = col2.number_input(
        "ERP (%)",
        value=5.5,
        step=0.1,
        key=f"{prefix}_erp"
    ) / 100

    credit_spread = col3.number_input(
        "信用利差 (%)",
        value=0.6,
        step=0.05,
        key=f"{prefix}_credit"
    ) / 100
    st.markdown("### 🏗 资本结构 & 税率")

    col4, col5 = st.columns(2)

    tax_rate = col4.number_input(
        "有效税率 (%)",
        value=21.0,
        step=0.5,
        key=f"{prefix}_tax"
    ) / 100

    equity_weight = col5.number_input(
        "权益占比 E/V (%)",
        value=85.0,
        step=1.0,
        key=f"{prefix}_ev"
    ) / 100
    cost_of_equity = rf + beta * erp
    rd = rf + credit_spread
    after_tax_rd = rd * (1 - tax_rate)
    debt_weight = 1 - equity_weight

    wacc = (
        equity_weight * cost_of_equity +
        debt_weight * after_tax_rd
    )
    st.markdown("### 📊 WACC 计算结果")

    col6, col7, col8 = st.columns(3)

    col6.metric("股权成本 Re", f"{cost_of_equity*100:.2f}%")
    col7.metric("税后债务成本 Rd", f"{after_tax_rd*100:.2f}%")
    col8.metric("WACC", f"{wacc*100:.2f}%")

    return wacc
