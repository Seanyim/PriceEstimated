import streamlit as st
import pandas as pd
from modules.core.db import (
    init_db, get_all_tickers, save_company_meta, get_financial_records, get_company_meta,
    get_categories_with_companies, get_all_categories, create_category, delete_category,
    rename_category, add_company_to_category, remove_company_from_category,
    delete_company, auto_assign_company_to_region_category
)
from modules.ui.data_entry import render_entry_tab
from modules.ui.charts import render_charts_tab
from modules.valuation.valuation_PE import render_valuation_PE_tab
from modules.valuation.valuation_DCF import render_valuation_DCF_tab
from modules.valuation.valuation_analyst import render_analyst_tab
from modules.valuation.valuation_advanced import render_advanced_valuation_tab
from modules.core.wacc import render_wacc_module
from modules.ai.analysis import render_ai_tab

st.set_page_config(page_title="Valuation Pro v2.1", layout="wide")
st.title("📊 企业估值系统 v2.1")

# 初始化数据库
init_db()

# --- 侧边栏 ---
st.sidebar.header("🏢 公司管理")

# 1. 新建公司 (v2.1 - 添加地区选择 + 自动分组)
with st.sidebar.expander("➕ 添加/更新公司", expanded=False):
    with st.form("add_company"):
        new_ticker = st.text_input("Ticker (e.g. AAPL)").upper()
        new_name = st.text_input("公司名称 (e.g. Apple)")
        new_region = st.selectbox(
            "地区/市场", 
            ["US", "CN", "HK", "JP", "TW"],
            format_func=lambda x: {
                "US": "🇺🇸 美国",
                "CN": "🇨🇳 中国大陆",
                "HK": "🇭🇰 香港",
                "JP": "🇯🇵 日本",
                "TW": "🇹🇼 台湾"
            }.get(x, x)
        )
        new_unit = st.selectbox("单位", ["Billion", "Million"])
        if st.form_submit_button("添加/更新公司"):
            if new_ticker:
                save_company_meta(new_ticker, new_name, new_unit, new_region)
                # v2.1: 自动分配到对应地区分组
                auto_assign_company_to_region_category(new_ticker, new_region)
                st.success(f"已添加 {new_ticker} ({new_region})")
                st.rerun()

# 2. 按分组选择公司 (v2.1)
categories_data = get_categories_with_companies()
all_tickers = get_all_tickers()

if not all_tickers:
    st.info("请先添加公司")
    st.stop()

# 构建分组化的选项列表
grouped_options = []  # [(display_label, ticker), ...]
for cat in categories_data:
    if cat["companies"]:
        for comp in cat["companies"]:
            label = f"[{cat['name']}] {comp['ticker']} - {comp['name']}"
            grouped_options.append((label, comp["ticker"]))

# 如果有分组数据，使用分组选择器
if grouped_options:
    display_labels = [opt[0] for opt in grouped_options]
    ticker_map = {opt[0]: opt[1] for opt in grouped_options}
    
    selected_label = st.sidebar.selectbox("选择公司", display_labels)
    selected_company = ticker_map[selected_label]
else:
    # 回退到简单列表
    selected_company = st.sidebar.selectbox("选择公司", all_tickers)

meta = get_company_meta(selected_company)
current_unit = meta.get('unit', 'Billion')
current_region = meta.get('region', 'US')

# 显示公司信息
region_flags = {
    "US": "🇺🇸", "CN": "🇨🇳", "HK": "🇭🇰", "JP": "🇯🇵", "TW": "🇹🇼"
}
st.sidebar.markdown(f"**当前单位**: {current_unit} | **地区**: {region_flags.get(current_region, '')} {current_region}")

# 3. 分组管理 (v2.1)
with st.sidebar.expander("📁 分组管理", expanded=False):
    mgmt_tab1, mgmt_tab2, mgmt_tab3 = st.tabs(["管理分组", "管理成员", "删除公司"])
    
    with mgmt_tab1:
        # 创建新分组
        new_cat_name = st.text_input("新分组名称", key="new_cat_name")
        if st.button("创建分组", key="btn_create_cat"):
            if new_cat_name.strip():
                if create_category(new_cat_name.strip()):
                    st.success(f"已创建分组: {new_cat_name}")
                    st.rerun()
                else:
                    st.error("分组名称已存在")
        
        st.markdown("---")
        # 删除/重命名现有分组
        existing_cats = get_all_categories()
        if existing_cats:
            cat_names = {c["name"]: c["id"] for c in existing_cats}
            selected_cat_for_edit = st.selectbox("选择分组操作", list(cat_names.keys()), key="cat_edit_select")
            
            col_rename, col_delete = st.columns(2)
            with col_rename:
                rename_val = st.text_input("重命名为", key="rename_cat_val")
                if st.button("重命名", key="btn_rename_cat"):
                    if rename_val.strip():
                        cat_id = cat_names[selected_cat_for_edit]
                        if rename_category(cat_id, rename_val.strip()):
                            st.success("已重命名")
                            st.rerun()
                        else:
                            st.error("名称重复")
            with col_delete:
                if st.button("🗑️ 删除分组", key="btn_delete_cat"):
                    cat_id = cat_names[selected_cat_for_edit]
                    if delete_category(cat_id):
                        st.success(f"已删除分组 (公司数据保留)")
                        st.rerun()
    
    with mgmt_tab2:
        # 添加/移除公司到分组
        existing_cats = get_all_categories()
        if existing_cats and all_tickers:
            cat_options = {c["name"]: c["id"] for c in existing_cats}
            target_cat = st.selectbox("目标分组", list(cat_options.keys()), key="member_target_cat")
            target_ticker = st.selectbox("公司", all_tickers, key="member_ticker")
            
            col_add, col_remove = st.columns(2)
            with col_add:
                if st.button("➕ 添加到分组", key="btn_add_member"):
                    add_company_to_category(cat_options[target_cat], target_ticker)
                    st.success(f"已添加 {target_ticker} → {target_cat}")
                    st.rerun()
            with col_remove:
                if st.button("➖ 从分组移除", key="btn_remove_member"):
                    remove_company_from_category(cat_options[target_cat], target_ticker)
                    st.success(f"已移除 {target_ticker} ← {target_cat} (数据保留)")
                    st.rerun()
    
    with mgmt_tab3:
        # 彻底删除公司
        st.warning("⚠️ 此操作将从数据库中彻底删除公司及所有关联数据，不可恢复！")
        del_ticker = st.selectbox("选择要删除的公司", all_tickers, key="del_ticker")
        confirm_del = st.checkbox(f"确认删除 {del_ticker} 及其所有数据", key="confirm_del")
        if st.button("🗑️ 彻底删除", key="btn_delete_company", type="primary"):
            if confirm_del:
                if delete_company(del_ticker):
                    st.success(f"已彻底删除 {del_ticker}")
                    st.rerun()
            else:
                st.error("请先勾选确认")

st.sidebar.markdown("---")

# 4. API 配置区域
st.sidebar.subheader("⚙️ API 配置")

# Proxy 设置
proxy = st.sidebar.text_input("Proxy URL", value="http://127.0.0.1:10808", key="proxy_url")

st.sidebar.caption("💡 Proxy 用于 yfinance 数据获取")

# 读取财务数据
raw_records = get_financial_records(selected_company)
df_raw = pd.DataFrame(raw_records)

# --- 主界面 ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 数据录入", "📈 趋势分析", "🧮 估值模型", "🤖 AI 分析"])

with tab1:
    render_entry_tab(selected_company, current_unit)

with tab2:
    render_charts_tab(df_raw, current_unit)

with tab3:
    # WACC 模块（在顶部，供所有子 Tab 使用）
    wacc, rf = render_wacc_module(df_raw)
    
    st.divider()
    
    # 估值模型子 Tab
    val_tab1, val_tab2, val_tab3, val_tab4 = st.tabs([
        "📉 PE 估值", 
        "🚀 DCF 估值",
        "🔬 高级模型",
        "📊 分析师预测"
    ])
    
    with val_tab1:
        render_valuation_PE_tab(df_raw, current_unit)
        
    with val_tab2:
        render_valuation_DCF_tab(df_raw, wacc, rf, current_unit)
    
    with val_tab3:
        render_advanced_valuation_tab(df_raw, current_unit, wacc, rf)
    
    with val_tab4:
        render_analyst_tab(selected_company, df_raw)

with tab4:
    render_ai_tab(selected_company, df_raw)
