import streamlit as st
import pandas as pd
from modules.core.db import (
    init_db, get_all_tickers, save_company_meta, get_financial_records, get_company_meta,
    get_categories_with_companies, get_all_categories, create_category, delete_category,
    rename_category, add_company_to_category, remove_company_from_category,
    delete_company, auto_assign_company_to_region_category,
    detect_region_from_ticker, get_companies_in_category, get_companies_not_in_category
)
from modules.ui.data_entry import render_entry_tab
from modules.ui.charts import render_charts_tab
from modules.valuation.valuation_PE import render_valuation_PE_tab
from modules.valuation.valuation_DCF import render_valuation_DCF_tab
from modules.valuation.valuation_analyst import render_analyst_tab
from modules.valuation.valuation_summary import render_summary_tab
from modules.valuation.valuation_dashboard import render_dashboard_tab
from modules.valuation.master_analysis import render_master_analysis_tab
from modules.valuation.valuation_advanced import (
    _render_ev_ebitda, _render_growth_analysis, 
    _render_monte_carlo, _render_profitability_analysis,
    safe_get
)
from modules.core.calculator import process_financial_data
from modules.core.wacc import render_wacc_module

st.set_page_config(page_title="Valuation Pro v2.5", layout="wide")
st.title("📊 企业估值系统 v2.5")

# 初始化数据库
init_db()

# --- 侧边栏 ---
st.sidebar.header("🏢 公司管理")

# 1. 新建公司 (v2.2 - 智能地区推断 + 自动分组)
with st.sidebar.expander("➕ 添加/更新公司", expanded=False):
    new_ticker = st.text_input("Ticker (e.g. AAPL, 600519.SS, 9988.HK)", key="add_ticker").upper()
    
    # v2.2: 根据 Ticker 后缀自动推断地区
    auto_detected_region = detect_region_from_ticker(new_ticker) if new_ticker else 'US'
    region_options = ["US", "CN", "HK", "JP", "TW"]
    default_region_idx = region_options.index(auto_detected_region) if auto_detected_region in region_options else 0
    
    with st.form("add_company"):
        new_name = st.text_input("公司名称 (e.g. Apple)")
        new_region = st.selectbox(
            "地区/市场", 
            region_options,
            index=default_region_idx,
            format_func=lambda x: {
                "US": "🇺🇸 美国",
                "CN": "🇨🇳 中国大陆",
                "HK": "🇭🇰 香港",
                "JP": "🇯🇵 日本",
                "TW": "🇹🇼 台湾"
            }.get(x, x)
        )
        # v2.2: 单位根据地区自动推断，无需手动选择
        st.caption("💡 财务数据单位自动处理：导入时系统自动识别 亿/万/百万 并统一转换")
        if st.form_submit_button("添加/更新公司"):
            if new_ticker:
                save_company_meta(new_ticker, new_name, region=new_region)
                # v2.2: 自动分配到对应地区分组
                auto_assign_company_to_region_category(new_ticker, new_region)
                st.success(f"已添加 {new_ticker} ({new_region})")
                st.rerun()

# 2. 按分组选择公司 (v2.2 - 两级联动：先选组，再选组内公司)
categories_data = get_categories_with_companies()
all_tickers = get_all_tickers()

if not all_tickers:
    st.info("请先添加公司")
    st.stop()

# v2.2: 构建分组列表（只显示有公司的分组）
available_categories = [cat for cat in categories_data if cat["companies"]]

if available_categories:
    # 第一级: 选择分组
    cat_names = [cat["name"] for cat in available_categories]
    selected_cat_name = st.sidebar.selectbox("📁 选择分组", cat_names, key="nav_category")
    
    # 找到对应分组的公司列表
    selected_cat_data = next((cat for cat in available_categories if cat["name"] == selected_cat_name), None)
    
    if selected_cat_data and selected_cat_data["companies"]:
        # 第二级: 选择组内公司
        company_options = [f"{comp['ticker']} - {comp['name']}" for comp in selected_cat_data["companies"]]
        ticker_map = {f"{comp['ticker']} - {comp['name']}": comp["ticker"] for comp in selected_cat_data["companies"]}
        
        selected_label = st.sidebar.selectbox("🏢 选择公司", company_options, key="nav_company")
        selected_company = ticker_map[selected_label]
    else:
        st.sidebar.warning("该分组暂无公司")
        st.stop()
else:
    # 回退到简单列表（无分组时）
    selected_company = st.sidebar.selectbox("🏢 选择公司", all_tickers, key="nav_company_fallback")

meta = get_company_meta(selected_company)
current_unit = meta.get('unit', 'Billion')
current_region = meta.get('region', 'US')

# 显示公司信息
region_flags = {
    "US": "🇺🇸", "CN": "🇨🇳", "HK": "🇭🇰", "JP": "🇯🇵", "TW": "🇹🇼"
}
st.sidebar.markdown(f"**当前单位**: {current_unit} | **地区**: {region_flags.get(current_region, '')} {current_region}")

# 3. 分组管理 (v2.2 - 优化交互逻辑)
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
        # v2.2: 优化成员管理 — 分添加/移除两个子区域
        existing_cats = get_all_categories()
        if existing_cats:
            cat_options = {c["name"]: c["id"] for c in existing_cats}
            target_cat = st.selectbox("目标分组", list(cat_options.keys()), key="member_target_cat")
            target_cat_id = cat_options[target_cat]
            
            # 添加区域：只显示不在该组的公司
            st.markdown("**➕ 添加公司到分组**")
            available_companies = get_companies_not_in_category(target_cat_id)
            if available_companies:
                add_options = [f"{c['ticker']} - {c['name']}" for c in available_companies]
                add_ticker_map = {f"{c['ticker']} - {c['name']}": c['ticker'] for c in available_companies}
                selected_add = st.selectbox("选择要添加的公司", add_options, key="member_add_select")
                if st.button("➕ 添加到分组", key="btn_add_member"):
                    add_company_to_category(target_cat_id, add_ticker_map[selected_add])
                    st.success(f"已添加 {add_ticker_map[selected_add]} → {target_cat}")
                    st.rerun()
            else:
                st.caption("✅ 所有公司已在该分组中")
            
            st.markdown("---")
            
            # 移除区域：只显示当前组内的公司
            st.markdown("**➖ 从分组移除公司**")
            current_members = get_companies_in_category(target_cat_id)
            if current_members:
                remove_options = [f"{c['ticker']} - {c['name']}" for c in current_members]
                remove_ticker_map = {f"{c['ticker']} - {c['name']}": c['ticker'] for c in current_members}
                selected_remove = st.selectbox("选择要移除的公司", remove_options, key="member_remove_select")
                if st.button("➖ 从分组移除", key="btn_remove_member"):
                    remove_company_from_category(target_cat_id, remove_ticker_map[selected_remove])
                    st.success(f"已移除 {remove_ticker_map[selected_remove]} ← {target_cat} (数据保留)")
                    st.rerun()
            else:
                st.caption("该分组暂无公司")
        else:
            st.info("请先创建分组")
    
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

# --- 主界面 (v2.5.2) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 数据录入", "📈 趋势分析", "🧮 估值模型", "🧠 大师分析", "📋 估值总结"])

with tab1:
    render_entry_tab(selected_company, current_unit)

with tab2:
    render_charts_tab(df_raw, current_unit)

with tab3:
    # WACC 模块（在顶部，供估值模型的所有子 Tab 使用）
    wacc, rf = render_wacc_module(df_raw)
    
    st.divider()
    
    # 估值模型子 Tab (移除了估值总结，保留前8个)
    vt1, vt2, vt3, vt4, vt5, vt6, vt7, vt8 = st.tabs([
        "📉 PE/PEG", 
        "🚀 DCF",
        "💹 EV/EBITDA",
        "📈 增长透视",
        "🎲 Monte Carlo",
        "📉 ROIC/ROA/ROE",
        "📊 分析师预测",
        "🔀 估值整合"
    ])
    
    with vt1:
        render_valuation_PE_tab(df_raw, current_unit)
        
    with vt2:
        render_valuation_DCF_tab(df_raw, wacc, rf, current_unit)
    
    with vt3:
        # EV/EBITDA 独立 Tab
        st.subheader("💹 EV/EBITDA 分析")
        if not df_raw.empty:
            _, _df_s = process_financial_data(df_raw)
            if not _df_s.empty:
                _latest = _df_s.iloc[-1]
                _meta = get_company_meta(selected_company)
                _render_ev_ebitda(_df_s, _latest, _meta, current_unit)
            else:
                st.warning("财务数据不足")
        else:
            st.warning("请先录入财务数据")
    
    with vt4:
        # 增长率透视独立 Tab
        st.subheader("📈 增长率透视")
        if not df_raw.empty:
            _, _df_s = process_financial_data(df_raw)
            if not _df_s.empty:
                _render_growth_analysis(_df_s, current_unit)
            else:
                st.warning("财务数据不足")
        else:
            st.warning("请先录入财务数据")
    
    with vt5:
        # Monte Carlo 独立 Tab
        st.subheader("🎲 Monte Carlo 模拟")
        if not df_raw.empty:
            _, _df_s = process_financial_data(df_raw)
            if not _df_s.empty:
                _latest = _df_s.iloc[-1]
                _meta = get_company_meta(selected_company)
                _render_monte_carlo(_df_s, _latest, _meta, wacc, current_unit)
            else:
                st.warning("财务数据不足")
        else:
            st.warning("请先录入财务数据")
    
    with vt6:
        # ROIC/ROA/ROE 独立 Tab
        st.subheader("📉 ROIC/ROA/ROE 分析")
        if not df_raw.empty:
            _, _df_s = process_financial_data(df_raw)
            if not _df_s.empty:
                _render_profitability_analysis(_df_s, current_unit)
            else:
                st.warning("财务数据不足")
        else:
            st.warning("请先录入财务数据")
    
    with vt7:
        render_analyst_tab(selected_company, df_raw)
    
    with vt8:
        # 估值整合仪表盘 — 正推/倒推动态整合
        render_dashboard_tab(selected_company, df_raw, current_unit, wacc, rf)

# 🛡️ NameError 防护: 如果 tab3 的 render_wacc_module 因异常未执行完毕，
# wacc / rf 变量不存在，使用保守默认值做降级回退 (WACC=10%, Rf=4%)
safe_wacc = wacc if 'wacc' in dir() else 0.10
safe_rf = rf if 'rf' in dir() else 0.04

with tab4:
    # 九大投资大师多维分析（先于估值总结，结果供总结使用）
    render_master_analysis_tab(selected_company, df_raw, current_unit, safe_wacc, safe_rf)

with tab5:
    # 估值总结作为最终汇总 Tab — 集成大师评分 + 敏感性分析
    render_summary_tab(selected_company, df_raw, current_unit, safe_wacc, safe_rf)
