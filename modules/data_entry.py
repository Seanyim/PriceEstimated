import streamlit as st
import pandas as pd
from datetime import date
from modules.config import FINANCIAL_METRICS
from modules.db import get_financial_records, save_financial_record, save_company_meta, get_company_meta
from modules.data_fetcher import get_fetcher

def render_entry_tab(selected_company, unit_label):
    st.subheader(f"📝 {selected_company} - 财务数据录入 (SQLite 版)")
    
    # 1. 自动同步区域
    with st.expander("☁️ 市场数据自动同步", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info("将从 Yahoo Finance 获取：1. 每日收盘价历史 (Max)  2. 最新市值 & EPS TTM 快照")
        with c2:
            if st.button("🚀 开始同步"):
                with st.spinner("Syncing..."):
                    fetcher = get_fetcher()
                    res = fetcher.sync_market_data(selected_company)
                    if "Error" in res["msg"]:
                        st.error(res["msg"])
                    else:
                        st.success(f"同步成功! {res['msg']}")
                        st.rerun()
        
        # 显示当前数据库中的快照信息
        meta = get_company_meta(selected_company)
        if meta.get('last_market_cap'):
            st.caption(f"当前库中快照: 市值 {meta['last_market_cap']/1e9:.2f}B | EPS-TTM {meta.get('last_eps_ttm', 0)}")

    st.markdown("---")

    # 2. 财务数据录入 (Cumulative Input)
    st.markdown("#### ➕ 录入累计财报 (Cumulative)")
    st.caption("系统将根据以下规则自动计算单季度数据：Q2=H1-Q1, Q3=Q9-H1, Q4=FY-Q9")
    
    # 基础选择
    c_base1, c_base2, c_base3 = st.columns(3)
    with c_base1:
        year_input = st.number_input("财年 (Year)", 2000, 2030, 2025)
    with c_base2:
        period_input = st.selectbox("累计周期", ["Q1", "H1", "Q9", "FY"])
    with c_base3:
        report_date_input = st.date_input("财报披露日", value=date.today())

    # 自动检测是否已有数据
    existing_records = get_financial_records(selected_company)
    existing_data = {}
    
    # 查找匹配记录
    for r in existing_records:
        if r['year'] == year_input and r['period'] == period_input:
            existing_data = r
            break
            
    if existing_data:
        st.info(f"💡 检测到 {year_input} {period_input} 已有数据，已自动回填。")

    # 动态表单
    with st.form("financial_form"):
        input_values = {}
        cols = st.columns(3)
        
        for i, m in enumerate(FINANCIAL_METRICS):
            # 从已有记录或Config默认值获取
            default_val = existing_data.get(m['id'], m['default'])
            
            with cols[i % 3]:
                val = st.number_input(
                    f"{m['label']}", 
                    value=float(default_val),
                    format=m['format'],
                    key=f"in_{m['id']}"
                )
                input_values[m['id']] = val
        
        submitted = st.form_submit_button("💾 保存/更新数据")
        
        if submitted:
            record = {
                "ticker": selected_company,
                "year": int(year_input),
                "period": period_input,
                "report_date": report_date_input.strftime("%Y-%m-%d")
            }
            record.update(input_values)
            
            if save_financial_record(record):
                st.success(f"已保存 {selected_company} {year_input} {period_input}")
                st.rerun()
            else:
                st.error("保存失败")

    # 3. 历史数据表格展示
    if existing_records:
        st.markdown("### 📋 已录入历史数据")
        df_show = pd.DataFrame(existing_records)
        # 简单排序展示
        p_map = {"Q1":1, "H1":2, "Q9":3, "FY":4}
        df_show['s'] = df_show['period'].map(p_map)
        df_show = df_show.sort_values(['year', 's'], ascending=[False, False])
        
        cols_to_show = ['year', 'period', 'report_date'] + [m['id'] for m in FINANCIAL_METRICS]
        st.dataframe(df_show[cols_to_show], use_container_width=True)