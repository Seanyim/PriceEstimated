import streamlit as st
import pandas as pd
from data_manager import save_data
from modules.config import FINANCIAL_METRICS # 导入配置

def render_entry_tab(selected_company, data_store, unit_label):
    st.subheader(f"{selected_company} - 累计季报数据录入")
    
    records = data_store[selected_company]["records"]
    
    # --- 1. 基础字段 (固定) ---
    # 基础字段决定了记录的唯一性(Key)，不适合动态生成
    c_base1, c_base2 = st.columns(2)
    with c_base1:
        year_input = st.number_input("财年 (Year)", 2000, 2030, 2024)
    with c_base2:
        period_input = st.selectbox("报告周期 (累计)", ["Q1", "H1", "Q9", "FY"])
    
    st.markdown("---")
    
    # --- 2. 动态生成财务指标输入框 ---
    # 使用字典来收集用户的输入值
    input_values = {}
    
    # 创建 3 列布局的网格
    cols = st.columns(3)
    
    for i, metric in enumerate(FINANCIAL_METRICS):
        # 动态分配列：0->col1, 1->col2, 2->col3, 3->col1...
        current_col = cols[i % 3]
        
        with current_col:
            # 动态生成 label，带上单位
            label_text = f"{metric['label']} ({unit_label})" if metric['id'] != "EPS" else metric['label']
            
            val = st.number_input(
                label_text,
                min_value=0.0,
                value=float(metric['default']),
                format="%.3f",
                help=metric.get('help', ''),
                key=f"input_{metric['id']}" # 保证唯一key
            )
            input_values[metric['id']] = val

    st.markdown("---")

    # --- 3. 保存逻辑 (通用化) ---
    if st.button("保存数据", type="primary"):
        # 构建新记录
        new_rec = {
            "Year": int(year_input),
            "Period": period_input,
        }
        # 将动态收集到的指标合并进去
        new_rec.update(input_values)
        
        # 更新逻辑：覆盖旧数据
        # 这里的筛选逻辑不变：Year + Period 是联合主键
        updated = [r for r in records if not (r['Year'] == int(year_input) and r['Period'] == period_input)]
        updated.append(new_rec)
        
        data_store[selected_company]["records"] = updated
        save_data(data_store)
        st.success(f"已保存 {year_input} {period_input}")
        st.rerun()
        
    # --- 4. 表格展示 (动态列) ---
    if records:
        df = pd.DataFrame(records)
        p_map = {"Q1":1, "H1":2, "Q9":3, "FY":4}
        df['s'] = df['Period'].map(p_map)
        df = df.sort_values(['Year', 's']).drop(columns=['s'])
        
        # 动态构建显示的列顺序
        # 基础列 + 配置中定义的指标列
        base_cols = ["Year", "Period"]
        metric_cols = [m["id"] for m in FINANCIAL_METRICS]
        
        # 确保只显示 DataFrame 中实际存在的列 (兼容旧数据)
        final_cols = base_cols + [c for c in metric_cols if c in df.columns]
        
        # 格式化字典：所有指标都保留2位小数
        format_dict = {m: "{:.2f}" for m in metric_cols}
        
        st.dataframe(df[final_cols].style.format(format_dict))