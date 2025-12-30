# modules/config.py

# 定义所有需要录入的财务指标
# id: JSON中的键名 (英文)
# label: 显示在界面的名称 (中文)
# help: 提示信息
# default: 默认值
FINANCIAL_METRICS = [
    {
        "id": "Revenue",
        "label": "累计营收",
        "help": "公司的主营业务收入",
        "default": 0.0
    },
    {
        "id": "Profit",
        "label": "累计净利润",
        "help": "归属于母公司股东的净利润",
        "default": 0.0
    },
    {
        "id": "EPS",
        "label": "累计 EPS",
        "help": "每股收益",
        "default": 0.0
    },
    {
        "id": "FCF",
        "label": "累计自由现金流",
        "help": "经营现金流净额 - 资本开支 (Capex)",
        "default": 0.0
    },
    # --- 以后要加新指标，直接在这里追加即可 ---
    # {
    #     "id": "R_and_D",
    #     "label": "研发费用",
    #     "help": "Research and Development",
    #     "default": 0.0
    # }
]

# 提取纯 ID 列表供 calculator 使用
METRIC_KEYS = [m["id"] for m in FINANCIAL_METRICS]