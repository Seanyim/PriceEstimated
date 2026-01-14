# modules/config.py

# 财务指标定义 (用于数据库建表和UI生成)
FINANCIAL_METRICS = [
    # --- 核心增长指标 (需要计算单季度/YoY) ---
    {"id": "Revenue", "label": "营收 (Revenue)", "format": "%.3f", "default": 0.0},
    {"id": "Profit", "label": "净利润 (Net Income)", "format": "%.3f", "default": 0.0},
    {"id": "EPS", "label": "每股收益 (EPS)", "format": "%.3f", "default": 0.0},
    {"id": "FCF", "label": "自由现金流 (FCF)", "format": "%.3f", "default": 0.0},
    
    # --- WACC/债务/存量指标 ---
    {"id": "Cash", "label": "现金及等价物", "format": "%.3f", "default": 0.0},
    {"id": "Total_Debt", "label": "总债务 (Total Debt)", "format": "%.3f", "default": 0.0},
    {"id": "Interest_Expense", "label": "利息支出", "format": "%.3f", "default": 0.0},
    {"id": "Income_Tax", "label": "所得税费用", "format": "%.3f", "default": 0.0},
    {"id": "Pre_Tax_Income", "label": "税前利润", "format": "%.3f", "default": 0.0},
    
    # --- 手动补充的市场数据 (作为备份或基准) ---
    {"id": "Shares", "label": "总股本 (Shares)", "format": "%.3f", "default": 0.0},
    {"id": "Manual_Market_Cap", "label": "市值 (录入时快照)", "format": "%.3f", "default": 0.0},
]

# 用于计算的指标列表
GROWTH_METRIC_KEYS = ["Revenue", "Profit", "EPS", "FCF"]
ALL_METRIC_KEYS = [m["id"] for m in FINANCIAL_METRICS]