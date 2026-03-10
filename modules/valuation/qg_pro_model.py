import pandas as pd
import numpy as np
from typing import Dict, Optional
from modules.valuation.valuation_advanced import safe_get
from modules.valuation.master_analysis import linear_scale, _weighted_score

def compute_qg_pro_score(df_single: pd.DataFrame, latest: pd.Series) -> Dict:
    """
    计算机构级 QG-Pro (Quality Growth Pro) 因子得分及其底层细节。
    使用基于基本面的绝对阈值 (Absolute Threshold Scoring)，适配单一个股评价场景。
    
    包含以下四个正交子维度：
    1. 增长与加速因子 (Growth & Acceleration, G_adj)
    2. 下行风险半方差 (Downside Semi-Variance, S_down)
    3. 连续暴雷风险 (Continuous Drawdown Risk, D_risk)
    4. 盈余质量因子 (Earnings Quality, CF_quality)
    """
    results = {}
    factors = {}
    
    # 强制使用单季度同比数据 (Quarterly YoY) 计算增长动能
    if 'NetIncome_YoY' in df_single.columns and not df_single['NetIncome_YoY'].isna().all():
        g_series = df_single['NetIncome_YoY'].dropna()
        g_name = "净利润"
    elif 'TotalRevenue_YoY' in df_single.columns and not df_single['TotalRevenue_YoY'].isna().all():
        g_series = df_single['TotalRevenue_YoY'].dropna()
        g_name = "营收"
    else:
        g_series = pd.Series(dtype=float)
        g_name = "基本面"
        
    g_t = 0.0
    g_t_1 = 0.0
    
    if len(g_series) >= 1:
        g_t = g_series.iloc[-1]
    if len(g_series) >= 2:
        g_t_1 = g_series.iloc[-2]
        
    # ========================================
    # 1. 增长与加速因子 G_adj
    # 严格使用单季度 YoY。绝对阈值: bad=0.0, target=0.15, excellent=0.30
    # ========================================
    g_adj = None
    g_adj_score = None
    if len(g_series) >= 1:
        # G_adj = sign(g_t) * ln(1+|g_t|) + 0.5 * (g_t - g_{t-1})
        g_adj = np.sign(g_t) * np.log1p(abs(g_t)) + 0.5 * (g_t - g_t_1)
        g_adj_score = linear_scale(g_adj, bad=0.0, target=0.15, excellent=0.30)
        factors[f"单季{g_name}同比 (g_t)"] = f"{g_t:+.1%}"
        factors[f"上单季{g_name}同比 (g_t-1)"] = f"{g_t_1:+.1%}"
        factors["增长与加速因子 (G_adj)"] = f"{g_adj:.3f}"
        
    # ========================================
    # 2. 下行风险半方差 S_down
    # 严格使用单季度 YoY。绝对阈值: bad=0.04, target=0.01, excellent=0.0
    # ========================================
    s_down = None
    s_down_score = None
    if len(g_series) >= 4:
        downside_g = np.minimum(g_series.iloc[-8:], 0) # 计算近8个季度的下行半方差
        s_down = np.var(downside_g)
        s_down_score = linear_scale(s_down, bad=0.04, target=0.01, excellent=0.0, reverse=True)
        factors["下行风险半方差 (S_down)"] = f"{s_down:.4f}"
        
    # ========================================
    # 3. 连续暴雷风险 D_risk
    # 严格使用单季度 YoY。如果有两期 < 0，得 0 分；否则得 100 分。
    # ========================================
    d_risk = None
    d_risk_score = None
    if len(g_series) >= 2:
        d_risk = 1.0 if (g_t < 0 and g_t_1 < 0) else 0.0
        d_risk_score = 0.0 if d_risk == 1.0 else 10.0 # linear_scale 为1-10对应10-100
        factors["连续暴雷风险 (D_risk)"] = "高风险 ⚠️" if d_risk == 1.0 else "正常 ✅"
        
    # ========================================
    # 4. 盈余质量因子 CF_quality
    # 强制使用 TTM 消除季节性。绝对阈值: bad=0.0, target=0.8, excellent=1.2
    # ========================================
    cf_quality = None
    cf_quality_score = None
    ocf = safe_get(latest, 'OperatingCashFlow_TTM', 0)
    net_profit = safe_get(latest, 'NetIncome_TTM', 0)
    
    if pd.notna(ocf) and pd.notna(net_profit):
        epsilon = 1e-4
        cf_quality = ocf / (abs(net_profit) + epsilon)
        # 如果 net_profit 极小，OCF > 0 给好评
        if abs(net_profit) < 1e-2 and ocf > 0:
            cf_quality_score = 10.0
        elif abs(net_profit) < 1e-2 and ocf <= 0:
            cf_quality_score = 0.0
        else:
            cf_quality_score = linear_scale(cf_quality, bad=0.0, target=0.8, excellent=1.2)
            
        factors["经营现金流TTM (OCF)"] = f"{ocf:,.1f}"
        factors["净利润TTM (Net Profit)"] = f"{net_profit:,.1f}"
        factors["盈余质量 (CF_quality)"] = f"{cf_quality:.2f}x"
        
    # ========================================
    # 综合计分与权重
    # ========================================
    score, status = _weighted_score([
        (g_adj_score, 0.40, "增长与加速"),
        (s_down_score, 0.25, "下行半方差"),
        (d_risk_score, 0.20, "暴雷风险"),
        (cf_quality_score, 0.15, "盈余质量"),
    ])
    factors.update(status)
    has_any = any(s is not None for s in [g_adj_score, s_down_score, d_risk_score, cf_quality_score])
    
    # 记录细分维度分值，范围 0-100 (用于雷达图)
    dim_scores = {
        "G_adj": g_adj_score * 10 if g_adj_score is not None else 50,
        "S_down": s_down_score * 10 if s_down_score is not None else 50,
        "D_risk": d_risk_score * 10 if d_risk_score is not None else 50,
        "CF_quality": cf_quality_score * 10 if cf_quality_score is not None else 50,
    }
    
    results = {
        "score": score,
        "factors": factors,
        "available": has_any,
        "dim_scores": dim_scores
    }
    
    return results

