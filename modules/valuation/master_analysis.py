# modules/valuation/master_analysis.py
# 九大投资大师多维分析模块 v2.5.1
# 修复 4 项致命缺陷：量纲归一化、缺失值动态权重、Soros 因子修正
# 基于 master_index_quats.md 的打分方法论

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from modules.core.calculator import process_financial_data
from modules.core.db import get_company_meta, get_market_history
from modules.data.industry_data import get_industry_benchmarks
from modules.valuation.valuation_advanced import safe_get


# ============================================================
# 1. 九大师评分公式定义 (v2.5.1 — 修正 Soros 因子)
# ============================================================

MASTER_DEFINITIONS = {
    "Buffett": {
        "name_cn": "沃伦·巴菲特",
        "philosophy": "护城河与现金回报",
        "icon": "🏰",
        "color": "#2E86AB",
        "formula": r"Score = α₁·S(ROE_Stability) + α₂·S(FCF_Mean) - α₃·S(GM_σ)",
        "factors": ["ROE 稳定性", "FCF 均值", "毛利率波动"],
        "weights": [0.45, 0.35, 0.20],
    },
    "Munger": {
        "name_cn": "查理·芒格",
        "philosophy": "质量风控与反转",
        "icon": "🛡️",
        "color": "#A23B72",
        "formula": r"Score = β₁·S(ROIC) + β₂·S(D/E↓) + β₃·S(FCF_Conv)",
        "factors": ["ROIC", "Debt/Equity (↓越低越好)", "FCF 转换率"],
        "weights": [0.40, 0.30, 0.30],
    },
    "Lynch": {
        "name_cn": "彼得·林奇",
        "philosophy": "动态 GARP",
        "icon": "📈",
        "color": "#F18F01",
        "formula": r"Score = γ₁·S(PEG↓) + γ₂·S(EPS_Trend)",
        "factors": ["调整后 PEG (↓越低越好)", "EPS 变化趋势"],
        "weights": [0.60, 0.40],
    },
    "Graham": {
        "name_cn": "本杰明·格雷厄姆",
        "philosophy": "深度价值与安全边际",
        "icon": "🔒",
        "color": "#C73E1D",
        "formula": r"Score = δ₁·S(NCAV/MCap) + δ₂·S(P/B↓)",
        "factors": ["调整后 NCAV/市值", "P/B 比率 (↓越低越好)"],
        "weights": [0.55, 0.45],
    },
    "Greenblatt": {
        "name_cn": "乔尔·格林布拉特",
        "philosophy": "神奇公式",
        "icon": "✨",
        "color": "#2D936C",
        "formula": r"Score = 0.5·S(ROC) + 0.5·S(EY)",
        "factors": ["资本回报率", "盈利收益率"],
        "weights": [0.50, 0.50],
    },
    "Fisher": {
        "name_cn": "菲利普·费雪",
        "philosophy": "极速成长与创新",
        "icon": "🚀",
        "color": "#6B4226",
        "formula": r"Score = ω₁·S(Sales_CAGR) + ω₂·S(R&D_Eff)",
        "factors": ["营收 CAGR", "研发/营收增长效率"],
        "weights": [0.55, 0.45],
    },
    "Templeton": {
        "name_cn": "约翰·邓普顿",
        "philosophy": "逆向估值与均值回归",
        "icon": "🔄",
        "color": "#5C4D7D",
        "formula": r"Score = φ₁·S(PE/PE_Ind↓) + φ₂·S(Price_Pctile↓)",
        "factors": ["PE 相对行业 (↓越低越好)", "价格历史分位 (↓越低越好)"],
        "weights": [0.50, 0.50],
    },
    "Dalio": {
        "name_cn": "瑞·达里奥",
        "philosophy": "宏观稳健与债务杠杆",
        "icon": "🌊",
        "color": "#1B4965",
        "formula": r"Score = ψ₁·S(FCF/Debt) + ψ₂·S(ND/EBITDA↓)",
        "factors": ["FCF/Debt 覆盖", "Net Debt/EBITDA (↓越低越好)"],
        "weights": [0.55, 0.45],
    },
    "Soros": {
        "name_cn": "乔治·索罗斯",
        "philosophy": "动量与反身性",
        "icon": "⚡",
        "color": "#E63946",
        # v2.5.1 修正: FCF趋势 → 均线乖离率 (反身性代理)
        "formula": r"Score = κ₁·S(Momentum_{12M-1M}) + κ₂·S(Price/MA200)",
        "factors": ["价格动量 (12M-1M)", "均线乖离率 (Price/MA200)"],
        "weights": [0.55, 0.45],
    },
}


# ============================================================
# 2. 数学工具函数 (v2.5.1 — 截断线性插值 + 动态权重)
# ============================================================

def linear_scale(value: float, bad: float, target: float, 
                 excellent: float, reverse: bool = False) -> float:
    """
    截断线性插值 — 将任意异构因子映射到统一的 [0, 10] 分体系
    
    三段映射:
      value <= bad       → 0.0
      bad < value < target → 0~5 (线性插值)
      target <= value < excellent → 5~10 (线性插值)
      value >= excellent → 10.0
    
    Args:
        value: 原始因子值
        bad: 差值阈值 (映射到 0 分)
        target: 中位目标值 (映射到 5 分)
        excellent: 优秀阈值 (映射到 10 分)
        reverse: 如果为 True，值越低越好 (如 P/B、D/E)
    """
    if pd.isna(value):
        return None  # 返回 None 表示不可用，由 _weighted_score 处理
    
    if reverse:
        # 反转：值越低越好 → 翻转所有阈值
        value, bad, target, excellent = -value, -bad, -target, -excellent
    
    if value <= bad:
        return 0.0
    elif value < target:
        denom = target - bad
        return 5.0 * (value - bad) / denom if denom != 0 else 2.5
    elif value < excellent:
        denom = excellent - target
        return 5.0 + 5.0 * (value - target) / denom if denom != 0 else 7.5
    else:
        return 10.0


def _weighted_score(scores_and_weights: List[Tuple[Optional[float], float, str]]) -> Tuple[float, Dict[str, str]]:
    """
    动态权重归一化 — 贝叶斯降级平滑
    
    如果某个因子的得分为 None (数据缺失)，则该因子不参与计算，
    其余因子的权重等比例归一化放大。
    
    Args:
        scores_and_weights: [(score_or_None, weight, factor_name), ...]
        
    Returns:
        (final_score_0_100, status_dict)
        status_dict 包含降级信息
    """
    available = [(s, w, n) for s, w, n in scores_and_weights if s is not None]
    
    if not available:
        return 50.0, {"⚠️ 降级": "所有因子数据缺失，使用中性分数 50"}
    
    total_weight = sum(w for _, w, _ in available)
    
    if total_weight == 0:
        return 50.0, {"⚠️ 降级": "权重总和为 0"}
    
    # 等比例归一化 + 加权求和
    weighted_sum = sum(s * (w / total_weight) for s, w, _ in available)
    
    # 从 [0, 10] 映射到 [0, 100]
    final = float(np.clip(weighted_sum * 10, 0, 100))
    
    status = {}
    missing = [(n, w) for s, w, n in scores_and_weights if s is None]
    if missing:
        names = ", ".join(f"{n}(w={w:.0%})" for n, w in missing)
        status["⚠️ 降级"] = f"因子缺失: {names}，剩余权重等比例放大"
    
    return final, status


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    if b == 0 or pd.isna(b) or pd.isna(a):
        return default
    return a / b


def _get_ma_deviation(prices: pd.Series, current_price: float) -> Optional[float]:
    """
    安全计算均线乖离率 (Price/MA - 1)
    
    🛡️ 新股长度保护:
    - len(prices) >= 200: 使用 MA200
    - 50 <= len(prices) < 200: 降级到 MA50
    - len(prices) < 50: 返回 None → 触发 _weighted_score 动态权重分配
    
    Args:
        prices: 价格序列 (已 dropna)
        current_price: 当前价格
        
    Returns:
        均线乖离率 float, 或 None (数据不足)
    """
    if current_price <= 0 or len(prices) == 0:
        return None
    
    if len(prices) >= 200:
        ma = prices.iloc[-200:].mean()
    elif len(prices) >= 50:
        ma = prices.iloc[-50:].mean()
    else:
        return None  # 交易日不足，无法计算 → _weighted_score 自动降级
    
    if pd.isna(ma) or ma == 0:
        return None
    
    return (current_price / ma) - 1.0


# ============================================================
# 3. 核心打分引擎 (v2.5.1 — 全面重写)
# ============================================================

def compute_master_scores(df_single: pd.DataFrame, 
                          latest: pd.Series, 
                          meta: dict,
                          df_price: pd.DataFrame = None) -> Dict:
    """
    计算 9 位大师的分数
    
    v2.5.1 变更:
    - 使用 linear_scale 替代 Z-score 归一化
    - 使用 _weighted_score 实现缺失因子动态降级
    - 修正 Soros 因子: FCF趋势 → 均线乖离率
    
    Args:
        df_single: 单季度财务数据 (含 TTM)
        latest: 最新一行数据
        meta: 公司元数据
        df_price: 价格历史数据
        
    Returns:
        {master_key: {"score": float, "factors": dict, "available": bool}}
    """
    results = {}
    market_cap = meta.get('last_market_cap', 0)
    sector = meta.get('sector', 'Unknown')
    bench = get_industry_benchmarks(sector)
    
    # 获取当前股价
    current_price = 0
    if df_price is not None and not df_price.empty:
        current_price = df_price.iloc[-1].get('close', 0) or 0
    
    # 预计算常用指标
    eps_val = safe_get(latest, 'EPS_TTM', 0)
    pe_ttm = current_price / eps_val if eps_val > 0 and current_price > 0 else 0
    
    # 历史序列
    n_quarters = len(df_single)
    
    # --- 提取历史序列 ---
    roe_series = df_single['ROE'].dropna() if 'ROE' in df_single.columns else pd.Series(dtype=float)
    gm_series = df_single['GrossMargin'].dropna() if 'GrossMargin' in df_single.columns else pd.Series(dtype=float)
    fcf_ttm_series = df_single['FreeCashFlow_TTM'].dropna() if 'FreeCashFlow_TTM' in df_single.columns else pd.Series(dtype=float)
    roic_series = df_single['ROIC'].dropna() if 'ROIC' in df_single.columns else pd.Series(dtype=float)
    rev_ttm_series = df_single['TotalRevenue_TTM'].dropna() if 'TotalRevenue_TTM' in df_single.columns else pd.Series(dtype=float)
    ni_ttm_series = df_single['NetIncome_TTM'].dropna() if 'NetIncome_TTM' in df_single.columns else pd.Series(dtype=float)
    
    # 最新财务值
    fcf = safe_get(latest, 'FreeCashFlow_TTM', 0)
    ocf = safe_get(latest, 'OperatingCashFlow_TTM', 0)
    ni = safe_get(latest, 'NetIncome_TTM', 0)
    rev = safe_get(latest, 'TotalRevenue_TTM', 0)
    total_debt = safe_get(latest, 'TotalDebt', 0) or safe_get(latest, 'TotalLiabilities', 0)
    total_equity = safe_get(latest, 'TotalEquity', 0)
    total_assets = safe_get(latest, 'TotalAssets', 0)
    ebitda = safe_get(latest, 'EBITDA_TTM', 0) or safe_get(latest, 'OperatingProfit_TTM', 0)
    cash = safe_get(latest, 'CashEndOfPeriod', 0)
    current_assets = safe_get(latest, 'CurrentAssets', 0)
    current_liabilities = safe_get(latest, 'CurrentLiabilities', 0)
    
    # 如果 FCF 为 0，尝试从 OCF - CapEx 计算
    if fcf == 0 and ocf != 0:
        capex = abs(safe_get(latest, 'CapEx', 0))
        fcf = ocf - capex
    
    # ========================================
    # Buffett: 护城河与现金回报
    # linear_scale 阈值:
    #   ROE 稳定性 (μ/σ): bad=0.5, target=2.0, excellent=5.0
    #   FCF 均值 (相对资产%): bad=0, target=5, excellent=15
    #   毛利率波动 (σ%): bad=15, target=5, excellent=1 (reverse)
    # ========================================
    buffett_factors = {}
    
    # ROE 稳定性
    roe_stability_score = None
    if len(roe_series) >= 4:
        roe_mean = roe_series.mean()
        roe_std = roe_series.std()
        roe_stability = _safe_div(roe_mean, roe_std + 0.01)
        roe_stability_score = linear_scale(roe_stability, bad=0.5, target=2.0, excellent=5.0)
        buffett_factors["ROE 均值"] = f"{roe_mean:.1f}%"
        buffett_factors["ROE 稳定性 (μ/σ)"] = f"{roe_stability:.2f}"
    
    # FCF 均值 (占总资产百分比)
    fcf_score = None
    if len(fcf_ttm_series) >= 4 and total_assets > 0:
        fcf_mean = fcf_ttm_series.mean()
        fcf_pct = fcf_mean / total_assets * 100
        fcf_score = linear_scale(fcf_pct, bad=-2, target=5, excellent=15)
        buffett_factors["FCF 均值"] = f"{fcf_mean:,.1f}"
        buffett_factors["FCF/总资产"] = f"{fcf_pct:.1f}%"
    elif fcf != 0 and total_assets > 0:
        fcf_pct = fcf / total_assets * 100
        fcf_score = linear_scale(fcf_pct, bad=-2, target=5, excellent=15)
        buffett_factors["FCF (当期)"] = f"{fcf:,.1f}"
        buffett_factors["FCF/总资产"] = f"{fcf_pct:.1f}%"
    
    # 毛利率波动 (reverse: 越低越好)
    gm_vol_score = None
    if len(gm_series) >= 4:
        gm_std = gm_series.std()
        gm_vol_score = linear_scale(gm_std, bad=15, target=5, excellent=1, reverse=True)
        buffett_factors["毛利率波动 (σ)"] = f"{gm_std:.2f}%"
    
    score, status = _weighted_score([
        (roe_stability_score, 0.45, "ROE稳定性"),
        (fcf_score, 0.35, "FCF均值"),
        (gm_vol_score, 0.20, "毛利率波动"),
    ])
    buffett_factors.update(status)
    has_any = any(s is not None for s in [roe_stability_score, fcf_score, gm_vol_score])
    results["Buffett"] = {"score": score, "factors": buffett_factors, "available": has_any}
    
    # ========================================
    # Munger: 质量风控与反转
    # ROIC: bad=5%, target=15%, excellent=25%
    # D/E: bad=2.0, target=1.0, excellent=0.3 (reverse)
    # FCF转换率: bad=0.3, target=0.8, excellent=1.2
    # ========================================
    munger_factors = {}
    roic_val = safe_get(latest, 'ROIC', 0)
    de_ratio = _safe_div(total_debt, total_equity) if total_equity > 0 else None
    fcf_conv = _safe_div(fcf, ni) if ni > 0 else None
    
    roic_score = linear_scale(roic_val, bad=5, target=15, excellent=25) if roic_val != 0 else None
    de_score = linear_scale(de_ratio, bad=2.0, target=1.0, excellent=0.3, reverse=True) if de_ratio is not None else None
    fcf_conv_score = linear_scale(fcf_conv, bad=0.3, target=0.8, excellent=1.2) if fcf_conv is not None else None
    
    if roic_val != 0:
        munger_factors["ROIC"] = f"{roic_val:.1f}%"
    if de_ratio is not None:
        munger_factors["Debt/Equity"] = f"{de_ratio:.2f}x"
    if fcf_conv is not None:
        munger_factors["FCF 转换率"] = f"{fcf_conv:.2f}x"
    
    score, status = _weighted_score([
        (roic_score, 0.40, "ROIC"),
        (de_score, 0.30, "D/E"),
        (fcf_conv_score, 0.30, "FCF转换率"),
    ])
    munger_factors.update(status)
    has_any = any(s is not None for s in [roic_score, de_score, fcf_conv_score])
    results["Munger"] = {"score": score, "factors": munger_factors, "available": has_any}
    
    # ========================================
    # Lynch: 动态 GARP (PEG 导向)
    # PEG: bad=3.0, target=1.0, excellent=0.5 (reverse)
    # EPS趋势: bad=-0.2, target=0.1, excellent=0.5
    # ========================================
    lynch_factors = {}
    eps_yoy = safe_get(latest, 'EPS_TTM_YoY', None)
    
    peg_score = None
    eps_trend_score = None
    
    if pe_ttm > 0 and eps_yoy is not None and eps_yoy > 0:
        adj_peg = pe_ttm / (eps_yoy * 100 + 0.01)
        peg_score = linear_scale(adj_peg, bad=3.0, target=1.0, excellent=0.5, reverse=True)
        lynch_factors["PE (TTM)"] = f"{pe_ttm:.1f}x"
        lynch_factors["EPS YoY"] = f"{eps_yoy:.1%}"
        lynch_factors["Adjusted PEG"] = f"{adj_peg:.2f}"
    
    # EPS 趋势
    eps_ttm_series = df_single['EPS_TTM'].dropna() if 'EPS_TTM' in df_single.columns else pd.Series(dtype=float)
    if len(eps_ttm_series) >= 4:
        recent = eps_ttm_series.iloc[-4:]
        eps_trend = (recent.iloc[-1] - recent.iloc[0]) / (abs(recent.iloc[0]) + 0.01)
        eps_trend_score = linear_scale(eps_trend, bad=-0.2, target=0.1, excellent=0.5)
        lynch_factors["EPS 趋势"] = f"{eps_trend:+.1%}"
    
    score, status = _weighted_score([
        (peg_score, 0.60, "PEG"),
        (eps_trend_score, 0.40, "EPS趋势"),
    ])
    lynch_factors.update(status)
    has_any = any(s is not None for s in [peg_score, eps_trend_score])
    results["Lynch"] = {"score": score, "factors": lynch_factors, "available": has_any}
    
    # ========================================
    # Graham: 深度价值与安全边际
    # NCAV/市值: bad=0, target=0.5, excellent=1.0
    # P/B: bad=5.0, target=1.5, excellent=0.8 (reverse)
    # ========================================
    graham_factors = {}
    ncav_adj = current_assets - total_debt if current_assets > 0 else 0
    pb_ratio = _safe_div(market_cap, total_equity * 1e9) if total_equity > 0 and market_cap > 0 else None
    ncav_to_mc = _safe_div(ncav_adj * 1e9, market_cap) if market_cap > 0 else None
    
    ncav_score = None
    pb_score = None
    
    if ncav_to_mc is not None and market_cap > 0:
        ncav_score = linear_scale(ncav_to_mc, bad=0, target=0.5, excellent=1.0)
        graham_factors["NCAV (adj)"] = f"{ncav_adj:,.1f}"
        graham_factors["NCAV/市值"] = f"{ncav_to_mc:.2f}"
    
    if pb_ratio is not None:
        pb_score = linear_scale(pb_ratio, bad=5.0, target=1.5, excellent=0.8, reverse=True)
        graham_factors["P/B"] = f"{pb_ratio:.2f}x"
    
    score, status = _weighted_score([
        (ncav_score, 0.55, "NCAV/市值"),
        (pb_score, 0.45, "P/B"),
    ])
    graham_factors.update(status)
    has_any = any(s is not None for s in [ncav_score, pb_score])
    results["Graham"] = {"score": score, "factors": graham_factors, "available": has_any}
    
    # ========================================
    # Greenblatt: 神奇公式 (ROC + Earnings Yield)
    # ROC: bad=5, target=15, excellent=30
    # EY: bad=2, target=7, excellent=15
    # ========================================
    greenblatt_factors = {}
    roc = roic_val if roic_val != 0 else safe_get(latest, 'ROA', 0)
    earnings_yield = _safe_div(eps_val, current_price) * 100 if current_price > 0 else 0
    
    roc_score = linear_scale(roc, bad=5, target=15, excellent=30) if roc != 0 else None
    ey_score = linear_scale(earnings_yield, bad=2, target=7, excellent=15) if earnings_yield > 0 else None
    
    if roc != 0:
        greenblatt_factors["ROC (资本回报率)"] = f"{roc:.1f}%"
    if earnings_yield > 0:
        greenblatt_factors["Earnings Yield"] = f"{earnings_yield:.1f}%"
    
    score, status = _weighted_score([
        (roc_score, 0.50, "ROC"),
        (ey_score, 0.50, "EY"),
    ])
    greenblatt_factors.update(status)
    has_any = any(s is not None for s in [roc_score, ey_score])
    results["Greenblatt"] = {"score": score, "factors": greenblatt_factors, "available": has_any}
    
    # ========================================
    # Fisher: 极速成长与创新
    # 营收CAGR: bad=0.02, target=0.15, excellent=0.30
    # 研发效率: bad=0, target=2.0, excellent=5.0
    # ========================================
    fisher_factors = {}
    rev_yoy = safe_get(latest, 'TotalRevenue_TTM_YoY', None)
    
    # 营收 CAGR
    sales_cagr_score = None
    sales_cagr = 0
    if len(rev_ttm_series) >= 8:
        oldest = rev_ttm_series.iloc[0]
        newest = rev_ttm_series.iloc[-1]
        years = len(rev_ttm_series) / 4
        if oldest > 0 and newest > 0 and years > 0:
            sales_cagr = (newest / oldest) ** (1 / years) - 1
    elif rev_yoy is not None:
        sales_cagr = rev_yoy
    
    if sales_cagr != 0:
        sales_cagr_score = linear_scale(sales_cagr, bad=0.02, target=0.15, excellent=0.30)
        fisher_factors["营收 CAGR"] = f"{sales_cagr:.1%}"
    
    # R&D 效率（代理指标: 营收增长/营业费用增长）
    rd_score = None
    opex_ttm_series = df_single['OperatingExpenses_TTM'].dropna() if 'OperatingExpenses_TTM' in df_single.columns else pd.Series(dtype=float)
    if len(rev_ttm_series) >= 4 and len(opex_ttm_series) >= 4:
        rev_growth = rev_ttm_series.iloc[-1] - rev_ttm_series.iloc[0]
        opex_total = opex_ttm_series.sum()
        rd_efficiency = _safe_div(rev_growth, abs(opex_total) + 0.01) * 100
        if rd_efficiency != 0:
            rd_score = linear_scale(rd_efficiency, bad=0, target=2.0, excellent=5.0)
            fisher_factors["营收增长效率"] = f"{rd_efficiency:.2f}"
    
    score, status = _weighted_score([
        (sales_cagr_score, 0.55, "营收CAGR"),
        (rd_score, 0.45, "研发效率"),
    ])
    fisher_factors.update(status)
    has_any = any(s is not None for s in [sales_cagr_score, rd_score])
    results["Fisher"] = {"score": score, "factors": fisher_factors, "available": has_any}
    
    # ========================================
    # Templeton: 逆向估值与均值回归
    # PE相对行业: bad=2.0, target=1.0, excellent=0.5 (reverse)
    # 价格历史分位: bad=0.9, target=0.5, excellent=0.1 (reverse)
    # ========================================
    templeton_factors = {}
    industry_pe = bench.get('pe_ttm', 20)
    pe_rel = _safe_div(pe_ttm, industry_pe) if pe_ttm > 0 else None
    
    pe_rel_score = None
    price_pct_score = None
    
    if pe_rel is not None and pe_ttm > 0:
        pe_rel_score = linear_scale(pe_rel, bad=2.0, target=1.0, excellent=0.5, reverse=True)
        templeton_factors["PE (TTM)"] = f"{pe_ttm:.1f}x"
        templeton_factors["行业 PE 中位数"] = f"{industry_pe:.1f}x"
        templeton_factors["PE 相对行业"] = f"{pe_rel:.2f}x"
    
    # 价格历史分位
    if df_price is not None and not df_price.empty and len(df_price) > 20:
        prices = df_price['close'].dropna()
        if current_price > 0 and len(prices) > 0:
            price_percentile = (prices < current_price).mean()
            price_pct_score = linear_scale(price_percentile, bad=0.9, target=0.5, excellent=0.1, reverse=True)
            templeton_factors["价格历史分位"] = f"{price_percentile:.0%}"
    
    score, status = _weighted_score([
        (pe_rel_score, 0.50, "PE相对行业"),
        (price_pct_score, 0.50, "价格分位"),
    ])
    templeton_factors.update(status)
    has_any = any(s is not None for s in [pe_rel_score, price_pct_score])
    results["Templeton"] = {"score": score, "factors": templeton_factors, "available": has_any}
    
    # ========================================
    # Dalio: 宏观稳健与债务杠杆
    # FCF/Debt: bad=0.05, target=0.3, excellent=0.6
    # ND/EBITDA: bad=5.0, target=2.0, excellent=0.5 (reverse)
    # ========================================
    dalio_factors = {}
    fcf_to_debt = _safe_div(fcf, total_debt) if total_debt > 0 else None
    net_debt = total_debt - cash
    nd_ebitda = _safe_div(net_debt, ebitda) if ebitda > 0 else None
    
    fcf_debt_score = None
    nd_ebitda_score = None
    
    if fcf_to_debt is not None:
        # 如果无债务，FCF/Debt 设为优秀
        if total_debt <= 0:
            fcf_debt_score = 10.0
            dalio_factors["FCF/Debt"] = "无债务 ✅"
        else:
            fcf_debt_score = linear_scale(fcf_to_debt, bad=0.05, target=0.3, excellent=0.6)
            dalio_factors["FCF/Debt"] = f"{fcf_to_debt:.2f}x"
    
    if nd_ebitda is not None:
        nd_ebitda_score = linear_scale(nd_ebitda, bad=5.0, target=2.0, excellent=0.5, reverse=True)
        dalio_factors["Net Debt"] = f"{net_debt:,.1f}"
        dalio_factors["Net Debt/EBITDA"] = f"{nd_ebitda:.2f}x"
    
    score, status = _weighted_score([
        (fcf_debt_score, 0.55, "FCF/Debt"),
        (nd_ebitda_score, 0.45, "ND/EBITDA"),
    ])
    dalio_factors.update(status)
    has_any = any(s is not None for s in [fcf_debt_score, nd_ebitda_score])
    results["Dalio"] = {"score": score, "factors": dalio_factors, "available": has_any}
    
    # ========================================
    # Soros: 动量与反身性
    # v2.5.1 修正: FCF趋势 → 均线乖离率 (Price/MA200 - 1)
    # v2.5.2 修正: 新股长度保护 — len(prices) < 200 时返回 np.nan
    # 净动量(12M-1M): bad=-0.10, target=0.10, excellent=0.40
    # 均线乖离率: bad=-0.20, target=0.05, excellent=0.30
    # ========================================
    soros_factors = {}
    
    momentum_score = None
    ma_deviation_score = None
    
    if df_price is not None and not df_price.empty:
        prices = df_price['close'].dropna()
        
        # 价格动量 (12M - 1M)
        momentum_12m = 0
        momentum_1m = 0
        if len(prices) >= 252 and current_price > 0:
            momentum_12m = (current_price / prices.iloc[-252] - 1) if prices.iloc[-252] > 0 else 0
        elif len(prices) >= 60 and current_price > 0:
            momentum_12m = (current_price / prices.iloc[0] - 1) if prices.iloc[0] > 0 else 0
        
        if len(prices) >= 21 and current_price > 0:
            momentum_1m = (current_price / prices.iloc[-21] - 1) if prices.iloc[-21] > 0 else 0
        
        net_momentum = momentum_12m - momentum_1m
        
        if net_momentum != 0 or momentum_12m != 0:
            momentum_score = linear_scale(net_momentum, bad=-0.10, target=0.10, excellent=0.40)
            soros_factors["12M 动量"] = f"{momentum_12m:+.1%}"
            soros_factors["1M 动量"] = f"{momentum_1m:+.1%}"
            soros_factors["净动量 (12M-1M)"] = f"{net_momentum:+.1%}"
        
        # v2.5.2: 均线乖离率 — 新股长度保护
        # 🛡️ 如果交易日 < 200天 (新股)，MA200 为 NaN，强制走降级路径
        ma_deviation = _get_ma_deviation(prices, current_price)
        if ma_deviation is not None:
            ma_deviation_score = linear_scale(ma_deviation, bad=-0.20, target=0.05, excellent=0.30)
            if len(prices) >= 200:
                ma200 = prices.iloc[-200:].mean()
                soros_factors["MA200"] = f"{ma200:.2f}"
                soros_factors["均线乖离率"] = f"{ma_deviation:+.1%}"
            else:
                ma50 = prices.iloc[-50:].mean()
                soros_factors["MA50 (降级)"] = f"{ma50:.2f}"
                soros_factors["均线乖离率 (MA50)"] = f"{ma_deviation:+.1%}"
                soros_factors["⚠️ 新股"] = f"交易日仅 {len(prices)} 天，MA200 不可用"
    
    score, status = _weighted_score([
        (momentum_score, 0.55, "净动量"),
        (ma_deviation_score, 0.45, "均线乖离率"),
    ])
    soros_factors.update(status)
    has_any = any(s is not None for s in [momentum_score, ma_deviation_score])
    results["Soros"] = {"score": score, "factors": soros_factors, "available": has_any}
    
    return results


# ============================================================
# 4. 雷达图渲染
# ============================================================

def _render_radar_chart(scores: Dict):
    """绘制 9 维大师雷达图"""
    
    categories = []
    values = []
    
    master_order = ["Buffett", "Munger", "Lynch", "Graham", "Greenblatt", 
                    "Fisher", "Templeton", "Dalio", "Soros"]
    
    for key in master_order:
        if key in scores:
            defn = MASTER_DEFINITIONS[key]
            label = f"{defn['icon']} {defn['name_cn']}"
            categories.append(label)
            # 确保值为有效数值，NaN/None → 50（中性）
            val = scores[key]["score"]
            values.append(float(val) if val is not None and not np.isnan(val) else 50.0)
    
    if not values:
        st.warning("无法生成雷达图")
        return
    
    # 闭合雷达图
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    # 确定颜色
    avg_score = np.mean(values)
    if avg_score >= 70:
        fill_color = "rgba(46, 204, 113, 0.25)"
        line_color = "#2ECC71"
    elif avg_score >= 40:
        fill_color = "rgba(241, 196, 15, 0.25)"
        line_color = "#F1C40F"
    else:
        fill_color = "rgba(231, 76, 60, 0.25)"
        line_color = "#E74C3C"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        marker=dict(size=8, color=line_color),
        name="大师评分",
        hovertemplate="%{theta}<br>评分: %{r:.0f}<extra></extra>"
    ))
    
    # 添加基准线 (60分)
    benchmark_values = [60] * len(categories_closed)
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values,
        theta=categories_closed,
        line=dict(color="rgba(150, 150, 150, 0.5)", width=1, dash="dash"),
        name="基准线 (60)",
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(200, 200, 200, 0.3)",
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
            ),
            angularaxis=dict(
                gridcolor="rgba(200, 200, 200, 0.3)",
            ),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        showlegend=True,
        legend=dict(x=0, y=-0.15, orientation="h"),
        margin=dict(l=80, r=80, t=40, b=40),
        height=520,
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 5. 详细分析面板
# ============================================================

def _render_detail_panels(scores: Dict):
    """展示每位大师的详细因子分析"""
    
    master_order = ["Buffett", "Munger", "Lynch", "Graham", "Greenblatt", 
                    "Fisher", "Templeton", "Dalio", "Soros"]
    
    for key in master_order:
        if key not in scores:
            continue
        
        defn = MASTER_DEFINITIONS[key]
        data = scores[key]
        score = data["score"]
        factors = data["factors"]
        available = data["available"]
        
        # 颜色指示
        if score >= 70:
            badge = "🟢"
        elif score >= 40:
            badge = "🟡"
        else:
            badge = "🔴"
        
        with st.expander(
            f"{defn['icon']} **{defn['name_cn']}** ({key}) — {defn['philosophy']} | {badge} {score:.0f}/100",
            expanded=False
        ):
            if not available:
                st.caption("⚠️ 数据不足，显示默认中性分数 (50)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 公式
                st.markdown(f"**评分公式**: `{defn['formula']}`")
                
                # 因子详情
                st.markdown("**底层因子**:")
                for fname, fval in factors.items():
                    st.write(f"- **{fname}**: {fval}")
            
            with col2:
                # 分数仪表
                st.metric(
                    label="综合评分",
                    value=f"{score:.0f}",
                    delta=f"{'优秀' if score >= 70 else '一般' if score >= 40 else '偏弱'}"
                )


# ============================================================
# 6. 综合评分与投资风格匹配
# ============================================================

def _render_composite_score(scores: Dict):
    """综合评分及投资风格分析"""
    
    all_scores = [scores[k]["score"] for k in scores if scores[k]["available"]]
    available_count = sum(1 for k in scores if scores[k]["available"])
    
    if not all_scores:
        st.warning("数据不足，无法计算综合评分")
        return
    
    avg_score = np.mean(all_scores)
    
    # 风格分类
    value_masters = ["Buffett", "Munger", "Graham", "Greenblatt", "Templeton"]
    growth_masters = ["Lynch", "Fisher"]
    trend_masters = ["Soros"]
    defense_masters = ["Dalio"]
    
    value_avg = np.mean([scores[k]["score"] for k in value_masters if k in scores and scores[k]["available"]] or [50])
    growth_avg = np.mean([scores[k]["score"] for k in growth_masters if k in scores and scores[k]["available"]] or [50])
    trend_avg = np.mean([scores[k]["score"] for k in trend_masters if k in scores and scores[k]["available"]] or [50])
    defense_avg = np.mean([scores[k]["score"] for k in defense_masters if k in scores and scores[k]["available"]] or [50])
    
    # 综合显示
    st.markdown("### 📊 综合评分")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 总分", f"{avg_score:.0f}/100", 
                  delta=f"{'优' if avg_score >= 70 else '中' if avg_score >= 40 else '弱'}")
    with col2:
        st.metric("🏰 价值维度", f"{value_avg:.0f}",
                  help="Buffett + Munger + Graham + Greenblatt + Templeton")
    with col3:
        st.metric("🚀 成长维度", f"{growth_avg:.0f}",
                  help="Lynch + Fisher")
    with col4:
        st.metric("⚡ 趋势维度", f"{trend_avg:.0f}",
                  help="Soros")
    with col5:
        st.metric("🛡️ 防御维度", f"{defense_avg:.0f}",
                  help="Dalio")
    
    # 风格判断
    st.markdown("### 🧭 投资风格匹配")
    
    style_scores = {
        "价值投资 (Value)": value_avg,
        "成长投资 (Growth)": growth_avg,
        "趋势/动量 (Momentum)": trend_avg,
        "防御/宏观 (Macro)": defense_avg,
    }
    
    dominant_style = max(style_scores, key=style_scores.get)
    
    # 风格柱状图
    fig = go.Figure()
    
    style_names = list(style_scores.keys())
    style_vals = list(style_scores.values())
    style_colors = ["#2E86AB", "#F18F01", "#E63946", "#1B4965"]
    
    fig.add_trace(go.Bar(
        x=style_names,
        y=style_vals,
        marker_color=style_colors,
        text=[f"{v:.0f}" for v in style_vals],
        textposition="auto",
    ))
    
    fig.update_layout(
        yaxis=dict(range=[0, 100], title="评分"),
        height=300,
        margin=dict(l=40, r=40, t=20, b=60),
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 文字解读
    st.markdown(f"""
**主导风格: {dominant_style}** (得分 {style_scores[dominant_style]:.0f})

| 维度 | 评分 | 解读 |
|------|------|------|
| 🏰 价值 | {value_avg:.0f}/100 | {"适合长线价值投资者" if value_avg >= 60 else "当前估值或质量不足以吸引严格的价值投资者"} |
| 🚀 成长 | {growth_avg:.0f}/100 | {"适合关注增长的 GARP 投资者" if growth_avg >= 60 else "增长动力不足或估值缺乏 PEG 支撑"} |
| ⚡ 趋势 | {trend_avg:.0f}/100 | {"顺势而为，具有正向动量" if trend_avg >= 60 else "动量信号较弱或处于反转中"} |
| 🛡️ 防御 | {defense_avg:.0f}/100 | {"资产负债表稳健，抗风险能力强" if defense_avg >= 60 else "杠杆偏高或现金流偏弱"} |
""")
    
    degraded = sum(1 for k in scores if not scores[k]["available"])
    st.caption(f"✅ 有效因子: {available_count}/9 | ⚠️ 不可用: {degraded}/9 (动态权重归一化)")


# ============================================================
# 7. 入口函数
# ============================================================

def render_master_analysis_tab(ticker: str, df_raw: pd.DataFrame, 
                                unit_label: str, wacc: float, rf: float):
    """
    九大投资大师多维分析 — 顶级 Tab 入口
    v2.5.1: 修复量纲灾难 + 缺失值降级 + Soros 幻觉
    """
    st.subheader(f"🧠 九大投资大师多维分析: {ticker}")
    st.caption(
        "基于 Warren Buffett、Charlie Munger、Peter Lynch、Benjamin Graham、"
        "Joel Greenblatt、Philip Fisher、John Templeton、Ray Dalio、George Soros "
        "的投资哲学进行多维度因子打分和风格匹配。"
    )
    
    if df_raw.empty:
        st.warning("请先录入财务数据")
        return
    
    # 处理数据
    _, df_single = process_financial_data(df_raw)
    if df_single.empty:
        st.warning("财务数据处理后为空")
        return
    
    latest = df_single.iloc[-1]
    meta = get_company_meta(ticker)
    
    # 获取价格数据
    df_price = get_market_history(ticker)
    
    # 计算评分
    with st.spinner("正在计算九大师评分..."):
        scores = compute_master_scores(df_single, latest, meta, df_price)
    
    # 🛡️ 存入 session_state 供估值总结 Tab 读取（避免重复计算）
    st.session_state['master_scores'] = scores
    
    # 渲染布局
    st.divider()
    
    # 上部：雷达图 + 综合评分
    radar_col, score_col = st.columns([3, 2])
    
    with radar_col:
        st.markdown("### 🎯 大师雷达图")
        _render_radar_chart(scores)
    
    with score_col:
        _render_composite_score(scores)
    
    st.divider()
    
    # 下部：详细面板
    st.markdown("### 📋 大师详细评分")
    _render_detail_panels(scores)
    
    # 方法论说明
    st.divider()
    with st.expander("📖 方法论说明 (v2.5.1)", expanded=False):
        st.markdown("""
**评分体系基于以下 9 位投资大师的核心投资哲学:**

1. **Buffett (巴菲特)** — 护城河与现金回报：ROE 稳定性 + FCF 均值 - 毛利率波动
2. **Munger (芒格)** — 质量风控：ROIC - 杠杆率(↓) + FCF 转换率
3. **Lynch (林奇)** — GARP 动态估值：调整后 PEG(↓) + EPS 变化趋势
4. **Graham (格雷厄姆)** — 深度价值：NCAV/市值 - P/B(↓)
5. **Greenblatt (格林布拉特)** — 神奇公式：ROC + Earnings Yield (等权)
6. **Fisher (费雪)** — 极速成长：营收 CAGR + 研发效率
7. **Templeton (邓普顿)** — 逆向估值：PE 相对行业(↓) + 价格历史分位(↓)
8. **Dalio (达里奥)** — 宏观稳健：FCF/Debt - Net Debt/EBITDA(↓)
9. **Soros (索罗斯)** — 动量与反身性：价格动量 (12M-1M) + 均线乖离率 (Price/MA200)

**v2.5.1 计算方法 (修正版):**
- 每个因子使用 **截断线性插值 (Truncated Linear Scaling)** 映射到 **[0, 10]** 分
- 定义三段阈值: `bad` (→0分) / `target` (→5分) / `excellent` (→10分)
- 各因子按大师哲学权重加权求和
- **缺失因子动态降级**: 如某因子数据不可用，剩余因子权重等比例归一化放大
- 最终映射到 0-100 分: 分数 ≥70 (🟢), 40-70 (🟡), <40 (🔴)

**注意:** 部分因子（如 R&D 支出、分析师情绪）不在当前数据库中，使用了代理指标。
Soros 的均线乖离率 (Price/MA200) 作为反身性正反馈循环的代理信号。
        """)
