"""
九大投资大师分析模块 v2.5.1 — 单元测试
测试 linear_scale / _weighted_score / 各大师得分方向性
不依赖 Streamlit / DB / API
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from modules.valuation.master_analysis import (
    linear_scale,
    _weighted_score,
    _safe_div,
    _get_ma_deviation,
)


# ============================================================
# linear_scale 测试
# ============================================================

class TestLinearScale:
    """截断线性插值函数测试"""
    
    def test_at_bad_returns_zero(self):
        """值等于 bad 阈值时返回 0"""
        assert linear_scale(5.0, bad=5.0, target=15.0, excellent=25.0) == 0.0
    
    def test_below_bad_returns_zero(self):
        """值低于 bad 阈值时返回 0"""
        assert linear_scale(1.0, bad=5.0, target=15.0, excellent=25.0) == 0.0
    
    def test_at_target_returns_five(self):
        """值等于 target 时返回 5.0"""
        assert linear_scale(15.0, bad=5.0, target=15.0, excellent=25.0) == 5.0
    
    def test_at_excellent_returns_ten(self):
        """值等于 excellent 时返回 10.0"""
        assert linear_scale(25.0, bad=5.0, target=15.0, excellent=25.0) == 10.0
    
    def test_above_excellent_returns_ten(self):
        """值超过 excellent 时仍返回 10.0 (截断)"""
        assert linear_scale(100.0, bad=5.0, target=15.0, excellent=25.0) == 10.0
    
    def test_midpoint_between_bad_and_target(self):
        """值在 bad 和 target 中间时返回 ~2.5"""
        result = linear_scale(10.0, bad=5.0, target=15.0, excellent=25.0)
        assert abs(result - 2.5) < 0.01
    
    def test_midpoint_between_target_and_excellent(self):
        """值在 target 和 excellent 中间时返回 ~7.5"""
        result = linear_scale(20.0, bad=5.0, target=15.0, excellent=25.0)
        assert abs(result - 7.5) < 0.01
    
    def test_nan_returns_none(self):
        """NaN 输入返回 None (表示不可用)"""
        assert linear_scale(float('nan'), bad=5.0, target=15.0, excellent=25.0) is None
    
    def test_reverse_mode(self):
        """reverse=True: 值越低越好"""
        # D/E=0.3 (低杠杆) 在 reverse 模式下应得高分
        low_de = linear_scale(0.3, bad=2.0, target=1.0, excellent=0.3, reverse=True)
        # D/E=2.0 (高杠杆) 在 reverse 模式下应得低分
        high_de = linear_scale(2.0, bad=2.0, target=1.0, excellent=0.3, reverse=True)
        assert low_de == 10.0
        assert high_de == 0.0
    
    def test_reverse_midpoint(self):
        """reverse 模式下的中间值"""
        # D/E=1.0 (target) 在 reverse 模式下应得 5.0
        mid = linear_scale(1.0, bad=2.0, target=1.0, excellent=0.3, reverse=True)
        assert abs(mid - 5.0) < 0.01
    
    def test_monotonic_increase(self):
        """正常模式下，值越大分数越高 (单调递增)"""
        scores = [linear_scale(v, bad=0, target=10, excellent=20) for v in range(25)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]
    
    def test_monotonic_reverse(self):
        """reverse 模式下，值越小分数越高 (单调递减)"""
        scores = [linear_scale(v, bad=5.0, target=2.0, excellent=0.5, reverse=True) for v in [0.5, 1.0, 2.0, 3.0, 5.0]]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]


# ============================================================
# _weighted_score 测试
# ============================================================

class TestWeightedScore:
    """动态权重归一化函数测试"""
    
    def test_all_factors_available(self):
        """所有因子都有值时正常加权"""
        score, status = _weighted_score([
            (8.0, 0.5, "A"),
            (6.0, 0.5, "B"),
        ])
        # (8*0.5 + 6*0.5) = 7.0 → 7.0 * 10 = 70
        assert abs(score - 70.0) < 0.1
        assert "⚠️ 降级" not in status
    
    def test_one_factor_missing(self):
        """一个因子缺失时，剩余因子权重归一化"""
        score, status = _weighted_score([
            (8.0, 0.5, "A"),
            (None, 0.3, "B_missing"),
            (6.0, 0.2, "C"),
        ])
        # 剩余: A(0.5) + C(0.2), 归一化为 A(5/7) + C(2/7)
        expected = (8.0 * (0.5 / 0.7) + 6.0 * (0.2 / 0.7)) * 10
        assert abs(score - expected) < 0.5
        assert "⚠️ 降级" in status
    
    def test_all_factors_missing(self):
        """所有因子都缺失时返回中性分数 50"""
        score, status = _weighted_score([
            (None, 0.5, "A"),
            (None, 0.5, "B"),
        ])
        assert score == 50.0
    
    def test_score_range_0_100(self):
        """输出始终在 [0, 100] 范围"""
        # 最高分
        score_max, _ = _weighted_score([(10.0, 1.0, "A")])
        assert score_max == 100.0
        
        # 最低分
        score_min, _ = _weighted_score([(0.0, 1.0, "A")])
        assert score_min == 0.0
    
    def test_no_nan_in_output(self):
        """输出永远不含 NaN"""
        score, _ = _weighted_score([
            (None, 0.4, "A"),
            (5.0, 0.3, "B"),
            (None, 0.3, "C"),
        ])
        assert not np.isnan(score)


# ============================================================
# _safe_div 测试
# ============================================================

class TestSafeDiv:
    """安全除法测试"""
    
    def test_normal_division(self):
        assert _safe_div(10, 2) == 5.0
    
    def test_divide_by_zero(self):
        assert _safe_div(10, 0) == 0.0
    
    def test_nan_numerator(self):
        assert _safe_div(float('nan'), 2) == 0.0
    
    def test_nan_denominator(self):
        assert _safe_div(10, float('nan')) == 0.0
    
    def test_custom_default(self):
        assert _safe_div(10, 0, default=-1) == -1


# ============================================================
# 大师得分方向性测试
# ============================================================

class TestScoringDirectionality:
    """验证各大师得分的方向性正确"""
    
    def test_high_roe_stability_scores_higher(self):
        """高 ROE 稳定性 → 高 Buffett 分数"""
        high = linear_scale(5.0, bad=0.5, target=2.0, excellent=5.0)
        low = linear_scale(0.5, bad=0.5, target=2.0, excellent=5.0)
        assert high > low
    
    def test_low_de_scores_higher_for_munger(self):
        """低杠杆 → 高 Munger 分数（reverse 模式）"""
        low_leverage = linear_scale(0.5, bad=2.0, target=1.0, excellent=0.3, reverse=True)
        high_leverage = linear_scale(1.8, bad=2.0, target=1.0, excellent=0.3, reverse=True)
        assert low_leverage > high_leverage
    
    def test_low_peg_scores_higher_for_lynch(self):
        """低 PEG → 高 Lynch 分数（reverse 模式）"""
        low_peg = linear_scale(0.5, bad=3.0, target=1.0, excellent=0.5, reverse=True)
        high_peg = linear_scale(2.5, bad=3.0, target=1.0, excellent=0.5, reverse=True)
        assert low_peg > high_peg
    
    def test_high_roc_scores_higher_for_greenblatt(self):
        """高 ROC → 高 Greenblatt 分数"""
        high = linear_scale(25, bad=5, target=15, excellent=30)
        low = linear_scale(8, bad=5, target=15, excellent=30)
        assert high > low
    
    def test_positive_momentum_scores_higher_for_soros(self):
        """正动量 → 高 Soros 分数"""
        positive = linear_scale(0.20, bad=-0.10, target=0.10, excellent=0.40)
        negative = linear_scale(-0.05, bad=-0.10, target=0.10, excellent=0.40)
        assert positive > negative
    
    def test_low_pe_relative_scores_higher_for_templeton(self):
        """低 PE 相对行业 → 高 Templeton 分数 (逆向)"""
        low_rel = linear_scale(0.7, bad=2.0, target=1.0, excellent=0.5, reverse=True)
        high_rel = linear_scale(1.5, bad=2.0, target=1.0, excellent=0.5, reverse=True)
        assert low_rel > high_rel


# ============================================================
# _get_ma_deviation 测试 (v2.5.2 新股长度保护)
# ============================================================

class TestGetMADeviation:
    """均线乖离率函数的新股长度保护测试"""
    
    def test_new_stock_under_50_days_returns_none(self):
        """交易日 < 50 天（新股）→ 返回 None，触发 _weighted_score 降级"""
        prices = pd.Series([100.0] * 30)  # 仅 30 天
        result = _get_ma_deviation(prices, current_price=105.0)
        assert result is None
    
    def test_degraded_to_ma50(self):
        """50 <= 交易日 < 200 → 降级使用 MA50"""
        prices = pd.Series([100.0] * 100)  # 100 天
        result = _get_ma_deviation(prices, current_price=110.0)
        # MA50 = 100, deviation = 110/100 - 1 = 0.10
        assert result is not None
        assert abs(result - 0.10) < 0.01
    
    def test_normal_ma200(self):
        """交易日 >= 200 → 使用 MA200"""
        prices = pd.Series([100.0] * 250)  # 250 天
        result = _get_ma_deviation(prices, current_price=120.0)
        # MA200 = 100, deviation = 120/100 - 1 = 0.20
        assert result is not None
        assert abs(result - 0.20) < 0.01
    
    def test_zero_price_returns_none(self):
        """当前价格为 0 → 返回 None"""
        prices = pd.Series([100.0] * 250)
        result = _get_ma_deviation(prices, current_price=0)
        assert result is None
    
    def test_empty_series_returns_none(self):
        """空价格序列 → 返回 None"""
        prices = pd.Series(dtype=float)
        result = _get_ma_deviation(prices, current_price=100.0)
        assert result is None

