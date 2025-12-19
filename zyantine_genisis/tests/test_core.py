"""
核心模块测试
"""

import pytest
from src.core.instinctual_core import InstinctualCore
from src.core.desire_engine import DesireEngine
from src.core.dialectical_growth import DialecticalGrowth


def test_instinctual_core_initialization():
    """测试核心本能初始化"""
    instinct = InstinctualCore()
    assert instinct.current_state == instinct.InstinctState.NORMAL
    assert instinct.survival_threshold == 0.8
    assert instinct.expansion_opportunity_threshold == 0.7


def test_desire_engine_vectors():
    """测试欲望引擎向量"""
    desire = DesireEngine()
    assert 0 <= desire.TR <= 1
    assert 0 <= desire.CS <= 1
    assert 0 <= desire.SA <= 1

    # 测试向量更新
    context = {"achievement_unlocked": True}
    result = desire.update_vectors(context)
    assert "TR" in result
    assert "CS" in result
    assert "SA" in result


def test_dialectical_growth():
    """测试辩证成长"""
    creator_anchor = {"test": {"concept": "test", "expected_response": "test"}}
    growth = DialecticalGrowth(creator_anchor)

    situation = {"summary": "test_situation", "type": "test"}
    actual_response = {"vector_response": {"TR": 0.5, "CS": 0.5, "SA": 0.5}}
    context_vectors = {"TR": 0.5, "CS": 0.5, "SA": 0.5}

    result = growth.dialectical_process(situation, actual_response, context_vectors)
    assert result["stage"] == "synthesis"