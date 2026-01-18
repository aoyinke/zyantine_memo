# ============ 记忆价值评估器 ============
"""
智能记忆价值评估系统

优化目标：
1. 减少对 LLM API 的依赖，优先使用规则引擎
2. 多维度评估：信息密度、情感价值、时间相关性、用户关联度等
3. 支持增量学习，根据用户行为调整评估策略
4. 保守策略：宁可多存储，不轻易过滤
"""
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class MemoryValueDimension(Enum):
    """记忆价值维度"""
    INFORMATION_DENSITY = "information_density"  # 信息密度
    EMOTIONAL_VALUE = "emotional_value"          # 情感价值
    TEMPORAL_RELEVANCE = "temporal_relevance"    # 时间相关性
    USER_RELEVANCE = "user_relevance"            # 用户关联度
    ACTIONABLE_VALUE = "actionable_value"        # 可执行价值
    KNOWLEDGE_VALUE = "knowledge_value"          # 知识价值
    SOCIAL_VALUE = "social_value"                # 社交价值
    UNIQUENESS = "uniqueness"                    # 独特性


@dataclass
class MemoryEvaluationResult:
    """记忆评估结果"""
    overall_score: float                          # 综合分数 (0-10)
    dimension_scores: Dict[str, float]            # 各维度分数
    should_store: bool                            # 是否应该存储
    storage_priority: str                         # 存储优先级 (critical/high/medium/low)
    suggested_ttl_hours: Optional[int]            # 建议的TTL（小时）
    extracted_entities: List[str]                 # 提取的实体
    detected_intents: List[str]                   # 检测到的意图
    confidence: float                             # 评估置信度
    evaluation_method: str                        # 评估方法 (rule/llm/hybrid)
    reason: str                                   # 评估理由

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "should_store": self.should_store,
            "storage_priority": self.storage_priority,
            "suggested_ttl_hours": self.suggested_ttl_hours,
            "extracted_entities": self.extracted_entities,
            "detected_intents": self.detected_intents,
            "confidence": self.confidence,
            "evaluation_method": self.evaluation_method,
            "reason": self.reason
        }


class MemoryEvaluator:
    """
    智能记忆价值评估器
    
    采用多层评估策略：
    1. 快速规则引擎（毫秒级）
    2. 特征提取器（毫秒级）
    3. LLM 增强评估（可选，秒级）
    """

    def __init__(self, 
                 llm_client: Optional[Any] = None,
                 enable_llm_evaluation: bool = False,
                 conservative_mode: bool = True):
        """
        初始化评估器
        
        Args:
            llm_client: LLM客户端（可选）
            enable_llm_evaluation: 是否启用LLM评估
            conservative_mode: 保守模式（宁可多存储）
        """
        self.llm_client = llm_client
        self.enable_llm_evaluation = enable_llm_evaluation
        self.conservative_mode = conservative_mode
        
        # 用户档案关键词（高价值）
        self._user_profile_keywords = {
            "名字", "姓名", "叫", "是", "我是", "年龄", "岁", "生日", "出生",
            "职业", "工作", "做", "从事", "专业", "学", "毕业",
            "住", "在", "来自", "家", "地址", "城市",
            "喜欢", "爱", "讨厌", "不喜欢", "偏好", "习惯",
            "电话", "手机", "邮箱", "微信", "QQ", "联系",
            "身高", "体重", "血型", "星座",
            "家人", "父母", "孩子", "配偶", "朋友",
        }
        
        # 事件/任务关键词（高价值）
        self._event_keywords = {
            "会议", "约会", "预约", "安排", "计划", "打算",
            "明天", "后天", "下周", "下个月", "周末",
            "提醒", "记得", "别忘了", "要", "需要",
            "买", "购买", "订", "预订",
            "去", "来", "到", "见面",
            "完成", "做完", "交", "提交", "截止",
        }
        
        # 知识/观点关键词（中高价值）
        self._knowledge_keywords = {
            "是什么", "为什么", "怎么", "如何", "什么是",
            "原因", "方法", "步骤", "技巧", "经验",
            "认为", "觉得", "看法", "观点", "意见",
            "学到", "了解", "知道", "发现", "明白",
            "建议", "推荐", "应该", "可以", "最好",
        }
        
        # 情感关键词
        self._emotion_keywords = {
            "positive": {"开心", "高兴", "快乐", "满意", "感谢", "喜欢", "爱", "棒", "好", "赞", "优秀", "成功"},
            "negative": {"难过", "伤心", "生气", "愤怒", "失望", "讨厌", "烦", "累", "困", "焦虑", "担心", "害怕"},
            "neutral": {"还好", "一般", "普通", "正常"}
        }
        
        # 低价值模式（可能需要过滤）
        self._low_value_patterns = [
            r'^(好的?|嗯|哦|啊|呃|额|ok|OK|是的?|对的?|行|可以|没问题)$',
            r'^(你好|您好|hi|hello|hey)$',
            r'^(谢谢|感谢|thanks|thank you)$',
            r'^(再见|拜拜|bye|goodbye)$',
            r'^[\s\n]*$',  # 空白内容
        ]
        
        # 数字和日期模式
        self._date_patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?',
            r'\d{1,2}[-/月]\d{1,2}[日号]?',
            r'(今天|明天|后天|昨天|前天)',
            r'(这|下|上)(周|个月|年)',
            r'\d{1,2}[点时](\d{1,2}分?)?',
            r'(早上|上午|中午|下午|晚上|凌晨)',
        ]
        
        self._number_patterns = [
            r'\d+\.?\d*[元块钱美元]',  # 金额
            r'\d+\.?\d*%',              # 百分比
            r'\d{11}',                  # 手机号
            r'\d+[岁年]',               # 年龄
        ]
        
        # 维度权重配置
        self._dimension_weights = {
            MemoryValueDimension.INFORMATION_DENSITY: 0.20,
            MemoryValueDimension.EMOTIONAL_VALUE: 0.10,
            MemoryValueDimension.TEMPORAL_RELEVANCE: 0.15,
            MemoryValueDimension.USER_RELEVANCE: 0.25,
            MemoryValueDimension.ACTIONABLE_VALUE: 0.15,
            MemoryValueDimension.KNOWLEDGE_VALUE: 0.10,
            MemoryValueDimension.UNIQUENESS: 0.05,
        }
        
        # 记忆类型的基础分数调整
        self._type_base_scores = {
            "user_profile": 8.0,    # 用户档案最重要
            "knowledge": 7.0,       # 知识类次之
            "experience": 6.5,      # 经验类
            "strategy": 6.5,        # 策略类
            "emotion": 6.0,         # 情感类
            "conversation": 5.0,    # 对话类基础分
            "system_event": 4.0,    # 系统事件
            "temporal": 5.5,        # 时间相关
        }
        
        # 存储阈值（保守模式下降低阈值）
        self._storage_thresholds = {
            "critical": 8.0,
            "high": 6.0,
            "medium": 4.0,
            "low": 2.0,
        }
        
        # 历史评估统计（用于自适应）
        self._evaluation_stats = {
            "total_evaluations": 0,
            "stored_count": 0,
            "filtered_count": 0,
            "average_score": 5.0,
        }

    def evaluate(self, 
                 content: str, 
                 memory_type: str = "conversation",
                 metadata: Optional[Dict[str, Any]] = None,
                 context: Optional[Dict[str, Any]] = None) -> MemoryEvaluationResult:
        """
        评估记忆内容的价值
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            metadata: 元数据
            context: 上下文信息（历史对话、用户档案等）
            
        Returns:
            评估结果
        """
        metadata = metadata or {}
        context = context or {}
        
        # 1. 快速检查：是否是明显的低价值内容
        if self._is_obviously_low_value(content):
            return self._create_low_value_result(content, "内容过短或为简单应答")
        
        # 2. 提取特征
        features = self._extract_features(content, memory_type, metadata, context)
        
        # 3. 多维度评分
        dimension_scores = self._calculate_dimension_scores(content, features, memory_type, context)
        
        # 4. 计算综合分数
        overall_score = self._calculate_overall_score(dimension_scores, memory_type)
        
        # 5. 确定存储策略
        should_store, priority, ttl = self._determine_storage_strategy(
            overall_score, memory_type, features
        )
        
        # 6. 可选：LLM 增强评估（仅对边界情况）
        evaluation_method = "rule"
        if self.enable_llm_evaluation and self.llm_client:
            if 3.0 <= overall_score <= 5.0:  # 边界情况才调用 LLM
                llm_adjustment = self._llm_evaluate(content, memory_type, context)
                if llm_adjustment is not None:
                    overall_score = (overall_score + llm_adjustment) / 2
                    evaluation_method = "hybrid"
        
        # 7. 保守模式调整
        if self.conservative_mode:
            # 保守模式下，提高存储倾向
            overall_score = min(overall_score + 1.0, 10.0)
            if overall_score >= 3.0:
                should_store = True
        
        # 8. 生成评估理由
        reason = self._generate_reason(features, dimension_scores, overall_score)
        
        # 9. 更新统计
        self._update_stats(overall_score, should_store)
        
        return MemoryEvaluationResult(
            overall_score=round(overall_score, 2),
            dimension_scores={k.value: round(v, 2) for k, v in dimension_scores.items()},
            should_store=should_store,
            storage_priority=priority,
            suggested_ttl_hours=ttl,
            extracted_entities=features.get("entities", []),
            detected_intents=features.get("intents", []),
            confidence=features.get("confidence", 0.8),
            evaluation_method=evaluation_method,
            reason=reason
        )

    def _is_obviously_low_value(self, content: str) -> bool:
        """检查是否是明显的低价值内容"""
        content = content.strip()
        
        # 空内容
        if not content:
            return True
        
        # 过短内容（少于3个字符）
        if len(content) < 3:
            return True
        
        # 匹配低价值模式
        for pattern in self._low_value_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return True
        
        return False

    def _extract_features(self, 
                          content: str, 
                          memory_type: str,
                          metadata: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """提取内容特征"""
        features = {
            "length": len(content),
            "word_count": len(content.split()),
            "has_numbers": False,
            "has_dates": False,
            "has_names": False,
            "has_locations": False,
            "entities": [],
            "intents": [],
            "emotion_type": "neutral",
            "emotion_intensity": 0.0,
            "information_density": 0.0,
            "user_profile_signals": 0,
            "event_signals": 0,
            "knowledge_signals": 0,
            "question_count": 0,
            "confidence": 0.8,
        }
        
        content_lower = content.lower()
        
        # 检测数字和日期
        for pattern in self._number_patterns:
            matches = re.findall(pattern, content)
            if matches:
                features["has_numbers"] = True
                features["entities"].extend(matches[:3])
        
        for pattern in self._date_patterns:
            matches = re.findall(pattern, content)
            if matches:
                features["has_dates"] = True
                features["entities"].extend(matches[:3])
        
        # 检测用户档案信号
        for keyword in self._user_profile_keywords:
            if keyword in content_lower:
                features["user_profile_signals"] += 1
                if features["user_profile_signals"] <= 5:
                    features["intents"].append(f"user_info:{keyword}")
        
        # 检测事件信号
        for keyword in self._event_keywords:
            if keyword in content_lower:
                features["event_signals"] += 1
                if features["event_signals"] <= 5:
                    features["intents"].append(f"event:{keyword}")
        
        # 检测知识信号
        for keyword in self._knowledge_keywords:
            if keyword in content_lower:
                features["knowledge_signals"] += 1
                if features["knowledge_signals"] <= 3:
                    features["intents"].append(f"knowledge:{keyword}")
        
        # 检测情感
        positive_count = sum(1 for kw in self._emotion_keywords["positive"] if kw in content_lower)
        negative_count = sum(1 for kw in self._emotion_keywords["negative"] if kw in content_lower)
        
        if positive_count > negative_count:
            features["emotion_type"] = "positive"
            features["emotion_intensity"] = min(positive_count / 3.0, 1.0)
        elif negative_count > positive_count:
            features["emotion_type"] = "negative"
            features["emotion_intensity"] = min(negative_count / 3.0, 1.0)
        
        # 检测问题
        features["question_count"] = content.count("?") + content.count("？")
        
        # 计算信息密度
        unique_chars = len(set(content))
        features["information_density"] = min(unique_chars / max(len(content), 1) * 2, 1.0)
        
        # 去重实体
        features["entities"] = list(set(features["entities"]))[:10]
        features["intents"] = list(set(features["intents"]))[:10]
        
        return features

    def _calculate_dimension_scores(self,
                                    content: str,
                                    features: Dict[str, Any],
                                    memory_type: str,
                                    context: Dict[str, Any]) -> Dict[MemoryValueDimension, float]:
        """计算各维度分数"""
        scores = {}
        
        # 1. 信息密度
        density_score = 5.0
        if features["length"] > 50:
            density_score += 1.0
        if features["length"] > 200:
            density_score += 1.0
        if features["has_numbers"]:
            density_score += 1.0
        if features["has_dates"]:
            density_score += 1.0
        density_score += features["information_density"] * 2
        scores[MemoryValueDimension.INFORMATION_DENSITY] = min(density_score, 10.0)
        
        # 2. 情感价值
        emotion_score = 5.0
        if features["emotion_type"] != "neutral":
            emotion_score += features["emotion_intensity"] * 3
        if features["emotion_type"] == "negative":
            emotion_score += 1.0  # 负面情感可能更需要记住
        scores[MemoryValueDimension.EMOTIONAL_VALUE] = min(emotion_score, 10.0)
        
        # 3. 时间相关性
        temporal_score = 5.0
        if features["has_dates"]:
            temporal_score += 3.0
        if features["event_signals"] > 0:
            temporal_score += min(features["event_signals"], 3)
        scores[MemoryValueDimension.TEMPORAL_RELEVANCE] = min(temporal_score, 10.0)
        
        # 4. 用户关联度
        user_score = 5.0
        if features["user_profile_signals"] > 0:
            user_score += min(features["user_profile_signals"] * 1.5, 4.0)
        if memory_type == "user_profile":
            user_score += 2.0
        scores[MemoryValueDimension.USER_RELEVANCE] = min(user_score, 10.0)
        
        # 5. 可执行价值
        action_score = 5.0
        if features["event_signals"] > 0:
            action_score += min(features["event_signals"] * 1.5, 3.0)
        if features["question_count"] > 0:
            action_score += 1.0
        scores[MemoryValueDimension.ACTIONABLE_VALUE] = min(action_score, 10.0)
        
        # 6. 知识价值
        knowledge_score = 5.0
        if features["knowledge_signals"] > 0:
            knowledge_score += min(features["knowledge_signals"] * 2, 4.0)
        if memory_type == "knowledge":
            knowledge_score += 2.0
        scores[MemoryValueDimension.KNOWLEDGE_VALUE] = min(knowledge_score, 10.0)
        
        # 7. 独特性（基于内容长度和信息密度）
        uniqueness_score = 5.0 + features["information_density"] * 3
        if len(features["entities"]) > 2:
            uniqueness_score += 1.0
        scores[MemoryValueDimension.UNIQUENESS] = min(uniqueness_score, 10.0)
        
        return scores

    def _calculate_overall_score(self,
                                  dimension_scores: Dict[MemoryValueDimension, float],
                                  memory_type: str) -> float:
        """计算综合分数"""
        # 加权平均
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self._dimension_weights.get(dimension, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 5.0
        
        # 应用类型调整
        type_adjustment = self._type_base_scores.get(memory_type, 5.0) - 5.0
        adjusted_score = base_score + type_adjustment * 0.3
        
        return max(0.0, min(10.0, adjusted_score))

    def _determine_storage_strategy(self,
                                     score: float,
                                     memory_type: str,
                                     features: Dict[str, Any]) -> Tuple[bool, str, Optional[int]]:
        """确定存储策略"""
        # 用户档案类型始终存储
        if memory_type == "user_profile":
            return True, "critical", None
        
        # 包含重要实体的内容
        if features["user_profile_signals"] >= 2 or features["event_signals"] >= 2:
            return True, "high", None
        
        # 根据分数确定
        if score >= self._storage_thresholds["critical"]:
            return True, "critical", None
        elif score >= self._storage_thresholds["high"]:
            return True, "high", None
        elif score >= self._storage_thresholds["medium"]:
            return True, "medium", 168  # 7天
        elif score >= self._storage_thresholds["low"]:
            return True, "low", 24  # 1天
        else:
            # 保守模式下仍然存储，但设置短TTL
            if self.conservative_mode:
                return True, "low", 6  # 6小时
            return False, "none", None

    def _llm_evaluate(self, 
                      content: str, 
                      memory_type: str,
                      context: Dict[str, Any]) -> Optional[float]:
        """使用 LLM 进行增强评估（仅边界情况）"""
        if not self.llm_client:
            return None
        
        try:
            prompt = f"""请评估以下内容作为长期记忆的价值（0-10分）。

评估标准：
- 是否包含用户的个人信息、偏好、习惯
- 是否包含重要的事件、日期、任务
- 是否包含有价值的知识或经验
- 是否对未来的对话有参考价值

内容类型：{memory_type}
内容：{content[:500]}

请只返回一个数字（0-10），不要其他内容。"""

            response = self.llm_client.get_response(prompt)
            
            # 解析响应
            score_match = re.search(r'(\d+\.?\d*)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(10.0, score))
        except Exception as e:
            print(f"[记忆评估器] LLM评估失败: {e}")
        
        return None

    def _generate_reason(self,
                         features: Dict[str, Any],
                         dimension_scores: Dict[MemoryValueDimension, float],
                         overall_score: float) -> str:
        """生成评估理由"""
        reasons = []
        
        if features["user_profile_signals"] > 0:
            reasons.append(f"包含{features['user_profile_signals']}个用户信息信号")
        
        if features["event_signals"] > 0:
            reasons.append(f"包含{features['event_signals']}个事件信号")
        
        if features["knowledge_signals"] > 0:
            reasons.append(f"包含{features['knowledge_signals']}个知识信号")
        
        if features["has_dates"]:
            reasons.append("包含日期时间信息")
        
        if features["has_numbers"]:
            reasons.append("包含数字信息")
        
        if features["emotion_type"] != "neutral":
            reasons.append(f"检测到{features['emotion_type']}情感")
        
        # 找出最高分维度
        if dimension_scores:
            top_dimension = max(dimension_scores.items(), key=lambda x: x[1])
            reasons.append(f"主要价值维度: {top_dimension[0].value}")
        
        if not reasons:
            if overall_score >= 5:
                reasons.append("内容具有一般参考价值")
            else:
                reasons.append("内容信息量较少")
        
        return "; ".join(reasons)

    def _update_stats(self, score: float, stored: bool) -> None:
        """更新统计信息"""
        self._evaluation_stats["total_evaluations"] += 1
        if stored:
            self._evaluation_stats["stored_count"] += 1
        else:
            self._evaluation_stats["filtered_count"] += 1
        
        # 更新平均分（增量计算）
        n = self._evaluation_stats["total_evaluations"]
        old_avg = self._evaluation_stats["average_score"]
        self._evaluation_stats["average_score"] = old_avg + (score - old_avg) / n

    def _create_low_value_result(self, content: str, reason: str) -> MemoryEvaluationResult:
        """创建低价值结果"""
        return MemoryEvaluationResult(
            overall_score=1.0,
            dimension_scores={d.value: 1.0 for d in MemoryValueDimension},
            should_store=self.conservative_mode,  # 保守模式下仍存储
            storage_priority="low" if self.conservative_mode else "none",
            suggested_ttl_hours=1 if self.conservative_mode else None,
            extracted_entities=[],
            detected_intents=[],
            confidence=0.95,
            evaluation_method="rule",
            reason=reason
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取评估统计"""
        stats = self._evaluation_stats.copy()
        if stats["total_evaluations"] > 0:
            stats["storage_rate"] = stats["stored_count"] / stats["total_evaluations"]
            stats["filter_rate"] = stats["filtered_count"] / stats["total_evaluations"]
        return stats

    def adjust_thresholds(self, adjustment: float) -> None:
        """调整存储阈值（用于自适应）"""
        for key in self._storage_thresholds:
            self._storage_thresholds[key] = max(
                0.0, 
                min(10.0, self._storage_thresholds[key] + adjustment)
            )

    def set_conservative_mode(self, enabled: bool) -> None:
        """设置保守模式"""
        self.conservative_mode = enabled
