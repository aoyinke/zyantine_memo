"""
长度规整器 - 根据情境重要性动态调整回复长度
"""
from datetime import datetime
from typing import Dict, List, Tuple, Any


class LengthRegulator:
    """长度规整器：动态长度与优先级规整器"""

    def __init__(self):
        self.regulation_log = []

        # 长度规则
        self.length_rules = {
            "critical_moment": {"min": 1, "max": None, "recommended": 5},  # 无上限
            "important_moment": {"min": 3, "max": 15, "recommended": 8},
            "routine_moment": {"min": 1, "max": 7, "recommended": 3},
            "quick_response": {"min": 1, "max": 5, "recommended": 2}
        }

        # 优先级标记词
        self.priority_indicators = {
            "high": ["紧急", "立刻", "马上", "救命", "帮帮我", "怎么办", "重要"],
            "medium": ["请问", "建议", "想法", "看法", "你觉得"],
            "low": ["随便", "聊聊", "没事", "对了", "话说"]
        }

    def regulate(self, reply_draft: str, cognitive_snapshot: Dict) -> str:
        """
        根据情境重要性规整回复长度
        返回: 规整后的回复
        """
        # 判断当前时刻的重要性
        moment_importance = self._determine_importance(cognitive_snapshot)

        # 获取对应的长度规则
        rules = self.length_rules.get(moment_importance, self.length_rules["routine_moment"])

        # 分析回复草案
        sentences = self._split_into_sentences(reply_draft)

        if not sentences:
            return ""

        # 对句子进行优先级排序
        prioritized_sentences = self._prioritize_sentences(sentences, cognitive_snapshot)

        # 应用长度规则
        regulated_sentences = self._apply_length_rules(
            prioritized_sentences, rules, moment_importance
        )

        # 重新组合成流畅回复
        final_reply = self._reconstruct_reply(regulated_sentences, moment_importance)

        # 记录规整过程
        self._log_regulation(
            original_draft=reply_draft,
            final_reply=final_reply,
            moment_importance=moment_importance,
            original_sentence_count=len(sentences),
            final_sentence_count=len(regulated_sentences),
            rules_applied=rules
        )

        return final_reply

    def _determine_importance(self, snapshot: Dict) -> str:
        """判断当前时刻的重要性"""
        # 兼容两种结构：扁平结构（ContextParser直接返回）和嵌套结构（包含external_context）
        if "external_context" in snapshot:
            # 嵌套结构
            external_context = snapshot.get("external_context", {})
            internal_tags = snapshot.get("internal_state_tags", [])
            user_input = external_context.get("raw_input_preview", "")
            user_emotion = external_context.get("user_emotion", "neutral")
            interaction_type = external_context.get("interaction_type", "")
            urgency_level = external_context.get("urgency_level", "low")
        else:
            # 扁平结构（ContextParser直接返回）
            external_context = snapshot
            internal_tags = snapshot.get("internal_state_tags", [])
            user_input = snapshot.get("raw_input_preview", "")
            user_emotion = snapshot.get("user_emotion", "neutral")
            interaction_type = snapshot.get("interaction_type", "")
            urgency_level = snapshot.get("urgency_level", "low")

        # 检查危机干预信号
        if self._is_crisis_intervention(user_input, urgency_level):
            return "critical_moment"

        # 检查极端情绪
        if user_emotion in ["sad", "angry", "desperate", "negative"]:
            return "important_moment"

        # 检查深度支持需求
        if interaction_type == "seeking_support":
            return "important_moment"

        # 检查核心价值观
        if self._touches_core_values(user_input):
            return "important_moment"

        # 检查战略规划
        if self._is_strategic_planning(user_input):
            return "important_moment"

        # 检查内部状态
        internal_text = " ".join(internal_tags)
        if any(word in internal_text for word in ["疲惫", "转不动", "耐心有限"]):
            return "quick_response"

        # 默认常规时刻
        return "routine_moment"

    def _is_crisis_intervention(self, user_input: str, urgency_level: str) -> bool:
        """判断是否需要危机干预"""
        # 紧急关键词
        emergency_keywords = ["自杀", "自残", "想死", "活不下去", "崩溃"]
        user_input_lower = user_input.lower()

        if any(keyword in user_input_lower for keyword in emergency_keywords):
            return True

        # 紧急交互类型
        if urgency_level == "high":
            return True

        return False

    def _touches_core_values(self, user_input: str) -> bool:
        """判断是否触及核心价值观"""
        # 核心价值观相关话题
        core_value_topics = [
            "意义", "价值", "信仰", "原则", "道德", "伦理",
            "人生目标", "存在意义", "我是谁"
        ]

        user_input_lower = user_input.lower()

        return any(topic in user_input_lower for topic in core_value_topics)

    def _is_strategic_planning(self, user_input: str) -> bool:
        """判断是否为战略规划"""
        # 规划相关话题
        planning_keywords = [
            "未来计划", "发展规划", "职业规划", "人生规划",
            "长期目标", "五年计划", "战略"
        ]

        user_input_lower = user_input.lower()

        return any(keyword in user_input_lower for keyword in planning_keywords)

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 简单分割
        delimiters = ['。', '！', '？', '!', '?', '\n']

        for delimiter in delimiters:
            text = text.replace(delimiter, '。')

        sentences = [s.strip() for s in text.split('。') if s.strip()]

        return sentences

    def _prioritize_sentences(self, sentences: List[str], snapshot: Dict) -> List[Dict]:
        """对句子进行优先级排序"""
        prioritized = []

        for i, sentence in enumerate(sentences):
            priority_score = self._calculate_sentence_priority(sentence, snapshot, i)

            prioritized.append({
                "sentence": sentence,
                "original_position": i,
                "priority_score": priority_score,
                "priority_level": self._score_to_level(priority_score)
            })

        # 按优先级排序（降序）
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)

        return prioritized

    def _calculate_sentence_priority(self, sentence: str, snapshot: Dict, position: int) -> float:
        """计算句子优先级分数"""
        score = 0.0

        # 1. 位置权重（开头的句子通常更重要）
        position_weight = 1.0 - (position / max(len(sentence.split()), 1)) * 0.3
        score += position_weight * 0.2

        # 2. 长度权重（适中的长度可能更重要）
        sentence_length = len(sentence)
        if 15 <= sentence_length <= 40:
            score += 0.15
        elif sentence_length > 40:
            score += 0.1

        # 3. 关键词权重
        keyword_score = self._assess_keyword_importance(sentence, snapshot)
        score += keyword_score

        # 4. 情感权重
        emotion_score = self._assess_emotional_importance(sentence, snapshot)
        score += emotion_score

        # 5. 信息密度权重
        info_density = self._assess_information_density(sentence)
        score += info_density * 0.1

        return min(1.0, score)

    def _assess_keyword_importance(self, sentence: str, snapshot: Dict) -> float:
        """评估关键词重要性"""
        score = 0.0

        # 当前目标相关关键词
        current_goal = snapshot.get("initial_desire_focus", "")
        goal_keywords = []

        if "CS" in current_goal:
            goal_keywords = ["信任", "理解", "安心", "安全", "支持"]
        elif "TR" in current_goal:
            goal_keywords = ["解决", "方案", "建议", "方法", "帮助"]

        for keyword in goal_keywords:
            if keyword in sentence:
                score += 0.05

        # 紧急关键词
        urgent_keywords = self.priority_indicators["high"]
        for keyword in urgent_keywords:
            if keyword in sentence:
                score += 0.1

        # 问题关键词（问句通常需要回答）
        if "?" in sentence or "？" in sentence:
            score += 0.08

        return min(0.3, score)

    def _assess_emotional_importance(self, sentence: str, snapshot: Dict) -> float:
        """评估情感重要性"""
        score = 0.0

        # 情感支持关键词
        support_keywords = ["别担心", "没关系", "我理解", "我在这里", "支持你"]
        for keyword in support_keywords:
            if keyword in sentence:
                score += 0.06

        # 共情表达
        empathy_keywords = ["感受", "体会", "理解", "懂你", "明白"]
        for keyword in empathy_keywords:
            if keyword in sentence:
                score += 0.04

        # 检查用户情绪状态
        user_emotion = snapshot.get("external_context", {}).get("user_emotion", "neutral")
        if user_emotion in ["sad", "angry"]:
            # 在用户情绪负面时，情感支持句子更重要
            if any(keyword in sentence for keyword in support_keywords + empathy_keywords):
                score += 0.1

        return min(0.25, score)

    def _assess_information_density(self, sentence: str) -> float:
        """评估信息密度"""
        words = sentence.split()
        if not words:
            return 0.0

        # 信息密集词（名词、动词、形容词）
        info_dense_words = 0
        for word in words:
            if len(word) >= 2:  # 假设较长的词携带更多信息
                info_dense_words += 1

        density = info_dense_words / len(words)

        # 适中的密度最好（0.4-0.7）
        if 0.4 <= density <= 0.7:
            return 0.8
        elif density > 0.7:
            return 0.6  # 可能过于密集
        else:
            return 0.4  # 信息稀疏

    def _score_to_level(self, score: float) -> str:
        """分数转级别"""
        if score >= 0.7:
            return "最高"
        elif score >= 0.5:
            return "高"
        elif score >= 0.3:
            return "中"
        else:
            return "低"

    def _apply_length_rules(self, prioritized_sentences: List[Dict],
                            rules: Dict, moment_importance: str) -> List[str]:
        """应用长度规则"""
        max_sentences = rules.get("max")
        recommended = rules.get("recommended", 3)

        # 选择句子
        selected_sentences = []

        # 先选优先级最高的
        for sentence_info in prioritized_sentences:
            if max_sentences and len(selected_sentences) >= max_sentences:
                break

            selected_sentences.append(sentence_info["sentence"])

        # 检查7句特许规则（针对常规时刻）
        if moment_importance == "routine_moment" and len(selected_sentences) < len(prioritized_sentences):
            # 检查是否有未选但必要的句子
            necessary_sentences = self._identify_necessary_sentences(
                prioritized_sentences, selected_sentences
            )

            # 如果必要句子不超过7句，可以加入
            total_count = len(selected_sentences) + len(necessary_sentences)
            if total_count <= 7:
                selected_sentences.extend(necessary_sentences)

        # 确保至少有所需的最小句子数
        min_sentences = rules.get("min", 1)
        if len(selected_sentences) < min_sentences and prioritized_sentences:
            # 补充优先级较高的句子
            needed = min_sentences - len(selected_sentences)
            additional = [
                s["sentence"] for s in prioritized_sentences
                if s["sentence"] not in selected_sentences
            ][:needed]
            selected_sentences.extend(additional)

        return selected_sentences

    def _identify_necessary_sentences(self, prioritized_sentences: List[Dict],
                                      already_selected: List[str]) -> List[str]:
        """识别必要的句子（7句特许规则）"""
        necessary = []

        for sentence_info in prioritized_sentences:
            sentence = sentence_info["sentence"]

            if sentence in already_selected:
                continue

            # 检查是否会导致严重误解
            if self._would_cause_misunderstanding(sentence, already_selected):
                necessary.append(sentence)

            # 检查是否包含关键信息
            if self._contains_critical_info(sentence, already_selected):
                necessary.append(sentence)

            # 最多添加2个必要句子
            if len(necessary) >= 2:
                break

        return necessary

    def _would_cause_misunderstanding(self, sentence: str, selected: List[str]) -> bool:
        """判断省略该句子是否会导致严重误解"""
        # 检查是否包含否定、转折、条件等关键逻辑词
        critical_logic_words = ["但是", "不过", "然而", "否则", "除非", "如果", "因为"]

        if any(word in sentence for word in critical_logic_words):
            # 检查是否已有对应的逻辑上下文
            has_context = False
            for selected_sentence in selected:
                if any(word in selected_sentence for word in ["虽然", "尽管", "既然"]):
                    has_context = True
                    break

            if not has_context:
                return True

        return False

    def _contains_critical_info(self, sentence: str, selected: List[str]) -> bool:
        """判断是否包含关键信息"""
        # 关键信息指示词
        critical_info_indicators = [
            "重要", "关键", "必须", "一定", "务必",
            "只有", "唯一", "最", "特别", "极其"
        ]

        if any(indicator in sentence for indicator in critical_info_indicators):
            # 检查信息是否已在其他句子中表达
            for selected_sentence in selected:
                # 简单重叠检查
                words1 = set(sentence.split())
                words2 = set(selected_sentence.split())
                overlap = len(words1.intersection(words2))

                if overlap >= 3:  # 如果有3个以上相同词，可能信息已覆盖
                    return False

            return True

        return False

    def _reconstruct_reply(self, sentences: List[str], moment_importance: str) -> str:
        """重新构建回复"""
        if not sentences:
            return ""

        # 根据重要性调整连接方式
        if moment_importance == "critical_moment":
            # 危机时刻：更直接，可能用短句
            connector = " "
        elif moment_importance == "important_moment":
            # 重要时刻：自然连接
            connector = "。"
        else:
            # 常规时刻：正常连接
            connector = "。"

        # 构建回复
        reply = connector.join(sentences)

        # 确保以句号结束（除非是短回复）
        if not reply.endswith('。') and len(reply) > 10:
            reply += '。'

        return reply

    def _log_regulation(self, original_draft: str, final_reply: str,
                        moment_importance: str, original_sentence_count: int,
                        final_sentence_count: int, rules_applied: Dict):
        """记录规整过程"""
        self.regulation_log.append({
            "timestamp": datetime.now().isoformat(),
            "moment_importance": moment_importance,
            "original_draft_preview": original_draft[:100],
            "final_reply_preview": final_reply[:100],
            "original_sentence_count": original_sentence_count,
            "final_sentence_count": final_sentence_count,
            "reduction_percentage": round(
                (1 - final_sentence_count / max(original_sentence_count, 1)) * 100, 1
            ),
            "rules_applied": rules_applied,
            "regulation_effect": "compressed" if final_sentence_count < original_sentence_count else "maintained"
        })

    def get_regulation_statistics(self) -> Dict[str, Any]:
        """获取规整统计信息"""
        total_regulations = len(self.regulation_log)
        compressed_count = sum(1 for log in self.regulation_log if log.get("regulation_effect") == "compressed")

        return {
            "total_regulations": total_regulations,
            "compressed_responses": compressed_count,
            "maintained_responses": total_regulations - compressed_count,
            "compression_rate": (compressed_count / total_regulations * 100) if total_regulations > 0 else 0,
            "last_regulation": self.regulation_log[-1] if self.regulation_log else None
        }

    def clear_regulation_log(self):
        """清空规整日志"""
        self.regulation_log = []