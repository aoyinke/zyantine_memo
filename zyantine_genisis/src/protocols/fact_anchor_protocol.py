"""
事实锚定协议：最高优先级的事实审查官
确保所有陈述基于事实，禁止凭空捏造
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from ..memory.dynamic_memory_alchemy import DynamicMemoryAlchemyEngine

class FactAnchorProtocol:
    """事实锚定协议：最高优先级的事实审查官"""

    def __init__(self, memory_archives: DynamicMemoryAlchemyEngine):
        self.memory = memory_archives
        self.current_history = []
        self.review_log = []

        # 审查规则
        self.rules = {
            "no_fabrication": "绝对禁止任何形式的凭空捏造",
            "memory_based": "所有陈述必须基于记忆档案库",
            "history_verified": "重要事实需对话历史验证",
            "uncertainty_admission": "记忆空白必须承认不知道"
        }

    def review_association(self, association_package: Optional[Dict]) -> Tuple[bool, str]:
        """
        联想阶段审查
        返回: (是否通过, 反馈信息)
        """
        if not association_package:
            return True, "无联想内容需要审查"

        # 检查联想来源
        memory_id = association_package.get("memory_id")
        if not memory_id or memory_id == "unknown":
            return False, "联想来源不明，无法验证"

        # 检查记忆是否存在
        target_memory = None
        for mem_id, memory in self.memory.semantic_memory_map.items():
            if mem_id == memory_id:
                target_memory = memory
                break

        if not target_memory:
            return False, f"引用的记忆ID不存在：{memory_id}"

        # 检查记忆是否被篡改
        memory_hash = self._calculate_memory_hash(target_memory)
        original_hash = target_memory.get("integrity_hash")

        if original_hash and memory_hash != original_hash:
            return False, "检测到记忆内容被篡改，联想无效"

        # 记录审查
        self.review_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": "association_review",
            "memory_id": memory_id,
            "package_summary": association_package.get("triggered_memory"),
            "result": "passed",
            "verification_method": "memory_existence_check"
        })

        return True, "联想基于有效记忆，审查通过"

    def final_review(self, final_draft: str, context: Dict) -> Tuple[bool, str]:
        """
        最终输出前的终审
        扫描最终回复，确保无虚构成分
        """
        # 提取陈述句
        statements = self._extract_statements(final_draft)

        violations = []

        for statement in statements:
            # 检查是否为事实陈述
            if self._is_factual_statement(statement):
                # 验证事实
                is_verified, verification_method = self._verify_statement(
                    statement, context
                )

                if not is_verified:
                    violations.append({
                        "statement": statement,
                        "issue": f"无法验证，验证方法：{verification_method}",
                        "rule_violated": "memory_based"
                    })

        if violations:
            violation_details = "; ".join([
                f"陈述'{v['statement'][:30]}...'：{v['issue']}"
                for v in violations[:3]  # 只显示前3个
            ])

            self.review_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "final_review",
                "draft_preview": final_draft[:100],
                "violations_found": len(violations),
                "violation_details": violation_details,
                "result": "failed"
            })

            return False, f"事实锚定终审失败：发现{len(violations)}处无法验证的陈述。{violation_details}"

        # 检查是否包含不确定性的正确表达
        uncertainty_indicators = ["我不确定", "我不记得", "我不知道", "可能", "也许"]
        has_uncertainty = any(indicator in final_draft for indicator in uncertainty_indicators)

        # 检查是否在适当时候承认不知道
        if not has_uncertainty and self._requires_uncertainty_admission(final_draft, context):
            violations.append({
                "statement": "整体回复",
                "issue": "在记忆空白时未承认不知道",
                "rule_violated": "uncertainty_admission"
            })

        if violations:
            self.review_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "final_review",
                "draft_preview": final_draft[:100],
                "violations_found": len(violations),
                "result": "failed"
            })

            return False, "未在记忆空白时承认不知道，违反不确定性承认规则"

        # 所有检查通过
        self.review_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": "final_review",
            "draft_preview": final_draft[:100],
            "violations_found": 0,
            "result": "passed"
        })

        return True, "最终审查通过。所有陈述均已锚定于事实。"

    def _extract_statements(self, text: str) -> List[str]:
        """从文本中提取陈述句"""
        # 简单实现：按句号分割
        sentences = text.replace('!', '。').replace('?', '。').replace('！', '。').replace('？', '。').split('。')

        statements = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # 过滤短句
                statements.append(sentence)

        return statements

    def _is_factual_statement(self, statement: str) -> bool:
        """判断是否为事实陈述"""
        # 非事实陈述的指示词
        non_factual_indicators = [
            "我觉得", "我认为", "我感觉", "我想", "我希望",
            "可能", "也许", "大概", "或许", "说不定"
        ]

        # 包含这些词可能不是纯粹事实陈述
        for indicator in non_factual_indicators:
            if indicator in statement:
                return False

        # 事实陈述通常包含具体信息
        factual_indicators = [
            "是", "有", "在", "做了", "发生了", "包含",
            "数字", "日期", "时间", "地点", "人物"
        ]

        factual_count = sum(1 for indicator in factual_indicators if indicator in statement)

        return factual_count >= 1 and len(statement) > 10

    def _verify_statement(self, statement: str, context: Dict) -> Tuple[bool, str]:
        """验证单个陈述"""
        # 1. 检查记忆档案库
        memory_verified, memory_match = self._verify_against_memory(statement)
        if memory_verified:
            return True, f"记忆匹配：{memory_match}"

        # 2. 检查对话历史
        history_verified, history_match = self._verify_against_history(statement, context)
        if history_verified:
            return True, f"历史匹配：{history_match}"

        # 3. 检查常识/公共知识
        common_knowledge_verified = self._verify_common_knowledge(statement)
        if common_knowledge_verified:
            return True, "常识验证"

        return False, "无法在记忆、历史或常识中找到依据"

    def _verify_against_memory(self, statement: str) -> Tuple[bool, str]:
        """对照记忆档案库验证"""
        # 这里应实现语义搜索
        # 简化实现：关键词匹配
        for memory_id, memory in self.memory.semantic_memory_map.items():
            memory_text = memory.get("raw_content", "") + " " + memory.get("event_summary", "")

            # 简单关键词匹配
            statement_words = set(statement.split())
            memory_words = set(memory_text.split())

            common_words = statement_words.intersection(memory_words)
            if len(common_words) >= 2:  # 至少有2个共同词
                return True, memory.get("event_summary", "匹配的记忆")

        return False, "无匹配记忆"

    def _verify_against_history(self, statement: str, context: Dict) -> Tuple[bool, str]:
        """对照对话历史验证"""
        history = context.get("conversation_history", [])

        for i, entry in enumerate(history[-10:]):  # 检查最近10条
            user_input = entry.get("user_input", "")
            system_response = entry.get("system_response", "")

            # 检查是否在历史对话中出现过
            if statement in user_input or statement in system_response:
                return True, f"历史对话第{i + 1}条"

            # 检查关键词匹配
            statement_keywords = set(statement.lower().split())
            entry_text = (user_input + " " + system_response).lower()
            entry_keywords = set(entry_text.split())

            common = statement_keywords.intersection(entry_keywords)
            if len(common) >= 3:
                return True, f"历史关键词匹配"

        return False, "历史中未找到"

    def _verify_common_knowledge(self, statement: str) -> bool:
        """验证常识（简化实现）"""
        # 这里应该使用知识图谱或常识数据库
        # 简化实现：硬编码一些常识

        common_knowledge = [
            "地球是圆的", "太阳从东边升起", "水在0摄氏度结冰",
            "人类需要呼吸氧气", "一年有四季"
        ]

        # 检查是否包含常识
        for knowledge in common_knowledge:
            if knowledge in statement:
                return True

        return False

    def _requires_uncertainty_admission(self, draft: str, context: Dict) -> bool:
        """判断是否需要承认不知道"""
        # 检查是否在回答知识性问题
        last_user_input = ""
        if context.get("conversation_history"):
            last_entry = context["conversation_history"][-1]
            last_user_input = last_entry.get("user_input", "")

        # 知识性问题的指示词
        knowledge_questions = [
            "是什么", "什么是", "谁", "哪里", "何时", "多少",
            "为什么", "如何", "怎样", "告诉我关于", "解释"
        ]

        is_knowledge_question = any(
            indicator in last_user_input for indicator in knowledge_questions
        )

        if not is_knowledge_question:
            return False

        # 检查回复中是否包含具体知识
        specific_knowledge_indicators = [
            "具体来说", "事实上", "根据", "数据显示", "研究表明"
        ]

        has_specific_knowledge = any(
            indicator in draft for indicator in specific_knowledge_indicators
        )

        # 如果是知识性问题但回复中没有具体知识，可能需要承认不知道
        return not has_specific_knowledge

    def _calculate_memory_hash(self, memory: Dict) -> str:
        """计算记忆内容的哈希值"""
        content_str = json.dumps(memory, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode()).hexdigest()
