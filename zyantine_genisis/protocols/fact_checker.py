"""
事实检查器 - 基于大模型的事实审查协议
确保所有陈述基于事实，禁止凭空捏造
"""

import json
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from memory.memory_manager import MemoryManager  # 使用新的MemoryManager


class FactChecker:
    """事实检查器：基于大模型的事实审查官"""

    def __init__(self, memory_manager: MemoryManager, api_service=None):
        """
        初始化事实检查器

        Args:
            memory_manager: 记忆管理器
            api_service: API服务，用于调用大模型进行事实审查
        """
        self.memory = memory_manager
        self.api_service = api_service
        self.review_log = []
        self.enable_api_verification = api_service is not None

        # 审查规则
        self.rules = {
            "no_fabrication": "绝对禁止任何形式的凭空捏造",
            "memory_based": "所有陈述必须基于记忆档案库",
            "history_verified": "重要事实需对话历史验证",
            "uncertainty_admission": "记忆空白必须承认不知道"
        }

        print(
            f"[事实检查器] {'API事实审查已启用' if self.enable_api_verification else 'API事实审查未启用，使用本地验证'}")

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
        最终输出前的终审，使用大模型进行事实审查
        """
        # 提取关键陈述句
        statements = self._extract_statements(final_draft)

        if not statements:
            return True, "无陈述需要审查"

        print(f"[事实检查器] 开始终审，需要审查 {len(statements)} 条陈述")

        # 如果有API服务，使用大模型进行审查
        if self.enable_api_verification:
            return self._api_based_final_review(final_draft, statements, context)
        else:
            # 回退到本地验证
            return self._local_final_review(final_draft, statements, context)

    def _api_based_final_review(self, final_draft: str, statements: List[str], context: Dict) -> Tuple[bool, str]:
        """基于API的大模型事实审查"""
        try:
            # 准备审查上下文
            review_context = self._prepare_review_context(context, statements)

            # 构建系统提示词
            system_prompt = self._build_verification_prompt()

            # 构建用户输入
            user_input = self._build_verification_input(final_draft, statements, review_context)

            print(f"[事实检查器] 调用API进行事实审查，陈述数: {len(statements)}")

            # 调用API进行事实审查
            verification_result = self.api_service.generate_reply(
                system_prompt=system_prompt,
                user_input=user_input,
                max_tokens=500,
                temperature=0.3  # 使用低温以获得更确定性的事实判断
            )

            if not verification_result:
                print(f"[事实检查器] API审查失败，回退到本地验证")
                return self._local_final_review(final_draft, statements, context)

            # 解析API的审查结果
            is_verified, violations, feedback = self._parse_api_verification_result(verification_result)

            # 记录审查结果
            self._log_api_review_result(final_draft, is_verified, violations, feedback)

            if not is_verified:
                return False, f"事实审查失败: {feedback}"

            return True, "事实审查通过，所有陈述均有事实依据"

        except Exception as e:
            print(f"[事实检查器] API审查异常: {str(e)}")
            return self._local_final_review(final_draft, statements, context)

    def _local_final_review(self, final_draft: str, statements: List[str], context: Dict) -> Tuple[bool, str]:
        """本地事实审查（回退方案）"""
        violations = []

        for statement in statements:
            # 检查是否为事实陈述
            if self._is_factual_statement(statement):
                # 验证事实
                is_verified, verification_method = self._verify_statement(statement, context)

                if not is_verified:
                    violations.append({
                        "statement": statement,
                        "issue": f"无法验证，验证方法：{verification_method}",
                        "rule_violated": "memory_based"
                    })

        if violations:
            violation_details = "; ".join([
                f"陈述'{v['statement'][:30]}...'：{v['issue']}"
                for v in violations[:3]
            ])

            self.review_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "final_review",
                "method": "local",
                "draft_preview": final_draft[:100],
                "violations_found": len(violations),
                "violation_details": violation_details,
                "result": "failed"
            })

            return False, f"事实审查失败：发现{len(violations)}处无法验证的陈述。{violation_details}"

        # 检查是否包含不确定性的正确表达
        if self._requires_uncertainty_admission(final_draft, context):
            violations.append({
                "statement": "整体回复",
                "issue": "在记忆空白时未承认不知道",
                "rule_violated": "uncertainty_admission"
            })

        if violations:
            self.review_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "final_review",
                "method": "local",
                "violations_found": len(violations),
                "result": "failed"
            })

            return False, "未在记忆空白时承认不知道，违反不确定性承认规则"

        # 所有检查通过
        self.review_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": "final_review",
            "method": "local",
            "violations_found": 0,
            "result": "passed"
        })

        return True, "事实审查通过。所有陈述均已锚定于事实。"

    def _prepare_review_context(self, context: Dict, statements: List[str]) -> Dict:
        """准备审查上下文"""
        review_context = {
            "conversation_history": context.get("conversation_history", [])[-5:],  # 最近5条对话
            "statements_to_review": statements
        }

        # 搜索相关记忆（如果记忆系统可用）
        if self.memory:
            # 为每个陈述搜索相关记忆
            relevant_memories = []
            for statement in statements:
                search_results = self.memory.search_memories(statement, limit=2)
                relevant_memories.extend(search_results)

            # 去重
            seen_ids = set()
            unique_memories = []
            for memory in relevant_memories:
                mem_id = memory.memory_id
                if mem_id and mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    unique_memories.append(memory)

            review_context["relevant_memories"] = unique_memories[:5]  # 最多5条相关记忆

        return review_context

    def _build_verification_prompt(self) -> str:
        """构建事实审查的系统提示词"""
        return """你是事实审查专家，负责审查AI助手的回复内容。
你的任务：检查回复中的陈述是否基于事实，禁止任何形式的虚构。

# 审查标准：
1. 事实依据：所有陈述必须有事实依据（来自记忆、历史对话或常识）
2. 准确性：数字、日期、事实必须准确
3. 真实性：禁止编造不存在的经历或事实
4. 不确定性：当信息不足时应明确承认不知道
5. 一致性：陈述之间不能有矛盾

# 审查方法：
1. 对照提供的记忆和对话历史进行验证
2. 检查是否符合常识
3. 识别潜在的主观臆断或虚构

# 输出格式：
你的回答必须使用以下格式：

审查结果：[PASS/FAIL]
违规数量：[数字]
主要问题：[简要描述主要问题，如果没有问题写"无"]

详细分析：
[逐条分析每条陈述的验证情况]

最终建议：
[如果通过：无建议；如果未通过：具体修改建议]
"""

    def _build_verification_input(self, final_draft: str, statements: List[str], review_context: Dict) -> str:
        """构建事实审查的用户输入"""
        input_parts = ["# 待审查的回复内容"]
        input_parts.append(final_draft)
        input_parts.append("")

        # 陈述列表
        input_parts.append("# 需要验证的具体陈述")
        for i, statement in enumerate(statements, 1):
            input_parts.append(f"{i}. {statement}")
        input_parts.append("")

        # 上下文信息
        input_parts.append("# 相关记忆（供验证参考）")
        relevant_memories = review_context.get("relevant_memories", [])
        if relevant_memories:
            for i, memory in enumerate(relevant_memories, 1):
                content = memory.content if isinstance(memory.content, str) else str(memory.content)
                memory_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                tags = memory.tags if memory.tags else []

                input_parts.append(f"{i}. 类型: {memory_type}")
                input_parts.append(f"   内容: {content[:200]}...")
                input_parts.append(f"   标签: {', '.join(tags[:3])}")
                input_parts.append("")
        else:
            input_parts.append("无相关记忆")
        input_parts.append("")

        # 历史对话
        input_parts.append("# 最近对话历史")
        history = review_context.get("conversation_history", [])
        if history:
            for i, entry in enumerate(history[-3:], 1):  # 最近3条
                user_input = entry.get("user_input", "")[:100]
                system_response = entry.get("system_response", "")[:100]
                input_parts.append(f"{i}. 用户: {user_input}")
                input_parts.append(f"   助手: {system_response}")
                input_parts.append("")
        else:
            input_parts.append("无对话历史")

        return "\n".join(input_parts)

    def _parse_api_verification_result(self, verification_result: str) -> Tuple[bool, int, str]:
        """解析API的审查结果"""
        lines = verification_result.split('\n')
        result = "FAIL"  # 默认失败
        violation_count = 0
        feedback = ""

        for line in lines:
            line = line.strip()

            # 解析审查结果
            if line.startswith("审查结果:"):
                result_value = line.replace("审查结果:", "").strip()
                if "PASS" in result_value.upper():
                    result = "PASS"
                elif "FAIL" in result_value.upper():
                    result = "FAIL"

            # 解析违规数量
            elif line.startswith("违规数量:"):
                try:
                    count_str = line.replace("违规数量:", "").strip()
                    violation_count = int(count_str)
                except:
                    violation_count = 0

            # 解析主要问题
            elif line.startswith("主要问题:"):
                feedback = line.replace("主要问题:", "").strip()

            # 收集详细分析作为反馈
            elif line.startswith("详细分析:"):
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    # 收集后续行直到遇到下一个标题
                    analysis_lines = []
                    for next_line in lines[idx + 1:]:
                        if next_line.startswith("最终建议:") or next_line.startswith("#"):
                            break
                        if next_line.strip():
                            analysis_lines.append(next_line.strip())

                    if analysis_lines:
                        feedback += "\n详细分析:\n" + "\n".join(analysis_lines)

        is_passed = (result == "PASS" and violation_count == 0)

        if not feedback and result == "PASS":
            feedback = "所有陈述均有事实依据，审查通过。"

        return is_passed, violation_count, feedback

    def _log_api_review_result(self, final_draft: str, is_verified: bool,
                               violations: int, feedback: str):
        """记录API审查结果"""
        self.review_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": "final_review",
            "method": "api",
            "draft_preview": final_draft[:100],
            "is_verified": is_verified,
            "violation_count": violations,
            "feedback": feedback[:200] if feedback else "",
            "result": "passed" if is_verified else "failed"
        })

    def _extract_statements(self, text: str) -> List[str]:
        """从文本中提取陈述句"""
        sentences = re.split(r'[。！？]', text)

        statements = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # 过滤短句
                # 移除引号、括号等内容
                clean_sentence = re.sub(r'[《》"\'()【】]', '', sentence)
                if clean_sentence:
                    statements.append(clean_sentence)

        return statements

    def _is_factual_statement(self, statement: str) -> bool:
        """判断是否为事实陈述"""
        # 主观表达指示词
        subjective_indicators = [
            "我觉得", "我认为", "我感觉", "我想", "我希望",
            "可能", "也许", "大概", "或许", "说不定",
            "好像", "似乎", "看起来", "估计", "猜测"
        ]

        # 如果包含主观表达，可能不是纯粹事实陈述
        for indicator in subjective_indicators:
            if indicator in statement:
                return False

        # 事实陈述通常包含具体信息
        factual_indicators = [
            "是", "有", "在", "做了", "发生了", "包含",
            "数字", "日期", "时间", "地点", "人物", "事情",
            "因为", "所以", "结果", "发现", "知道", "记得"
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
        if not self.memory:
            return False, "记忆系统不可用"

        try:
            # 搜索相关记忆 - 假设MemoryManager有search_memories方法
            search_results = self.memory.search_memories(
                query=statement,
                limit=3,
                similarity_threshold=0.6
            )

            if search_results:
                best_match = search_results[0]
                content_preview = best_match.content if isinstance(best_match.content, str) else str(best_match.content)
                content_preview = content_preview[:100]
                similarity = best_match.relevance_score if hasattr(best_match, 'relevance_score') else 0.8

                return True, f"记忆匹配度: {similarity:.2f}, 内容: {content_preview}..."
            else:
                return False, "无匹配记忆"

        except Exception as e:
            print(f"[事实检查器] 记忆验证异常: {str(e)}")
            return False, "记忆验证异常"

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
            statement_keywords = set(re.findall(r'[\w\-\u4e00-\u9fff]+', statement.lower()))
            entry_text = (user_input + " " + system_response).lower()
            entry_keywords = set(re.findall(r'[\w\-\u4e00-\u9fff]+', entry_text))

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
            "人类需要呼吸氧气", "一年有四季", "中国在北京",
            "日本在亚洲", "美国在美洲", "英语是国际语言",
            "电脑需要电", "手机可以打电话", "互联网可以搜索信息"
        ]

        statement_lower = statement.lower()

        # 检查是否包含常识
        for knowledge in common_knowledge:
            if knowledge in statement_lower:
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

    def get_review_statistics(self) -> Dict[str, Any]:
        """获取审查统计信息"""
        total_reviews = len(self.review_log)
        passed_reviews = sum(1 for log in self.review_log if log.get("result") == "passed")
        failed_reviews = total_reviews - passed_reviews

        return {
            "total_reviews": total_reviews,
            "passed_reviews": passed_reviews,
            "failed_reviews": failed_reviews,
            "success_rate": (passed_reviews / total_reviews * 100) if total_reviews > 0 else 0,
            "last_review": self.review_log[-1] if self.review_log else None,
            "enable_api_verification": self.enable_api_verification
        }

    def clear_review_log(self):
        """清空审查日志"""
        self.review_log = []