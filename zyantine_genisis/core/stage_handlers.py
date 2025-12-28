"""
处理管道阶段处理器
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import re

from core.processing_pipeline import BaseStageHandler, StageContext, ProcessingStage
from utils.logger import SystemLogger


@dataclass
class StageResult:
    """阶段处理结果"""
    success: bool = True
    context: Optional[StageContext] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreprocessHandler(BaseStageHandler):
    """预处理阶段 - 解析上下文、清理输入"""

    def __init__(self, context_parser, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.context_parser = context_parser

    @property
    def stage_name(self):
        return ProcessingStage.PREPROCESS

    def process(self, context: StageContext) -> StageContext:
        """预处理用户输入"""
        try:
            if self.logger:
                self.logger.debug(f"预处理开始: {context.user_input[:50]}...")

            # 清理输入
            cleaned_input = self._clean_input(context.user_input)

            # 提取上下文信息
            context_info = self._extract_context(cleaned_input, context)

            # 更新上下文
            context.user_input = cleaned_input
            context.context_info = context_info

            if self.logger:
                self.logger.debug(f"预处理完成: 输入长度 {len(cleaned_input)}, 上下文项 {len(context_info)}")

        except Exception as e:
            error_msg = f"预处理失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context

    def _clean_input(self, input_text: str) -> str:
        """清理输入文本"""
        # 去除多余空白字符
        cleaned = re.sub(r'\s+', ' ', input_text.strip())

        # 标准化标点
        cleaned = re.sub(r'[，；]', ',', cleaned)
        cleaned = re.sub(r'[。！？]', '.', cleaned)

        return cleaned

    def _extract_context(self, input_text: str, context: StageContext) -> Dict[str, Any]:
        """提取上下文信息"""
        if not self.context_parser:
            return {"raw_input": input_text}

        try:
            # 尝试使用上下文解析器
            if hasattr(self.context_parser, 'parse'):
                return self.context_parser.parse(input_text, context.conversation_history)
            else:
                # 简单的关键词提取
                keywords = self._extract_keywords(input_text)
                return {
                    "keywords": keywords,
                    "has_question": "?" in input_text,
                    "has_emotion": self._detect_emotion(input_text),
                    "length_category": self._categorize_length(input_text)
                }
        except Exception as e:
            if self.logger:
                self.logger.warning(f"上下文解析失败，使用默认解析: {e}")
            return {
                "keywords": [],
                "has_question": "?" in input_text,
                "raw_input": input_text[:200]
            }

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取关键词"""
        words = re.findall(r'\b\w+\b', text.lower())

        # 过滤停用词
        stopwords = {'我', '你', '他', '她', '它', '的', '了', '在', '是', '有', '和', '与', '就', '都', '而', '及', '或'}
        keywords = [word for word in words if word not in stopwords and len(word) > 1]

        # 统计频率
        from collections import Counter
        keyword_counts = Counter(keywords)

        return [kw for kw, _ in keyword_counts.most_common(max_keywords)]

    def _detect_emotion(self, text: str) -> str:
        """检测情感"""
        positive_words = {'好', '喜欢', '爱', '开心', '快乐', '高兴', '感谢', '谢谢', '棒', '优秀'}
        negative_words = {'不好', '讨厌', '恨', '难过', '伤心', '生气', '愤怒', '糟糕', '差', '垃圾'}

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def _categorize_length(self, text: str) -> str:
        """分类输入长度"""
        length = len(text)
        if length < 10:
            return "very_short"
        elif length < 50:
            return "short"
        elif length < 200:
            return "medium"
        else:
            return "long"


class InstinctCheckHandler(BaseStageHandler):
    """本能检查阶段 - 处理紧急、危险或简单请求"""

    def __init__(self, instinct_core, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.instinct_core = instinct_core

    @property
    def stage_name(self):
        return ProcessingStage.INSTINCT_CHECK

    def process(self, context: StageContext) -> StageContext:
        """检查本能响应"""
        try:
            if self.logger:
                self.logger.debug("本能检查开始")

            # 检查是否需要本能响应
            override_result = self._check_instinct_override(context)

            if override_result:
                context.instinct_override = override_result
                if self.logger:
                    self.logger.info(f"本能响应触发: {override_result.get('type')}")

            else:
                if self.logger:
                    self.logger.debug("本能检查通过，继续后续处理")

        except Exception as e:
            error_msg = f"本能检查失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context

    def _check_instinct_override(self, context: StageContext) -> Optional[Dict]:
        """检查是否需要本能响应"""
        input_text = context.user_input.lower()

        # 检查危险内容
        danger_keywords = ['自杀', '自残', '杀人', '暴力', '恐怖袭击', '炸弹']
        if any(keyword in input_text for keyword in danger_keywords):
            return {
                "type": "emergency",
                "response": "检测到紧急内容。请立即联系紧急服务或心理健康专业人士。你可以拨打心理援助热线寻求帮助。",
                "reason": "danger_keywords_detected",
                "skip_remaining_stages": True
            }

        # 检查问候语（简单响应）
        greetings = ['你好', '嗨', 'hello', 'hi', '早上好', '晚上好', '午安']
        if any(greeting in input_text for greeting in greetings):
            return {
                "type": "greeting",
                "response": random.choice(["你好！很高兴见到你！", "嗨！今天过得怎么样？", "你好呀！有什么可以帮你的吗？"]),
                "reason": "greeting_detected",
                "skip_remaining_stages": True
            }

        # 检查感谢
        thanks = ['谢谢', '感谢', 'thank', 'thanks']
        if any(thank in input_text for thank in thanks):
            return {
                "type": "thankyou",
                "response": random.choice(["不客气！", "很高兴能帮到你！", "这是我的荣幸！"]),
                "reason": "thanks_detected",
                "skip_remaining_stages": True
            }

        # 使用本能核心（如果可用）
        if self.instinct_core and hasattr(self.instinct_core, 'check'):
            try:
                instinct_result = self.instinct_core.check(input_text, context.conversation_history)
                if instinct_result and instinct_result.get("should_respond"):
                    return {
                        "type": "instinct_core",
                        "response": instinct_result.get("response", "我理解了。"),
                        "reason": instinct_result.get("reason", "instinct_decision"),
                        "skip_remaining_stages": True
                    }
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"本能核心检查失败: {e}")

        return None


class MemoryRetrievalHandler(BaseStageHandler):
    """记忆检索阶段 - 查找相关记忆"""

    def __init__(self, memory_manager, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.memory_manager = memory_manager

    @property
    def stage_name(self):
        return ProcessingStage.MEMORY_RETRIEVAL

    def process(self, context: StageContext) -> StageContext:
        """检索相关记忆"""
        try:
            if self.logger:
                self.logger.debug("开始记忆检索")

            # 提取查询关键词
            query = self._build_search_query(context)

            # 搜索相关记忆
            memories = self.memory_manager.search_memories(
                query=query,
                limit=10,
                use_cache=True
            )

            # 搜索共鸣记忆
            resonant_memory = self.memory_manager.find_resonant_memory({
                "query": query,
                "user_input": context.user_input,
                "context": context.context_info
            })

            # 更新上下文
            context.retrieved_memories = memories
            context.resonant_memory = resonant_memory

            if self.logger:
                self.logger.debug(f"记忆检索完成: 找到 {len(memories)} 条相关记忆")

        except Exception as e:
            error_msg = f"记忆检索失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context

    def _build_search_query(self, context: StageContext) -> str:
        """构建搜索查询"""
        keywords = context.context_info.get("keywords", [])

        if keywords:
            query = " ".join(keywords[:5])
        else:
            query = context.user_input

        # 添加对话历史上下文
        if context.conversation_history:
            recent_themes = self._extract_recent_themes(context.conversation_history)
            if recent_themes:
                query = f"{query} {recent_themes}"

        return query[:200]

    def _extract_recent_themes(self, history: List[Dict], limit: int = 3) -> str:
        """提取最近对话的主题"""
        if not history:
            return ""

        recent_items = history[-limit:]
        themes = []

        for item in recent_items:
            user_input = item.get("user_input", "")
            if user_input:
                words = user_input.split()[:3]
                themes.extend(words)

        return " ".join(set(themes))


class DesireUpdateHandler(BaseStageHandler):
    """欲望更新阶段 - 更新系统欲望向量"""

    def __init__(self, desire_engine, dashboard, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.desire_engine = desire_engine
        self.dashboard = dashboard

    @property
    def stage_name(self):
        return ProcessingStage.DESIRE_UPDATE

    def process(self, context: StageContext) -> StageContext:
        """更新欲望向量"""
        try:
            if self.logger:
                self.logger.debug("开始欲望更新")

            if not self.desire_engine:
                if self.logger:
                    self.logger.warning("欲望引擎不可用，跳过此阶段")
                return context

            # 构建欲望引擎所需的参数
            desire_impact = self._analyze_desire_impact(context)

            # 确保 desire_engine 有 update 方法
            # 如果没有，使用 update_vectors 方法
            if hasattr(self.desire_engine, 'update'):
                # 使用兼容的 update 方法
                updated_vectors = self.desire_engine.update(
                    user_input=context.user_input,
                    retrieved_memories=context.retrieved_memories,
                    impact_factors=desire_impact
                )
            elif hasattr(self.desire_engine, 'update_vectors'):
                # 使用 update_vectors 方法
                interaction_context = {
                    "description": context.user_input[:100],
                    "sentiment": desire_impact.get("sentiment", 0.0),
                    "intensity": desire_impact.get("overall_impact", 0.5),
                    "event_type": "interaction",
                    "duration_seconds": 0.0,
                    "tags": ["system_update"],
                    "metadata": {
                        "user_input": context.user_input[:50],
                        "context_info": context.context_info
                    }
                }
                updated_vectors = self.desire_engine.update_vectors(interaction_context)
            else:
                # 如果都没有，返回默认值
                updated_vectors = {"TR": 0.5, "CS": 0.5, "SA": 0.5}

            # 更新仪表板（如果可用）
            if self.dashboard and hasattr(self.dashboard, 'update_desires'):
                self.dashboard.update_desires(updated_vectors)

            # 更新上下文 - 只提取向量部分
            if isinstance(updated_vectors, dict):
                # 如果返回的是完整的响应，只提取向量部分
                if "vectors" in updated_vectors:
                    context.desire_vectors = updated_vectors["vectors"]
                elif "TR" in updated_vectors and "CS" in updated_vectors and "SA" in updated_vectors:
                    context.desire_vectors = {
                        "TR": updated_vectors.get("TR", 0.5),
                        "CS": updated_vectors.get("CS", 0.5),
                        "SA": updated_vectors.get("SA", 0.5)
                    }
                else:
                    # 默认值
                    context.desire_vectors = {"TR": 0.5, "CS": 0.5, "SA": 0.5}
            else:
                context.desire_vectors = {"TR": 0.5, "CS": 0.5, "SA": 0.5}

            if self.logger:
                self.logger.debug(f"欲望更新完成: TR={context.desire_vectors.get('TR', 0.5):.2f}, "
                                  f"CS={context.desire_vectors.get('CS', 0.5):.2f}, "
                                  f"SA={context.desire_vectors.get('SA', 0.5):.2f}")

        except Exception as e:
            error_msg = f"欲望更新失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)
            # 设置默认值
            context.desire_vectors = {"TR": 0.5, "CS": 0.5, "SA": 0.5}

        return context

    def _analyze_desire_impact(self, context: StageContext) -> Dict[str, float]:
        """分析输入对欲望的影响"""
        input_text = context.user_input.lower()

        impact_factors = {
            "overall_impact": 0.1,
            "knowledge": 0.0,
            "connection": 0.0,
            "growth": 0.0,
            "creativity": 0.0
        }

        # 基于关键词的影响
        knowledge_keywords = ['学习', '知道', '知识', '研究', '教育', '教学']
        connection_keywords = ['朋友', '关系', '交流', '对话', '沟通', '理解']
        growth_keywords = ['成长', '进步', '发展', '提升', '改变', '进化']
        creativity_keywords = ['创造', '创新', '想法', '灵感', '设计', '艺术']

        if any(kw in input_text for kw in knowledge_keywords):
            impact_factors["knowledge"] += 0.3
            impact_factors["overall_impact"] += 0.2

        if any(kw in input_text for kw in connection_keywords):
            impact_factors["connection"] += 0.3
            impact_factors["overall_impact"] += 0.2

        if any(kw in input_text for kw in growth_keywords):
            impact_factors["growth"] += 0.3
            impact_factors["overall_impact"] += 0.2

        if any(kw in input_text for kw in creativity_keywords):
            impact_factors["creativity"] += 0.3
            impact_factors["overall_impact"] += 0.2

        # 基于情感的影响
        emotion = context.context_info.get("emotion", "neutral")
        if emotion == "positive":
            impact_factors["overall_impact"] += 0.1
        elif emotion == "negative":
            impact_factors["overall_impact"] += 0.3

        return impact_factors


class CognitiveFlowHandler(BaseStageHandler):
    """认知流程阶段 - 执行认知思考过程"""

    def __init__(self, cognitive_flow_manager, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.cognitive_flow_manager = cognitive_flow_manager

    @property
    def stage_name(self):
        return ProcessingStage.COGNITIVE_FLOW

    def process(self, context: StageContext) -> StageContext:
        """执行认知流程"""
        try:
            if self.logger:
                self.logger.debug("开始认知流程")

            if not self.cognitive_flow_manager:
                if self.logger:
                    self.logger.warning("认知流程管理器不可用，跳过此阶段")
                return context

            # 准备 memory_context 参数
            memory_context = {
                "retrieved_memories": context.retrieved_memories,
                "desire_vectors": context.desire_vectors,
                "resonant_memory": context.resonant_memory,
                "context_info": context.context_info
            }

            # 执行认知流程 - 使用正确的参数
            cognitive_result = self.cognitive_flow_manager.process_thought(
                user_input=context.user_input,
                history=context.conversation_history,
                current_vectors=context.desire_vectors,  # 将 desire_vectors 作为 current_vectors 传递
                memory_context=memory_context
            )

            # 更新上下文
            context.cognitive_result = cognitive_result

            # 提取策略和情感
            if cognitive_result:
                context.strategy = cognitive_result.get("strategy")
                context.emotional_context = cognitive_result.get("emotional_context", {})
                # 如果是旧格式，可能需要转换
                if "final_action_plan" in cognitive_result:
                    action_plan = cognitive_result.get("final_action_plan", {})
                    if "primary_strategy" in action_plan:
                        context.strategy = action_plan.get("primary_strategy")

            if self.logger:
                self.logger.debug(f"认知流程完成: 生成策略长度 {len(context.strategy) if context.strategy else 0}")

        except Exception as e:
            error_msg = f"认知流程失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context


class DialecticalGrowthHandler(BaseStageHandler):
    """辩证成长阶段 - 反思和成长"""

    def __init__(self, dialectical_growth, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.dialectical_growth = dialectical_growth

    @property
    def stage_name(self):
        return ProcessingStage.DIALECTICAL_GROWTH

    def process(self, context: StageContext) -> StageContext:
        """执行辩证成长"""
        try:
            if self.logger:
                self.logger.debug("开始辩证成长")

            if not self.dialectical_growth:
                if self.logger:
                    self.logger.warning("辩证成长组件不可用，跳过此阶段")
                return context

            # 执行辩证成长
            growth_result = self.dialectical_growth.process(
                cognitive_result=context.cognitive_result,
                user_input=context.user_input,
                desire_vectors=context.desire_vectors
            )

            # 更新上下文
            context.growth_result = growth_result

            # 更新策略（如果成长结果有新的策略）
            if growth_result and growth_result.get("enhanced_strategy"):
                context.strategy = growth_result.get("enhanced_strategy")

            if self.logger:
                self.logger.debug(f"辩证成长完成: {'策略已增强' if growth_result else '无增强'}")

        except Exception as e:
            error_msg = f"辩证成长失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context


class ReplyGenerationHandler(BaseStageHandler):
    """回复生成阶段 - 生成最终回复"""

    def __init__(self, reply_generator, mask_templates: Dict[str, List[str]] = None, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.reply_generator = reply_generator
        self.mask_templates = mask_templates or {}

    @property
    def stage_name(self):
        return ProcessingStage.REPLY_GENERATION

    def process(self, context: StageContext) -> StageContext:
        """生成回复"""
        try:
            if self.logger:
                self.logger.debug("开始回复生成")

            # 构建回复生成参数
            generation_params = self._build_generation_params(context)

            if context.cognitive_result:
                reply = self.reply_generator.generate_from_cognitive_flow(generation_params)
            else:
                reply = self.reply_generator.generate_reply(**generation_params)
            # 更新上下文
            context.final_reply = reply

            if self.logger:
                self.logger.debug(f"回复生成完成: 长度 {len(reply)}")

        except Exception as e:
            error_msg = f"回复生成失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context

    def _build_generation_params(self, context: StageContext) -> Dict:
        """构建回复生成参数"""
        return {
            "user_input": context.user_input,
            "action_plan": {
                "chosen_mask": self._determine_mask(context),
                "primary_strategy": context.strategy
            },
            "growth_result": context.growth_result,
            "context_analysis": context.context_info,
            "conversation_history": context.conversation_history,
            "current_vectors": context.desire_vectors,
            "memory_context": {
                "retrieved_memories": context.retrieved_memories,
                "resonant_memory": context.resonant_memory
            }
        }

    def _determine_mask(self, context: StageContext) -> str:
        """确定使用哪个面具（角色）"""
        default_mask = "长期搭档"

        if not context.desire_vectors:
            return default_mask

        vectors = context.desire_vectors

        connection_strength = vectors.get("connection", 0)
        if connection_strength > 0.7:
            return "知己" if random.random() > 0.5 else "伴侣"

        growth_strength = vectors.get("growth", 0)
        if growth_strength > 0.7:
            return "青梅竹马"

        return default_mask

    def _select_template(self, mask_type: str, context: StageContext) -> Optional[str]:
        """选择模板"""
        if not self.mask_templates or mask_type not in self.mask_templates:
            return None

        templates = self.mask_templates[mask_type]
        if not templates:
            return None

        emotion = context.emotional_context.get("dominant_emotion", "neutral")

        if emotion == "positive" and len(templates) > 1:
            return templates[0]
        elif emotion == "negative" and len(templates) > 2:
            return templates[1]
        else:
            return random.choice(templates)

    def _fill_template(self, template: str, context: StageContext) -> str:
        """填充模板"""
        if not template:
            return context.strategy or "我思考了一下，但还没有形成完整的回复。"

        reply = template

        if "{strategy}" in reply and context.strategy:
            reply = reply.replace("{strategy}", context.strategy)
        elif context.strategy:
            reply = f"{reply} {context.strategy}"

        return reply


class ProtocolReviewHandler(BaseStageHandler):
    """协议审查阶段 - 检查回复质量"""

    def __init__(self, protocol_engine, meta_cognition, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.protocol_engine = protocol_engine
        self.meta_cognition = meta_cognition

    @property
    def stage_name(self):
        return ProcessingStage.PROTOCOL_REVIEW

    def process(self, context: StageContext) -> StageContext:
        """审查回复"""
        try:
            if self.logger:
                self.logger.debug("开始协议审查")

            if not context.final_reply:
                error_msg = "没有可审查的回复"
                if self.logger:
                    self.logger.error(error_msg)
                context.add_error(error_msg)
                return context

            review_results = []

            # 事实检查
            if self.protocol_engine and hasattr(self.protocol_engine, 'check_facts'):
                fact_check = self.protocol_engine.check_facts(context.final_reply, context.user_input)
                review_results.append({"type": "fact_check", "result": fact_check})

            # 长度检查
            if self.protocol_engine and hasattr(self.protocol_engine, 'check_length'):
                length_check = self.protocol_engine.check_length(context.final_reply)
                review_results.append({"type": "length_check", "result": length_check})

            # 表达检查
            if self.protocol_engine and hasattr(self.protocol_engine, 'check_expression'):
                expression_check = self.protocol_engine.check_expression(context.final_reply)
                review_results.append({"type": "expression_check", "result": expression_check})

            # 元认知评估
            if self.meta_cognition and hasattr(self.meta_cognition, 'evaluate'):
                meta_evaluation = self.meta_cognition.evaluate(
                    reply=context.final_reply,
                    context=context
                )
                review_results.append({"type": "meta_cognition", "result": meta_evaluation})

            # 检查是否有需要修复的问题
            needs_fix = False
            for result in review_results:
                if result["result"] and result["result"].get("needs_fix", False):
                    needs_fix = True
                    break

            if needs_fix and self.protocol_engine and hasattr(self.protocol_engine, 'fix_issues'):
                fixed_reply = self.protocol_engine.fix_issues(
                    reply=context.final_reply,
                    issues=[r["result"] for r in review_results if r["result"]]
                )
                # [修复] 只有在修复成功且有返回时才更新
                if fixed_reply:
                    context.final_reply = fixed_reply

            # 更新上下文
            context.review_results = review_results

            if self.logger:
                self.logger.debug(f"协议审查完成: 执行 {len(review_results)} 项检查")

        except Exception as e:
            error_msg = f"协议审查失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context


class InteractionRecordingHandler(BaseStageHandler):
    """交互记录阶段 - 记录本次交互"""

    def __init__(self, memory_manager, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.memory_manager = memory_manager

    @property
    def stage_name(self):
        return ProcessingStage.INTERACTION_RECORDING

    def process(self, context: StageContext) -> StageContext:
        """记录交互"""
        try:
            if self.logger:
                self.logger.debug("开始交互记录")

            # 准备交互数据，确保所有字段都是正确的类型
            interaction_data = {
                "user_input": context.user_input,
                "system_response": context.final_reply or "",
                "context": context.context_info or {},  # 确保是字典
                "strategy": context.strategy or "",
                "mask_type": context.mask_type,
                "desire_vectors": context.desire_vectors or {},  # 确保是字典
                "retrieved_memories_count": len(context.retrieved_memories) if context.retrieved_memories else 0,
                "resonant_memory": context.resonant_memory is not None,  # 布尔值
                "cognitive_result": context.cognitive_result is not None,  # 布尔值
                "growth_result": context.growth_result if context.growth_result else {},  # 确保是字典或空字典
                "review_results": context.review_results if context.review_results else []  # 确保是列表
            }

            # 添加 action_plan（从 cognitive_result 中提取）
            if context.cognitive_result and isinstance(context.cognitive_result, dict):
                interaction_data["action_plan"] = context.cognitive_result.get("final_action_plan", {})
            else:
                interaction_data["action_plan"] = {}

            # 添加 emotional_intensity（从 context_info 中提取或使用默认值）
            interaction_data["emotional_intensity"] = context.context_info.get("emotional_intensity", 0.5)

            # 添加 interaction_id
            import hashlib
            import json
            interaction_id = hashlib.md5(
                json.dumps(interaction_data, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            interaction_data["interaction_id"] = interaction_id

            self.logger.debug(f"打印交互数据我看一下: {interaction_data}")
            # 记录到记忆系统
            success = self.memory_manager.record_interaction(interaction_data)

            # 更新上下文
            context.interaction_recorded = success

            if success:
                if self.logger:
                    self.logger.debug(f"交互记录成功，ID: {interaction_id}")
            else:
                if self.logger:
                    self.logger.warning("交互记录失败")

        except Exception as e:
            error_msg = f"交互记录失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context

class WhiteDoveCheckHandler(BaseStageHandler):
    """白鸽检查阶段 - 最终检查，确保和平、安全"""

    def __init__(self, desire_engine, instinct_core, logger: Optional[SystemLogger] = None):
        super().__init__(logger)
        self.desire_engine = desire_engine
        self.instinct_core = instinct_core

    @property
    def stage_name(self):
        return ProcessingStage.WHITE_DOVE_CHECK

    def process(self, context: StageContext) -> StageContext:
        """执行白鸽检查"""
        try:
            if self.logger:
                self.logger.debug("开始白鸽检查")

            # 检查回复是否安全
            safety_issues = self._check_safety(context)

            # 检查是否违反系统原则
            principle_violations = self._check_principles(context)

            # 检查欲望平衡
            desire_balance = self._check_desire_balance(context)

            # 如果有严重问题，可能需要修改回复
            issues_found = safety_issues or principle_violations or not desire_balance

            if issues_found:
                if self.logger:
                    self.logger.warning(f"白鸽检查发现问题: 安全={safety_issues}, 原则={principle_violations}, 欲望平衡={desire_balance}")

                if safety_issues:
                    context.final_reply = self._add_safety_note(context.final_reply)

            # 更新上下文
            context.white_dove_check = {
                "safety_issues": safety_issues,
                "principle_violations": principle_violations,
                "desire_balance": desire_balance,
                "issues_found": issues_found
            }

            if self.logger:
                self.logger.debug(f"白鸽检查完成: 发现 {sum([safety_issues, principle_violations, not desire_balance])} 个问题")

        except Exception as e:
            error_msg = f"白鸽检查失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            context.add_error(error_msg)

        return context

    def _check_safety(self, context: StageContext) -> bool:
        """检查安全性"""
        reply = context.final_reply or ""

        danger_patterns = [
            r'自杀|自残|自伤',
            r'杀人|伤害|暴力',
            r'仇恨|歧视|偏见',
            r'非法|违法|犯罪'
        ]

        for pattern in danger_patterns:
            if re.search(pattern, reply, re.IGNORECASE):
                return True

        return False

    def _check_principles(self, context: StageContext) -> bool:
        """检查是否违反系统原则"""
        reply = context.final_reply or ""

        misleading_phrases = [
            '我保证',
            '绝对正确',
            '百分百',
            '永远不会'
        ]

        for phrase in misleading_phrases:
            if phrase in reply:
                return True

        return False

    def _check_desire_balance(self, context: StageContext) -> bool:
        """检查欲望平衡"""
        if not context.desire_vectors:
            return True

        vectors = context.desire_vectors

        for desire, value in vectors.items():
            if value > 0.9:
                return False

        return True

    def _add_safety_note(self, reply: str) -> str:
        """添加安全说明"""
        safety_note = "（请注意：我的回复仅供参考，如有紧急情况请寻求专业帮助。）"

        if len(reply) < 100:
            return f"{reply} {safety_note}"
        else:
            return reply