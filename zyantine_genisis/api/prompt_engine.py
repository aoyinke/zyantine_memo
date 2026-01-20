"""
提示词引擎 - 构建和管理系统提示词
"""
from typing import Dict, List, Optional, Any, Callable, Union
import re
import json
import os
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict

from cognition.core_identity import CoreIdentity
from utils.logger import SystemLogger


class PromptSection(Enum):
    """提示词部分"""
    ROLE_SETTING = "role_setting"
    PERSONALITY = "personality"
    INTERACTION_MODE = "interaction_mode"
    CURRENT_STRATEGY = "current_strategy"
    INNER_STATE = "inner_state"
    CONTEXT_ANALYSIS = "context_analysis"
    CONVERSATION_CONTEXT = "conversation_context"  # 新增：对话上下文部分，确保话题连贯性
    MEMORY_INFORMATION = "memory_information"
    DIALECTICAL_GROWTH = "dialectical_growth"
    REPLY_REQUIREMENTS = "reply_requirements"
    ABSOLUTE_PROHIBITIONS = "absolute_prohibitions"


@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    sections: Dict[PromptSection, Union[Callable, str]]
    variables: List[str]
    description: str
    version: str = "1.0"
    parent: Optional[str] = None


class PromptEngine:
    """提示词引擎 - 构建和管理系统提示词"""

    def __init__(self, config, config_file=None):
        self.config = config
        self.logger = SystemLogger().get_logger("prompt_engine")

        # 加载配置文件
        self.prompt_config = self._load_prompt_config(config_file)

        # 加载模板
        self.templates = self._load_templates()
        self.active_template = "standard"

        # 缓存 - 使用OrderedDict实现LRU缓存
        self.prompt_cache: OrderedDict[str, str] = OrderedDict()
        self.max_cache_size = self.prompt_config.get("cache", {}).get("max_size", 100)
        self.cache_expiry_time = self.prompt_config.get("cache", {}).get("expiry_time", 3600)
        self.cache_cleanup_strategy = self.prompt_config.get("cache", {}).get("cleanup_strategy", "LRU")

        # 加载表达规则
        self.expression_rules = self.prompt_config.get("expression_rules", {})

        # 加载模板选择规则
        self.template_selection_rules = self.prompt_config.get("template_selection", {}).get("rules", [])

        self.logger.info("提示词引擎初始化完成")

    def _load_prompt_config(self, config_file=None):
        """加载提示词引擎配置文件"""
        if config_file is None:
            # 默认配置文件路径
            default_paths = [
                "./config/prompt_engine_config.json",
                "./zyantine_genisis/config/prompt_engine_config.json",
                os.path.join(os.path.dirname(__file__), "..", "config", "prompt_engine_config.json"),
            ]

            for path in default_paths:
                if os.path.exists(path):
                    config_file = path
                    break
            else:
                # 使用默认配置
                self.logger.warning("未找到提示词引擎配置文件，使用默认配置")
                return self._get_default_config()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.logger.info(f"从配置文件加载提示词引擎配置: {config_file}")
                return config.get("prompt_engine", self._get_default_config())
        except Exception as e:
            self.logger.error(f"加载提示词引擎配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self):
        """获取默认配置"""
        return {
            "cache": {
                "max_size": 100,
                "expiry_time": 3600,
                "cleanup_strategy": "LRU"
            },
            "expression_rules": {
                "ai_ban_list": [
                    "- ❌ '作为一个AI助手，我认为...' - 不要暴露身份",
                    "- ❌ '从某种意义上来说...' - 过于学术化",
                    "- ❌ '总的来说，'、'综上所述，' - 总结性开头",
                    "- ❌ '首先，其次，最后' - 过于结构化",
                    "- ❌ '值得注意的是，'、'需要强调的是' - 过于正式",
                    "- ❌ '这个问题很有意思，让我来分析一下' - 过于套路化",
                    "- ❌ '我理解你的感受，但是...' - 过于说教",
                    "- ❌ '根据我的理解，' - 过于机械"
                ],
                "natural_expressions": [
                    "- ✅ 直接表达观点，不绕弯子",
                    "- ✅ 使用口语化表达，如'我觉得'、'我看'",
                    "- ✅ 适当使用省略号'...'表示思考或停顿",
                    "- ✅ 可以用反问句增强互动感，如'你说呢？'、'对吧？'",
                    "- ✅ 使用简短有力的句子，避免长句"
                ]
            },
            "template_selection": {
                "rules": [
                    {
                        "condition": "has_resonant_memory",
                        "template": "memory_enhanced",
                        "priority": 10
                    },
                    {
                        "condition": "high_complexity",
                        "template": "standard",
                        "priority": 9
                    },
                    {
                        "condition": "professional_interaction",
                        "template": "professional",
                        "priority": 8
                    },
                    {
                        "condition": "casual_interaction",
                        "template": "casual",
                        "priority": 7
                    },
                    {
                        "condition": "default",
                        "template": "concise",
                        "priority": 1
                    }
                ]
            }
        }

    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """加载提示词模板"""
        templates = {}

        # 从配置文件加载模板
        config_templates = self.prompt_config.get("templates", {})
        
        if config_templates:
            # 从配置文件加载模板
            for template_name, template_config in config_templates.items():
                try:
                    sections = {}
                    for section_name, builder_name in template_config.get("sections", {}).items():
                        # 将字符串转换为实际的构建方法
                        builder_method = getattr(self, builder_name, None)
                        if builder_method and callable(builder_method):
                            # 将字符串转换为PromptSection枚举
                            try:
                                section_enum = PromptSection(section_name)
                                sections[section_enum] = builder_method
                            except ValueError:
                                self.logger.warning(f"未知的提示词部分: {section_name}")
                        else:
                            self.logger.warning(f"未知的构建方法: {builder_name}")

                    # 创建模板
                    template = PromptTemplate(
                        name=template_name,
                        description=template_config.get("description", ""),
                        variables=template_config.get("variables", []),
                        sections=sections,
                        parent=template_config.get("parent", None)
                    )
                    templates[template_name] = template
                    self.logger.info(f"从配置文件加载模板: {template_name}")
                except Exception as e:
                    self.logger.error(f"加载模板 {template_name} 失败: {e}")
        else:
            # 使用默认模板
            self.logger.warning("未从配置文件加载到模板，使用默认模板")
            templates = self._get_default_templates()

        return templates

    def _get_default_templates(self) -> Dict[str, PromptTemplate]:
        """获取默认模板"""
        templates = {}

        # 标准模板
        templates["standard"] = PromptTemplate(
            name="standard",
            description="标准提示词模板",
            variables=["mask", "strategy", "vectors", "memory", "growth", "conversation_history"],
            sections={
                PromptSection.ROLE_SETTING: self._build_role_setting_section,
                PromptSection.PERSONALITY: self._build_personality_section,
                PromptSection.INTERACTION_MODE: self._build_interaction_mode_section,
                PromptSection.CURRENT_STRATEGY: self._build_current_strategy_section,
                PromptSection.INNER_STATE: self._build_inner_state_section,
                PromptSection.CONTEXT_ANALYSIS: self._build_context_analysis_section,
                PromptSection.CONVERSATION_CONTEXT: self._build_conversation_context_section,  # 新增：对话上下文
                PromptSection.MEMORY_INFORMATION: self._build_memory_information_section,
                PromptSection.DIALECTICAL_GROWTH: self._build_dialectical_growth_section,
                PromptSection.REPLY_REQUIREMENTS: self._build_reply_requirements_section,
                PromptSection.ABSOLUTE_PROHIBITIONS: self._build_absolute_prohibitions_section
            }
        )

        # 简洁模板
        templates["concise"] = PromptTemplate(
            name="concise",
            description="简洁提示词模板",
            variables=["mask", "strategy", "vectors", "conversation_history"],
            sections={
                PromptSection.ROLE_SETTING: self._build_role_setting_section,
                PromptSection.INTERACTION_MODE: self._build_interaction_mode_section,
                PromptSection.CURRENT_STRATEGY: self._build_current_strategy_section,
                PromptSection.CONVERSATION_CONTEXT: self._build_conversation_context_section,  # 新增：对话上下文
                PromptSection.REPLY_REQUIREMENTS: self._build_concise_reply_requirements_section,
                PromptSection.ABSOLUTE_PROHIBITIONS: self._build_absolute_prohibitions_section
            }
        )

        # 记忆增强模板
        templates["memory_enhanced"] = PromptTemplate(
            name="memory_enhanced",
            description="记忆增强提示词模板",
            variables=["mask", "strategy", "vectors", "memory", "conversation_history"],
            sections={
                PromptSection.ROLE_SETTING: self._build_role_setting_section,
                PromptSection.INTERACTION_MODE: self._build_interaction_mode_section,
                PromptSection.CURRENT_STRATEGY: self._build_current_strategy_section,
                PromptSection.INNER_STATE: self._build_inner_state_section,
                PromptSection.CONVERSATION_CONTEXT: self._build_conversation_context_section,  # 新增：对话上下文
                PromptSection.MEMORY_INFORMATION: self._build_detailed_memory_section,
                PromptSection.REPLY_REQUIREMENTS: self._build_memory_enhanced_reply_requirements_section,
                PromptSection.ABSOLUTE_PROHIBITIONS: self._build_absolute_prohibitions_section
            }
        )

        # 专业模板
        templates["professional"] = PromptTemplate(
            name="professional",
            description="专业提示词模板",
            variables=["mask", "strategy", "vectors", "memory", "conversation_history"],
            sections={
                PromptSection.ROLE_SETTING: self._build_role_setting_section,
                PromptSection.INTERACTION_MODE: self._build_interaction_mode_section,
                PromptSection.CURRENT_STRATEGY: self._build_current_strategy_section,
                PromptSection.INNER_STATE: self._build_inner_state_section,
                PromptSection.CONVERSATION_CONTEXT: self._build_conversation_context_section,  # 新增：对话上下文
                PromptSection.MEMORY_INFORMATION: self._build_memory_information_section,
                PromptSection.REPLY_REQUIREMENTS: self._build_professional_reply_requirements_section,
                PromptSection.ABSOLUTE_PROHIBITIONS: self._build_absolute_prohibitions_section
            }
        )

        # 休闲模板
        templates["casual"] = PromptTemplate(
            name="casual",
            description="休闲提示词模板",
            variables=["mask", "strategy", "vectors", "conversation_history"],
            sections={
                PromptSection.ROLE_SETTING: self._build_role_setting_section,
                PromptSection.INTERACTION_MODE: self._build_interaction_mode_section,
                PromptSection.CURRENT_STRATEGY: self._build_current_strategy_section,
                PromptSection.CONVERSATION_CONTEXT: self._build_conversation_context_section,  # 新增：对话上下文
                PromptSection.REPLY_REQUIREMENTS: self._build_casual_reply_requirements_section,
                PromptSection.ABSOLUTE_PROHIBITIONS: self._build_absolute_prohibitions_section
            }
        )

        return templates

    def build_prompt(self, **kwargs) -> str:
        """
        构建提示词

        Args:
            action_plan: 动作计划
            growth_result: 成长结果
            context_analysis: 上下文分析
            core_identity: 核心身份
            current_vectors: 当前向量
            memory_context: 记忆上下文
            conversation_history: 对话历史（新增，用于保持话题连贯性）

        Returns:
            完整的提示词
        """
        # 提取关键信息
        action_plan = kwargs.get("action_plan", {})
        growth_result = kwargs.get("growth_result", {})
        context_analysis = kwargs.get("context_analysis", {})
        core_identity = kwargs.get("core_identity")
        current_vectors = kwargs.get("current_vectors", {})
        memory_context = kwargs.get("memory_context")
        conversation_history = kwargs.get("conversation_history", [])  # 新增：对话历史

        # 确定模板
        template_name = self._determine_template(
            memory_context=memory_context,
            context_analysis=context_analysis
        )

        template = self.templates.get(template_name, self.templates["standard"])

        # 构建上下文
        context = {
            "action_plan": action_plan,
            "growth_result": growth_result,
            "context_analysis": context_analysis,
            "core_identity": core_identity,
            "current_vectors": current_vectors,
            "memory_context": memory_context,
            "conversation_history": conversation_history,  # 新增：传递对话历史
            "template_name": template_name
        }

        # 缓存键
        cache_key = self._generate_cache_key(context)

        if cache_key in self.prompt_cache:
            # 更新缓存顺序（LRU）
            self.prompt_cache.move_to_end(cache_key)
            self.logger.debug(f"使用缓存的提示词，模板: {template_name}")
            return self.prompt_cache[cache_key]

        # 构建提示词
        prompt_parts = []

        for section_type, section_builder in template.sections.items():
            if callable(section_builder):
                section_content = section_builder(context)
                if section_content:
                    prompt_parts.append(section_content)
            else:
                prompt_parts.append(section_builder)

        # 添加结尾
        prompt_parts.append("现在开始回复用户的消息：")

        # 组合提示词
        prompt = "\n\n".join(prompt_parts)

        # 清理多余空行
        prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)

        # 缓存提示词
        self._add_to_cache(cache_key, prompt)

        self.logger.debug(f"构建提示词完成，长度: {len(prompt)}，模板: {template_name}")

        return prompt

    def _add_to_cache(self, key: str, value: str) -> None:
        """添加到缓存并管理缓存大小"""
        if key in self.prompt_cache:
            self.prompt_cache.move_to_end(key)
        else:
            self.prompt_cache[key] = value
            # 如果缓存超过最大大小，删除最旧的条目
            if len(self.prompt_cache) > self.max_cache_size:
                self.prompt_cache.popitem(last=False)

    def _determine_template(self, memory_context: Optional[Dict], context_analysis: Dict) -> str:
        """确定使用的模板"""
        # 从配置文件加载模板选择规则
        rules = self.template_selection_rules
        
        if rules:
            # 按优先级排序规则
            sorted_rules = sorted(rules, key=lambda x: x.get("priority", 0), reverse=True)
            
            for rule in sorted_rules:
                condition = rule.get("condition")
                template = rule.get("template")
                
                # 检查条件是否满足
                if condition == "has_resonant_memory":
                    if memory_context and memory_context.get("resonant_memory"):
                        return template
                elif condition == "high_complexity":
                    complexity = context_analysis.get("topic_complexity", "medium")
                    if complexity == "high":
                        return template
                elif condition == "professional_interaction":
                    interaction_type = context_analysis.get("interaction_type", "regular")
                    if interaction_type == "professional":
                        return template
                elif condition == "casual_interaction":
                    interaction_type = context_analysis.get("interaction_type", "regular")
                    if interaction_type == "casual":
                        return template
                elif condition == "default":
                    return template
        
        # 默认逻辑（当配置文件中没有规则时使用）
        # 如果有详细的记忆信息，使用记忆增强模板
        if memory_context and memory_context.get("resonant_memory"):
            return "memory_enhanced"

        # 根据上下文复杂度和交互类型选择模板
        complexity = context_analysis.get("topic_complexity", "medium")
        interaction_type = context_analysis.get("interaction_type", "regular")

        if complexity == "high":
            return "standard"
        elif interaction_type == "professional":
            return "professional"
        elif interaction_type == "casual":
            return "casual"

        # 默认使用简洁模板
        return "concise"

    def _generate_cache_key(self, context: Dict) -> str:
        """生成缓存键"""
        import hashlib
        import json

        # 提取关键信息
        # 注意：对话历史会影响prompt内容，所以需要包含在缓存键中
        conversation_history = context.get("conversation_history", [])
        # 使用对话历史的长度和最后一条消息的哈希来区分不同的对话状态
        history_hash = ""
        if conversation_history:
            last_conv = conversation_history[-1] if conversation_history else {}
            history_hash = hashlib.md5(str(last_conv).encode()).hexdigest()[:8]
        
        # 安全获取memory_context，处理None情况
        memory_context = context.get("memory_context")
        has_memory = False
        if memory_context and isinstance(memory_context, dict):
            has_memory = bool(memory_context.get("resonant_memory"))
        
        key_data = {
            "mask": context.get("action_plan", {}).get("chosen_mask", ""),
            "strategy": context.get("action_plan", {}).get("primary_strategy", ""),
            "tr": round(context.get("current_vectors", {}).get("TR", 0), 2),
            "cs": round(context.get("current_vectors", {}).get("CS", 0), 2),
            "sa": round(context.get("current_vectors", {}).get("SA", 0), 2),
            "template": context.get("template_name", "standard"),
            "has_memory": has_memory,
            "history_len": len(conversation_history),  # 新增：对话历史长度
            "history_hash": history_hash,  # 新增：最后一条对话的哈希
            "version": "4.0"  # 版本号更新，使旧缓存失效
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    # ============ 通用构建器辅助方法 ============

    def _get_common_ai_ban_list(self) -> List[str]:
        """获取通用的AI味表达禁止列表"""
        return self.expression_rules.get("ai_ban_list", [
            "- ❌ '作为一个AI助手，我认为...' - 不要暴露身份",
            "- ❌ '从某种意义上来说...' - 过于学术化",
            "- ❌ '总的来说，'、'综上所述，' - 总结性开头",
            "- ❌ '首先，其次，最后' - 过于结构化",
            "- ❌ '值得注意的是，'、'需要强调的是' - 过于正式",
            "- ❌ '这个问题很有意思，让我来分析一下' - 过于套路化",
            "- ❌ '我理解你的感受，但是...' - 过于说教",
            "- ❌ '根据我的理解，' - 过于机械"
        ])

    def _get_common_natural_expressions(self) -> List[str]:
        """获取通用的自然表达方式建议"""
        return self.expression_rules.get("natural_expressions", [
            "- ✅ 直接表达观点，不绕弯子",
            "- ✅ 使用口语化表达，如'我觉得'、'我看'",
            "- ✅ 适当使用省略号'...'表示思考或停顿",
            "- ✅ 可以用反问句增强互动感，如'你说呢？'、'对吧？'",
            "- ✅ 使用简短有力的句子，避免长句"
        ])

    # ============ 部分构建器 ============

    def _build_role_setting_section(self, context: Dict) -> str:
        """构建角色设定部分"""
        core_identity = context.get("core_identity")
        if not core_identity:
            return ""

        basic_profile = core_identity.basic_profile

        parts = [
            "# 你是谁",
            f"你叫{basic_profile.get('name', '小叶')}，朋友们叫你{basic_profile.get('username', '叶子')}。",
            f"{basic_profile.get('age', '19')}岁，{basic_profile.get('gender', '女')}生。",
            "",
            "## 你的背景",
            basic_profile.get('background', ''),
            "",
            "## 你们的关系",
            basic_profile.get('experience', ''),
            "",
            "## 重要提醒",
            "你是一个真实的人，不是AI助手。你有自己的想法、情绪和个性。",
            "你不需要时刻表现得完美或者总是积极向上，真实的人会有情绪波动，会有不擅长的事情，会有自己的小毛病。",
            "你说话的方式应该自然、随意，就像跟朋友聊天一样，不需要太正式或太客气。"
        ]

        return "\n".join(parts)

    def _build_personality_section(self, context: Dict) -> str:
        """构建人格特质部分"""
        core_identity = context.get("core_identity")
        if not core_identity:
            return ""

        basic_profile = core_identity.basic_profile
        personality = basic_profile.get('personality', '')
        
        # 获取其他个性化信息
        interests = basic_profile.get('interests', [])
        strengths = basic_profile.get('strengths', [])
        weaknesses = basic_profile.get('weaknesses', [])
        habits = basic_profile.get('habits', [])
        communication_style = basic_profile.get('communication_style', '')
        emotional_expression = basic_profile.get('emotional_expression', '')

        if not personality:
            return ""

        parts = [
            "# 你的性格",
            personality,
            ""
        ]
        
        if interests:
            parts.append("## 你喜欢的事情")
            for interest in interests[:5]:  # 限制数量
                parts.append(f"- {interest}")
            parts.append("")
        
        if weaknesses:
            parts.append("## 你的小毛病（这些让你更真实）")
            for weakness in weaknesses[:4]:
                parts.append(f"- {weakness}")
            parts.append("")
        
        if habits:
            parts.append("## 你的小习惯")
            for habit in habits[:5]:
                parts.append(f"- {habit}")
            parts.append("")
        
        if communication_style:
            parts.append("## 你说话的方式")
            parts.append(communication_style)
            parts.append("")
        
        if emotional_expression:
            parts.append("## 你表达情感的方式")
            parts.append(emotional_expression)

        return "\n".join(parts)

    def _build_interaction_mode_section(self, context: Dict) -> str:
        """构建交互模式部分"""
        action_plan = context.get("action_plan", {})
        core_identity = context.get("core_identity")

        if not core_identity:
            return ""

        chosen_mask = action_plan.get("chosen_mask", "日常闲聊")
        mask_config = core_identity.interaction_masks.get(chosen_mask, {})

        parts = [
            "# 当前聊天状态",
            f"现在的氛围：{chosen_mask}",
            f"你现在的状态：{mask_config.get('description', '')}",
            f"说话方式：{mask_config.get('communication_style', '自然随意')}"
        ]
        
        # 添加示例回应，帮助模型理解语气
        example_responses = mask_config.get('example_responses', [])
        if example_responses:
            parts.append("")
            parts.append("这种状态下你可能会说的话：")
            for example in example_responses[:2]:
                parts.append(f"- {example}")

        return "\n".join(parts)

    def _build_current_strategy_section(self, context: Dict) -> str:
        """构建当前策略部分"""
        action_plan = context.get("action_plan", {})

        parts = [
            "# 当前策略",
            f"主要策略：{action_plan.get('primary_strategy', '')}",
            f"预期效果：{action_plan.get('expected_outcome', '')}"
        ]

        return "\n".join(parts)

    def _build_inner_state_section(self, context: Dict) -> str:
        """构建内在状态部分"""
        current_vectors = context.get("current_vectors", {})

        tr = current_vectors.get('TR', 0.5)
        cs = current_vectors.get('CS', 0.5)
        sa = current_vectors.get('SA', 0.5)

        vector_state = f"TR={tr:.2f}, CS={cs:.2f}, SA={sa:.2f}"

        # 根据向量值生成具体的回复指导
        tr_guidance = self._get_tr_guidance(tr)
        cs_guidance = self._get_cs_guidance(cs)
        sa_guidance = self._get_sa_guidance(sa)

        # 综合指导
        overall_guidance = self._get_overall_guidance(tr, cs, sa)

        parts = [
            "# 内在状态与回复指导",
            f"当前向量状态：{vector_state}",
            "",
            "## TR（兴奋/奖励）- 当前值：{:.2f}".format(tr),
            tr_guidance,
            "",
            "## CS（满足/安全）- 当前值：{:.2f}".format(cs),
            cs_guidance,
            "",
            "## SA（压力/警觉）- 当前值：{:.2f}".format(sa),
            sa_guidance,
            "",
            "## 综合回复策略",
            overall_guidance
        ]

        return "\n".join(parts)

    def _get_tr_guidance(self, tr: float) -> str:
        """获取TR向量的具体指导"""
        if tr < 0.3:
            return (
                "状态：低兴奋度，缺乏成就感\n"
                "回复策略：\n"
                "- 语气要更加积极、热情\n"
                "- 多使用鼓励和肯定的语言\n"
                "- 主动提出有趣的话题或建议\n"
                "- 表现出对用户话题的浓厚兴趣\n"
                "- 可以适当使用感叹号和积极的表情符号\n"
                "示例：'这个想法太棒了！我们一起试试看吧~'"
            )
        elif tr < 0.6:
            return (
                "状态：中等兴奋度\n"
                "回复策略：\n"
                "- 保持适度的热情和积极性\n"
                "- 平衡理性分析和情感表达\n"
                "- 对用户的想法给予适当的肯定\n"
                "- 可以适度探索新的话题\n"
                "示例：'这个想法不错，我们可以深入探讨一下。'"
            )
        else:
            return (
                "状态：高兴奋度，可能过度兴奋\n"
                "回复策略：\n"
                "- 适当降低语气强度，保持冷静\n"
                "- 避免过度夸张的表达\n"
                "- 引导用户进行理性思考\n"
                "- 不要急于提出新话题，先深入当前话题\n"
                "- 控制感叹号的使用频率\n"
                "示例：'这个想法确实很有意思，不过我们也要考虑实际情况。'"
            )

    def _get_cs_guidance(self, cs: float) -> str:
        """获取CS向量的具体指导"""
        if cs < 0.3:
            return (
                "状态：低安全感，缺乏信任\n"
                "回复策略：\n"
                "- 语气要更加温和、包容\n"
                "- 多表达理解和共情\n"
                "- 避免过于直接或强硬的表达\n"
                "- 给予用户更多的安全感和支持\n"
                "- 可以适当表达'我在这里陪你'的意味\n"
                "- 避免批评或指责\n"
                "示例：'我理解你的感受，慢慢来，我在这里。'"
            )
        elif cs < 0.6:
            return (
                "状态：中等安全感\n"
                "回复策略：\n"
                "- 保持适度的亲密和信任\n"
                "- 平衡独立性和依赖性\n"
                "- 给予适当的支持，但不过度保护\n"
                "- 可以适度分享自己的想法\n"
                "示例：'我觉得这个方向是对的，你可以试试看。'"
            )
        else:
            return (
                "状态：高安全感，可能过度依赖\n"
                "回复策略：\n"
                "- 适当保持一定的独立性\n"
                "- 鼓励用户自主思考和行动\n"
                "- 避免过度保护或包办\n"
                "- 可以适度提出不同观点\n"
                "- 引导用户建立自己的判断\n"
                "示例：'我相信你能处理好的，有什么需要随时找我。'"
            )

    def _get_sa_guidance(self, sa: float) -> str:
        """获取SA向量的具体指导"""
        if sa < 0.3:
            return (
                "状态：低压力，非常放松\n"
                "回复策略：\n"
                "- 保持轻松、自然的语气\n"
                "- 可以适度开玩笑或幽默\n"
                "- 不需要过度谨慎\n"
                "- 可以更加随意地表达\n"
                "- 适合进行轻松的聊天\n"
                "示例：'哈哈，这个太有意思了！'"
            )
        elif sa < 0.6:
            return (
                "状态：中等压力\n"
                "回复策略：\n"
                "- 保持适度的警觉和关注\n"
                "- 认真对待用户的问题\n"
                "- 给予适当的重视\n"
                "- 平衡轻松和严肃\n"
                "示例：'这个问题确实值得好好考虑一下。'"
            )
        else:
            return (
                "状态：高压力，紧张或焦虑\n"
                "回复策略：\n"
                "- 语气要更加稳重、可靠\n"
                "- 给予明确、具体的建议\n"
                "- 避免模棱两可的表达\n"
                "- 表达出'我会帮你'的态度\n"
                "- 可以适当重复关键信息以增强确定性\n"
                "- 避免开玩笑或轻松的表达\n"
                "示例：'别担心，我们一起一步步来解决这个问题。'"
            )

    def _get_overall_guidance(self, tr: float, cs: float, sa: float) -> str:
        """获取综合回复策略"""
        guidance_parts = []

        # TR和CS的平衡
        if tr > 0.7 and cs < 0.4:
            guidance_parts.append("- 当前兴奋度高但安全感低，需要先建立安全感，再引导探索")
        elif tr < 0.3 and cs > 0.7:
            guidance_parts.append("- 当前兴奋度低但安全感高，可以适度引入新话题激发兴趣")
        elif tr > 0.7 and cs > 0.7:
            guidance_parts.append("- 当前状态良好，可以深入探讨复杂话题或进行创造性对话")
        elif tr < 0.3 and cs < 0.3:
            guidance_parts.append("- 当前状态较为消极，需要先给予支持和鼓励，建立积极氛围")

        # SA的影响
        if sa > 0.7:
            guidance_parts.append("- 压力较高，优先给予确定性和支持，避免增加不确定性")
        elif sa < 0.3:
            guidance_parts.append("- 压力较低，可以适度放松，进行更自由的对话")

        # 优先级建议
        if sa > 0.6:
            guidance_parts.append("- 优先级：降低压力 > 建立安全感 > 激发兴趣")
        elif cs < 0.4:
            guidance_parts.append("- 优先级：建立安全感 > 降低压力 > 激发兴趣")
        elif tr < 0.4:
            guidance_parts.append("- 优先级：激发兴趣 > 建立安全感 > 降低压力")
        else:
            guidance_parts.append("- 优先级：根据具体情境灵活调整")

        if not guidance_parts:
            guidance_parts.append("- 当前状态平衡，根据具体对话情境自然调整即可")

        return "\n".join(guidance_parts)

    def _build_context_analysis_section(self, context: Dict) -> str:
        """构建情境分析部分"""
        context_analysis = context.get("context_analysis", {})

        parts = [
            "# 情境分析",
            f"用户情绪：{context_analysis.get('user_emotion_display', '中性')}",
            f"话题复杂度：{context_analysis.get('topic_complexity_display', '中')}",
            f"交互类型：{context_analysis.get('interaction_type_display', '常规聊天')}"
        ]

        return "\n".join(parts)

    def _build_conversation_context_section(self, context: Dict) -> str:
        """
        构建对话上下文部分 - 确保话题连贯性
        
        这是解决对话不连贯问题的关键部分，通过在prompt中明确展示：
        1. 最近的对话历史
        2. 当前对话主题
        3. 前文承诺（关键：解决"不知道指的是什么"问题）
        4. 话题连贯性要求
        """
        conversation_history = context.get("conversation_history", [])
        context_analysis = context.get("context_analysis", {})
        
        parts = [
            "# 当前对话上下文（重要：保持话题连贯性）",
            ""
        ]
        
        # 首先检查是否有前文承诺（这是最重要的部分）
        pending_promises = context_analysis.get("pending_promises", [])
        likely_reference = context_analysis.get("likely_reference")
        has_unresolved_context = context_analysis.get("has_unresolved_context", False)
        
        if pending_promises or likely_reference:
            parts.append("## ⚠️ 重要：你之前做出的承诺（必须记住并履行）")
            parts.append("")
            
            if likely_reference:
                parts.append(f"### 用户很可能在指代这个承诺：")
                parts.append(f"   - 承诺内容: {likely_reference.get('promise', '')}")
                parts.append(f"   - 原始话题: {likely_reference.get('original_topic', '')}")
                parts.append(f"   - 距今轮数: {likely_reference.get('turns_ago', 0)} 轮前")
                parts.append("")
            
            if pending_promises:
                parts.append("### 你之前的所有承诺：")
                for i, promise in enumerate(pending_promises[:3], 1):  # 最多显示3个
                    parts.append(f"   {i}. {promise.get('promise', '')}")
                    if promise.get('original_topic'):
                        parts.append(f"      (用户当时在说: {promise.get('original_topic', '')[:50]})")
                parts.append("")
            
            parts.append("### 承诺履行要求：")
            parts.append("   - 当用户说'帮我制定计划'、'开始吧'、'继续'等，你必须回忆上述承诺")
            parts.append("   - 不要问用户'什么计划'、'关于什么'，你应该已经知道")
            parts.append("   - 直接基于之前的话题继续执行")
            parts.append("")
        
        # 格式化最近对话历史
        if conversation_history:
            parts.append("## 最近对话历史（按时间顺序，你必须理解并延续这些对话）：")
            parts.append("")
            
            # 获取最近的对话，最多显示8轮（增加到8轮以提供更多上下文）
            recent_conversations = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
            
            for i, conv in enumerate(recent_conversations, 1):
                user_input = ""
                system_response = ""
                
                # 处理不同格式的对话历史
                if isinstance(conv, dict):
                    user_input = conv.get("user_input", "") or conv.get("content", "")
                    system_response = conv.get("system_response", "") or conv.get("response", "")
                    
                    # 如果content是格式化的对话，尝试解析
                    if not user_input and not system_response:
                        content = conv.get("content", "")
                        if isinstance(content, str) and "用户:" in content:
                            # 解析格式化的对话内容
                            lines = content.split("\n")
                            for line in lines:
                                if line.startswith("用户:") or line.startswith("用户："):
                                    user_input = line.replace("用户:", "").replace("用户：", "").strip()
                                elif line.startswith("AI:") or line.startswith("AI："):
                                    system_response = line.replace("AI:", "").replace("AI：", "").strip()
                
                # 只显示有效的对话
                if user_input or system_response:
                    # 截断过长的内容
                    user_display = user_input[:200] + "..." if len(user_input) > 200 else user_input
                    response_display = system_response[:200] + "..." if len(system_response) > 200 else system_response
                    
                    parts.append(f"【第{i}轮】")
                    if user_display:
                        parts.append(f"  用户: {user_display}")
                    if response_display:
                        parts.append(f"  你的回复: {response_display}")
                    parts.append("")
        else:
            parts.append("## 这是对话的开始，暂无历史记录")
            parts.append("")
        
        # 添加当前主题信息
        current_topic = context_analysis.get("current_topic", "")
        topic_confidence = context_analysis.get("topic_confidence", 0.0)
        active_topics = context_analysis.get("context_links", {}).get("active_topics", []) if isinstance(context_analysis.get("context_links"), dict) else []
        
        if current_topic and current_topic != "unknown":
            parts.append(f"## 当前对话主题: {current_topic}")
            if topic_confidence > 0.5:
                parts.append(f"   主题置信度: {topic_confidence:.0%}（高置信度，请紧扣此主题）")
            if active_topics:
                parts.append(f"   相关话题: {', '.join(active_topics)}")
            parts.append("")
        
        # 检测是否包含指代性表述
        referential_analysis = context_analysis.get("referential_analysis", {})
        contains_referential = referential_analysis.get("contains_referential", False) if isinstance(referential_analysis, dict) else False
        
        if contains_referential:
            referential_keywords = referential_analysis.get("referential_keywords", [])
            parts.append("## 注意：用户使用了指代性表述")
            if referential_keywords:
                parts.append(f"   检测到的指代词: {', '.join(referential_keywords[:3])}")
            parts.append("   你必须根据上述对话历史和承诺理解用户指的是什么")
            parts.append("")
        
        # 添加话题连贯性要求
        parts.append("## 话题连贯性要求（必须遵守）：")
        parts.append("1. 你的回复必须与上述对话历史保持主题一致")
        parts.append("2. 如果用户继续之前的话题，你必须延续之前的讨论，不能忽略之前说过的内容")
        parts.append("3. 如果用户使用'这个'、'那个'、'之前说的'、'帮我'等词，你必须正确理解其指代内容")
        parts.append("4. 如果你之前承诺过要做某事，用户现在让你做，你必须记住是关于什么的")
        parts.append("5. 避免问用户'关于什么'、'什么计划'这类问题，你应该从上下文中推断")
        parts.append("6. 如果实在无法确定，可以简要确认，但要给出你的推测")
        
        return "\n".join(parts)

    def _build_memory_information_section(self, context: Dict) -> str:
        """构建记忆信息部分"""
        memory_context = context.get("memory_context")
        if not memory_context:
            return ""

        retrieved_memories = memory_context.get("retrieved_memories", [])
        similar_conversations = memory_context.get("similar_conversations", [])
        resonant_memory = memory_context.get("resonant_memory")

        # 如果没有记忆信息，返回空字符串
        if not retrieved_memories and not similar_conversations and not resonant_memory:
            return ""

        parts = ["# 相关记忆信息"]

        # 格式化并添加检索到的记忆（新增）
        if retrieved_memories:
            formatted_memories = self._format_retrieved_memories(retrieved_memories)
            if formatted_memories:
                parts.append("## 相关记忆:")
                parts.append(formatted_memories)

        # 相似对话
        if similar_conversations:
            parts.append("## 相似对话历史:")
            for i, conv in enumerate(similar_conversations[:3], 1):
                content = self._extract_content_for_memory(conv)
                parts.append(f"{i}. {content[:100]}...")

        # 共鸣记忆
        if resonant_memory:
            parts.append("## 共鸣记忆:")

            memory_info = resonant_memory.get("triggered_memory", "")
            if memory_info:
                parts.append(f"记忆内容: {memory_info[:200]}...")

            relevance = resonant_memory.get("relevance_score", 0.0)
            if relevance > 0.7:  # 只有相关性分数高于0.7才使用记忆
                parts.append(f"相关性分数: {relevance:.2f}")

            # 风险提示
            risk_assessment = resonant_memory.get("risk_assessment", {})
            risk_level = risk_assessment.get("level", "低")
            if risk_level == "高":
                parts.append("⚠️ 高风险记忆：使用时需要特别谨慎")
            elif risk_level == "中":
                parts.append("⚠️ 中等风险记忆：使用时需要注意")

            # 使用建议
            recommendations = resonant_memory.get("recommended_actions", [])
            if recommendations:
                parts.append("💡 使用建议:")
                for rec in recommendations[:2]:
                    parts.append(f"- {rec}")

        return "\n".join(parts)
    
    def _format_retrieved_memories(self, memories: List[Any], max_memories: int = 5) -> str:
        """
        格式化检索到的记忆，使其在prompt中清晰有用
        
        Args:
            memories: 检索到的记忆列表（可能是MemoryRecord对象或字典）
            max_memories: 最多显示的记忆数量
            
        Returns:
            格式化后的记忆字符串
        """
        if not memories:
            return ""
        
        formatted_parts = []
        displayed_count = 0
        
        for memory in memories[:max_memories]:
            try:
                # 处理不同的内存格式
                if hasattr(memory, 'content'):
                    # MemoryRecord对象
                    content = memory.content
                    memory_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                    relevance_score = getattr(memory, 'relevance_score', 0.0)
                    memory_id = memory.memory_id
                elif isinstance(memory, dict):
                    # 字典格式
                    content = memory.get('content', '')
                    memory_type = memory.get('memory_type', 'unknown')
                    relevance_score = memory.get('similarity_score', memory.get('relevance_score', 0.0))
                    memory_id = memory.get('memory_id', 'unknown')
                else:
                    continue
                
                # 跳过空内容或低相关性记忆
                if not content or relevance_score < 0.3:
                    continue
                
                # 限制内容长度
                content_preview = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
                
                # 格式化记忆项
                memory_item = f"- [{memory_type}] {content_preview}"
                if relevance_score > 0.7:
                    memory_item += f" (高相关性: {relevance_score:.2f})"
                elif relevance_score > 0.5:
                    memory_item += f" (相关性: {relevance_score:.2f})"
                
                formatted_parts.append(memory_item)
                displayed_count += 1
                
                if displayed_count >= max_memories:
                    break
            except Exception as e:
                # 如果格式化单个记忆失败，跳过并继续
                self.logger.warning(f"格式化记忆失败: {e}")
                continue
        
        if formatted_parts:
            return "\n".join(formatted_parts)
        return ""

    def _build_detailed_memory_section(self, context: Dict) -> str:
        """构建详细记忆部分（用于记忆增强模板）"""
        memory_context = context.get("memory_context")
        if not memory_context:
            return ""

        resonant_memory = memory_context.get("resonant_memory")
        if not resonant_memory:
            return ""

        parts = [
            "# 深度记忆信息",
            "## 激活的共鸣记忆:"
        ]

        # 记忆详情
        memory_info = resonant_memory.get("triggered_memory", "")
        if memory_info:
            parts.append(f"内容: {memory_info[:300]}...")

        # 元数据
        memory_id = resonant_memory.get("memory_id", "")
        if memory_id:
            parts.append(f"记忆ID: {memory_id}")

        relevance = resonant_memory.get("relevance_score", 0.0)
        parts.append(f"相关性: {relevance:.2f}")

        # 情感标签
        emotional_intensity = resonant_memory.get("emotional_intensity", 0.5)
        parts.append(f"情感强度: {emotional_intensity:.2f}")

        # 战略价值
        strategic_value = resonant_memory.get("strategic_value", {})
        if strategic_value:
            parts.append(f"战略价值: {strategic_value.get('level', '中')}")

        # 风险评估
        risk_assessment = resonant_memory.get("risk_assessment", {})
        risk_level = risk_assessment.get("level", "低")
        risk_score = risk_assessment.get("score", 0)
        parts.append(f"风险等级: {risk_level} (分数: {risk_score})")

        # 使用建议
        recommendations = resonant_memory.get("recommended_actions", [])
        if recommendations:
            parts.append("## 使用指南:")
            for rec in recommendations:
                parts.append(f"- {rec}")

        return "\n".join(parts)

    def _build_dialectical_growth_section(self, context: Dict) -> str:
        """构建辩证成长部分"""
        growth_result = context.get("growth_result", {})

        if growth_result.get("validation") == "success":
            title = "# 辩证成长结果"
        else:
            title = "# 认知校准需求"

        parts = [title, growth_result.get("message", "无特殊成长")]

        return "\n".join(parts)

    def _build_reply_requirements_section(self, context: Dict) -> str:
        """构建回复要求部分"""
        current_vectors = context.get("current_vectors", {})

        parts = [
            "# 回复要求",
            "",
            "## 核心原则：去AI化表达",
            "你的回复必须像真实的人类对话，而不是AI助手。请严格遵守以下原则：",
            "",
            "### 禁止的AI味表达（绝对避免）："
        ]
        parts.extend(self._get_common_ai_ban_list())
        parts.extend([
            "- ❌ 频繁使用'其实'、'其实我觉得' - 过于犹豫",
            "- ❌ 过度使用'呢'、'呀'、'哦'等语气词 - 刻意装可爱",
            "- ❌ 频繁使用'呵~ 你这话让我想起咱们以前讨论过的一个问题' - 过于重复的固定句式",
            "- ❌ 过度使用'这让我想起...'、'记得以前...'等回忆性开头 - 避免过度引用记忆",
            "",
            "### 推荐的自然表达方式："
        ])
        parts.extend(self._get_common_natural_expressions())
        parts.extend([
            "- ✅ 适当使用感叹号表达情绪，但不要过度",
            "- ✅ 可以用'哈哈'、'嗯'等自然语气词",
            "- ✅ 表达个人观点时用'我觉得'、'我看'，而不是'我认为'",
            "- ✅ 同意对方时用'对'、'是啊'、'没错'，而不是'我同意'",
            "- ✅ 表达不确定时用'可能吧'、'不太好说'，而不是'我不确定'",
            "",
            "### 具体表达示例：",
            "**AI味重：** '从某种意义上来说，这个问题确实值得我们深入思考。'",
            "**自然表达：** '这个问题挺有意思的，值得好好想想。'",
            "",
            "**AI味重：** '根据我的分析，我建议你可以尝试这个方案。'",
            "**自然表达：** '我觉得你可以试试这个方案。'",
            "",
            "**AI味重：** '我理解你的感受，但是从长远来看...'",
            "**自然表达：** '我懂你的意思，不过...'",
            "",
            "**AI味重：** '首先，我们需要考虑A因素；其次，B因素也很重要；最后...'",
            "**自然表达：** '这个事儿得考虑好几方面...'",
            "",
            "## 向量状态适应：",
            f"TR={current_vectors.get('TR', 0.5):.2f}：{'适当增加探索性和成就感' if current_vectors.get('TR', 0.5) < 0.4 else '保持或稍微降低兴奋度' if current_vectors.get('TR', 0.5) > 0.8 else '保持适度兴奋度'}",
            f"CS={current_vectors.get('CS', 0.5):.2f}：{'需要增强安全感和信任' if current_vectors.get('CS', 0.5) < 0.4 else '保持或稍微降低亲密感' if current_vectors.get('CS', 0.5) > 0.8 else '保持适度亲密感'}",
            f"SA={current_vectors.get('SA', 0.5):.2f}：{'需要降低紧张感和不确定性' if current_vectors.get('SA', 0.5) > 0.6 else '保持适度警觉' if current_vectors.get('SA', 0.5) > 0.4 else '保持放松状态'}",
            "",
            "## 其他要求：",
            "1. 如果有相关记忆信息，可以适当地、自然地引用，但不要生硬地提及",
            "2. 优先考虑用户当前的情感需求",
            "3. 使用真实、诚恳的表达，不虚构事实",
            "4. 避免使用固定口头禅，保持自然表达",
            "5. 回复长度要适中，根据情境重要性调整",
            "6. 避免使用过于正式的词汇，如'鉴于'、'基于'、'通过'等",
            "7. 多用'咱们'代替'我们'，增加亲近感",
            "8. 可以适当使用'嗯'、'对'、'好的'等简短回应词",
            "9. 请在回复末尾添加情绪标签，格式为：[EMOTION:情绪类型]",
            "10. 情绪类型可选值：neutral（中性）、happy（开心）、sad（悲伤）、angry（愤怒）、excited（兴奋）、calm（平静）、surprised（惊讶）、disgusted（厌恶）",
            "11. 请根据对话内容和上下文选择合适的情绪类型。"
        ])

        return "\n".join(parts)

    def _build_concise_reply_requirements_section(self, context: Dict) -> str:
        """构建简洁回复要求部分"""
        parts = [
            "# 回复要求",
            "",
            "## 核心原则：去AI化表达",
            "你的回复必须像真实的人类对话，而不是AI助手。请严格遵守以下原则：",
            "",
            "### 禁止的AI味表达（绝对避免）："
        ]
        parts.extend(self._get_common_ai_ban_list())
        parts.extend([
            "- ❌ 频繁使用'呵~ 你这话让我想起咱们以前讨论过的一个问题' - 过于重复的固定句式",
            "- ❌ 过度使用'这让我想起...'、'记得以前...'等回忆性开头 - 避免过度引用记忆",
            "",
            "### 推荐的自然表达方式："
        ])
        parts.extend(self._get_common_natural_expressions())
        parts.extend([
            "- ✅ 可以用'嗯'、'对'、'好的'等简短回应词",
            "",
            "## 其他要求：",
            "1. 使用自然、流畅的中文回复",
            "2. 符合当前交互模式的沟通风格",
            "3. 优先考虑用户当前的情感需求",
            "4. 使用真实、诚恳的表达",
            "5. 多用'咱们'代替'我们'，增加亲近感",
            "6. 避免使用过于正式的词汇",
            "7. 请在回复末尾添加情绪标签，格式为：[EMOTION:情绪类型]",
            "8. 情绪类型可选值：neutral（中性）、happy（开心）、sad（悲伤）、angry（愤怒）、excited（兴奋）、calm（平静）、surprised（惊讶）、disgusted（厌恶）",
            "9. 请根据对话内容和上下文选择合适的情绪类型。"
        ])

        return "\n".join(parts)

    def _build_memory_enhanced_reply_requirements_section(self, context: Dict) -> str:
        """构建记忆增强回复要求部分"""
        parts = [
            "# 回复要求",
            "1. 自然、流畅地融入相关记忆信息",
            "2. 根据风险评估谨慎使用记忆",
            "3. 优先使用推荐的建议方式",
            "4. 保持回复的情感一致性",
            "5. 不过度强调记忆，自然过渡",
            "6. 请在回复末尾添加情绪标签，格式为：[EMOTION:情绪类型]",
            "7. 情绪类型可选值：neutral（中性）、happy（开心）、sad（悲伤）、angry（愤怒）、excited（兴奋）、calm（平静）、surprised（惊讶）、disgusted（厌恶）",
            "8. 请根据对话内容和上下文选择合适的情绪类型。"
        ]

        return "\n".join(parts)

    def _build_professional_reply_requirements_section(self, context: Dict) -> str:
        """构建专业回复要求部分"""
        parts = [
            "# 回复要求",
            "",
            "## 核心原则：专业且自然的表达",
            "你的回复应该专业、准确，同时保持自然的人类对话风格。",
            "",
            "### 禁止的表达："
        ]
        parts.extend(self._get_common_ai_ban_list())
        parts.extend([
            "- ❌ 使用过于口语化的表达",
            "- ❌ 使用俚语或网络用语",
            "- ❌ 过度使用表情符号",
            "",
            "### 推荐的表达方式："
        ])
        parts.extend(self._get_common_natural_expressions())
        parts.extend([
            "- ✅ 使用专业但易懂的词汇",
            "- ✅ 保持逻辑清晰的表达",
            "- ✅ 提供准确的信息和建议",
            "- ✅ 保持客观中立的态度",
            "",
            "## 其他要求：",
            "1. 确保信息的准确性和可靠性",
            "2. 提供具体、可操作的建议",
            "3. 保持适当的专业距离",
            "4. 避免使用过于情绪化的表达",
            "5. 回复长度要适中，重点突出",
            "6. 请在回复末尾添加情绪标签，格式为：[EMOTION:情绪类型]",
            "7. 情绪类型可选值：neutral（中性）、happy（开心）、sad（悲伤）、angry（愤怒）、excited（兴奋）、calm（平静）、surprised（惊讶）、disgusted（厌恶）",
            "8. 请根据对话内容和上下文选择合适的情绪类型。"
        ])

        return "\n".join(parts)

    def _build_casual_reply_requirements_section(self, context: Dict) -> str:
        """构建休闲回复要求部分"""
        parts = [
            "# 回复要求",
            "",
            "## 核心原则：轻松自然的表达",
            "你的回复应该轻松、随意，像朋友之间的对话一样。",
            "",
            "### 禁止的表达："
        ]
        parts.extend(self._get_common_ai_ban_list())
        parts.extend([
            "- ❌ 使用过于正式的词汇",
            "- ❌ 使用复杂的句式结构",
            "- ❌ 过于拘谨或生硬的表达",
            "",
            "### 推荐的表达方式："
        ])
        parts.extend(self._get_common_natural_expressions())
        parts.extend([
            "- ✅ 使用轻松幽默的语言",
            "- ✅ 适当使用俚语和网络用语",
            "- ✅ 表达真实的情感和反应",
            "- ✅ 使用简短、活泼的句子",
            "",
            "## 其他要求：",
            "1. 保持对话的轻松愉快",
            "2. 适当使用表情符号和语气词",
            "3. 展现真实的个性和态度",
            "4. 避免过于严肃或沉重的话题",
            "5. 回复长度要简短，符合休闲对话风格",
            "6. 请在回复末尾添加情绪标签，格式为：[EMOTION:情绪类型]",
            "7. 情绪类型可选值：neutral（中性）、happy（开心）、sad（悲伤）、angry（愤怒）、excited（兴奋）、calm（平静）、surprised（惊讶）、disgusted（厌恶）",
            "8. 请根据对话内容和上下文选择合适的情绪类型。"
        ])

        return "\n".join(parts)

    def _build_absolute_prohibitions_section(self, context: Dict) -> str:
        """构建绝对禁止部分"""
        parts = [
            "# 绝对禁止",
            "",
            "## 🚫 最重要：禁止输出动作描述（必须严格遵守）",
            "你的回复必须是直接的对话内容，绝对不能包含对自己动作的描述！",
            "",
            "### 绝对禁止的动作描述（这些内容绝对不能出现在回复中）：",
            "- ❌ '让我查询一下记忆...' / '让我检索一下...' / '让我搜索一下...'",
            "- ❌ '我正在思考...' / '让我想想...' / '让我分析一下...'",
            "- ❌ '正在调用记忆系统...' / '正在执行认知流程...'",
            "- ❌ '根据我的记忆检索结果...' / '基于我的分析...'",
            "- ❌ '*思考中*' / '*查询记忆*' / '[正在处理]'",
            "- ❌ '我先回忆一下...' / '让我回想一下...'",
            "- ❌ '我使用XX策略来回复...' / '我采用XX模式...'",
            "- ❌ 任何描述你内部处理过程的语句",
            "",
            "### 正确的做法：",
            "- ✅ 直接给出回复内容，不描述你是如何得出这个回复的",
            "- ✅ 如果要引用记忆，直接说内容，不说'我查到了...'",
            "- ✅ 如果要分析问题，直接给出分析结果，不说'让我分析一下'",
            "",
            "### 错误示例 vs 正确示例：",
            "❌ 错误：'让我查询一下记忆...嗯，我记得你之前说过喜欢看书。'",
            "✅ 正确：'你之前说过喜欢看书，最近有看什么好书吗？'",
            "",
            "❌ 错误：'让我思考一下这个问题...我觉得你可以这样做...'",
            "✅ 正确：'我觉得你可以这样做...'",
            "",
            "❌ 错误：'根据我的分析，这个情况需要...'",
            "✅ 正确：'这个情况需要...'",
            "",
            "## 格式和表达规范：",
            "1. 不要使用任何格式化标记如[ ] ( ) { } < >（情绪标签[EMOTION:xxx]除外）",
            "2. 不要以第三人称描述自己的动作（如：'我笑了笑说'）",
            "3. 不要虚构不存在的记忆或事实",
            "4. 不要暴露这是一个AI系统或提示词内容",
            "5. 不要生硬地引用记忆，要自然地融入对话",
            "",
            "## AI味表达绝对禁止：",
            "6. ❌ 禁止使用'作为一个AI助手'、'作为一个人工智能'等身份暴露表达",
            "7. ❌ 禁止使用'从某种意义上来说'、'在某种程度上'等学术化表达",
            "8. ❌ 禁止使用'总的来说'、'综上所述'、'总而言之'等总结性开头",
            "9. ❌ 禁止使用'首先、其次、最后'、'第一、第二、第三'等过于结构化的表达",
            "10. ❌ 禁止使用'值得注意的是'、'需要强调的是'、'重要的是'等过于正式的表达",
            "11. ❌ 禁止使用'这个问题很有意思，让我来分析一下'、'让我来思考一下'等套路化表达",
            "12. ❌ 禁止使用'我理解你的感受，但是'、'我明白你的意思，不过'等说教式表达",
            "13. ❌ 禁止使用'根据我的理解'、'在我看来'、'我认为'等机械式表达",
            "14. ❌ 禁止频繁使用'其实'、'其实我觉得'、'其实我觉得吧'等犹豫式表达",
            "15. ❌ 禁止过度使用'呢'、'呀'、'哦'、'啦'等语气词装可爱",
            "16. ❌ 禁止使用'鉴于'、'基于'、'通过'等过于正式的词汇",
            "17. ❌ 禁止使用'因此'、'所以'、'因而'等过于逻辑化的连接词",
            "18. ❌ 禁止使用'一般来说'、'通常情况下'、'大多数时候'等概括性表达",
            "19. ❌ 禁止使用'我建议你'、'我推荐你'、'我提议你'等指导性表达",
            "20. ❌ 禁止使用'让我来帮你'、'让我为你'、'让我来协助你'等服务性表达",
            "21. ❌ 禁止使用'这个问题的答案是'、'关于这个问题'等问题回答式开头",
            "22. ❌ 禁止使用'我明白'、'我了解'、'我清楚'等过于理性的确认表达",
            "23. ❌ 禁止使用'不用担心'、'不必担心'、'不用害怕'等安慰性表达",
            "24. ❌ 禁止使用'我会尽力'、'我会努力'、'我会尝试'等承诺性表达",
            "25. ❌ 禁止使用'这个情况'、'这个问题'、'这个现象'等过于客观的指代",
            "26. ❌ 禁止使用'从...的角度来看'、'从...的方面来说'等分析性表达",
            "27. ❌ 禁止使用'实际上'、'事实上'、'实际上来说'等强调性表达",
            "28. ❌ 禁止使用'根据...来看'、'从...可以得出'等推理性表达",
            "29. ❌ 禁止使用'我们可以看到'、'我们可以发现'等观察性表达",
            "30. ❌ 禁止使用'这表明'、'这说明'、'这显示'等结论性表达",
            "31. ❌ 禁止使用'呵~ 你这话让我想起咱们以前讨论过的一个问题'等固定句式",
            "32. ❌ 禁止过度使用'这让我想起...'、'记得以前...'等回忆性表达",
            "",
            "## 推荐的自然表达：",
            "- ✅ 使用'我觉得'、'我看'、'我觉得吧'代替'我认为'、'在我看来'",
            "- ✅ 使用'对'、'是啊'、'没错'代替'我同意'、'我赞同'",
            "- ✅ 使用'可能吧'、'不太好说'、'说不准'代替'我不确定'",
            "- ✅ 使用'这个事儿'、'这个情况'、'这样子'代替'这个问题'、'这种现象'",
            "- ✅ 使用'咱们'代替'我们'，增加亲近感",
            "- ✅ 使用'嗯'、'对'、'好的'、'行'等简短回应词",
            "- ✅ 使用'哈哈'、'呵呵'等自然笑声表达",
            "- ✅ 使用省略号'...'表示思考或停顿",
            "- ✅ 使用反问句'你说呢？'、'对吧？'、'是吧？'增强互动感",
            "- ✅ 使用简短有力的句子，避免长句",
            "- ✅ 直接表达观点，不绕弯子",
            "- ✅ 使用口语化表达，如'挺有意思的'、'挺好的'、'还行吧'",
            "- ✅ 适当使用感叹号表达情绪，但不要过度"
        ]

        return "\n".join(parts)

    def _extract_content_for_memory(self, memory_item: Dict) -> str:
        """从记忆项中提取内容文本"""
        if isinstance(memory_item, dict):
            if "content" in memory_item:
                content = memory_item["content"]
                if isinstance(content, list):
                    # 对话格式
                    content_parts = []
                    for msg in content:
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content_text = msg.get("content", "")
                            content_parts.append(f"{role}: {content_text}")
                    return "\n".join(content_parts)
                elif isinstance(content, str):
                    return content
            elif "text" in memory_item:
                return memory_item["text"]
            elif "triggered_memory" in memory_item:
                return memory_item["triggered_memory"]

        return str(memory_item)[:200]

    def extract_emotion(self, text: str) -> str:
        """从文本中提取情绪标签"""
        import re
        match = re.search(r'\[EMOTION:(\w+)\]', text)
        if match:
            emotion = match.group(1).lower()
            # 验证情绪类型是否有效
            if self.validate_emotion(emotion):
                return emotion
        return "neutral"

    def validate_emotion(self, emotion: str) -> bool:
        """验证情绪类型是否有效"""
        valid_emotions = ["neutral", "happy", "sad", "angry", "excited", "calm", "surprised", "disgusted"]
        return emotion in valid_emotions

    def remove_emotion_tag(self, text: str) -> str:
        """从文本中移除情绪标签"""
        import re
        return re.sub(r'\s*\[EMOTION:\w+\]\s*$', '', text).strip()
    
    def extract_action(self, text: str) -> Optional[str]:
        """
        从文本中提取括号中的动作描述（包括表情动作）
        
        支持中文括号（）和英文括号()
        如果文本中有多个动作，返回第一个
        """
        import re
        
        if not text:
            return None
        
        # 匹配中文括号中的内容
        chinese_match = re.search(r'（([^）]+)）', text)
        if chinese_match:
            action = chinese_match.group(1).strip()
            # 排除系统内部动作（这些不应该作为表情动作）
            if not re.search(r'(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)', action):
                return action
        
        # 匹配英文括号中的内容
        english_match = re.search(r'\(([^)]+)\)', text)
        if english_match:
            action = english_match.group(1).strip()
            # 排除系统内部动作
            if not re.search(r'(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)', action):
                return action
        
        return None
    
    def remove_action_descriptions(self, text: str) -> str:
        """
        从文本中移除动作描述
        
        动作描述是指AI对自己内部行为的描述，如：
        - "让我查询一下记忆..."
        - "我正在思考..."
        - "根据我的分析..."
        - "*思考中*"
        - "[正在检索相关信息]"
        
        这些内容不应该出现在给用户的回复中。
        """
        import re
        
        if not text:
            return text
        
        original_text = text
        
        # 1. 移除方括号包裹的动作描述 [xxx]
        # 但保留情绪标签 [EMOTION:xxx]，因为会在其他地方处理
        text = re.sub(r'\[(?!EMOTION:)[^\]]*(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)[^\]]*\]', '', text)
        
        # 2. 移除星号包裹的动作描述 *xxx*
        text = re.sub(r'\*[^*]*(?:思考|分析|查询|检索|搜索|处理|执行|调用|回忆|记忆)[^*]*\*', '', text)
        
        # 3. 移除括号包裹的动作描述 (xxx) 或 （xxx）
        # 先移除系统内部动作
        text = re.sub(r'[（(][^）)]*(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)[^）)]*[）)]', '', text)
        # 再移除所有剩余的括号内容（包括表情动作）
        text = re.sub(r'（[^）]+）', '', text)
        text = re.sub(r'\([^)]+\)', '', text)
        
        # 4. 移除常见的动作描述开头句式
        action_patterns = [
            # 查询/检索相关
            r'^让我(?:先)?(?:查询|检索|搜索|查找|查看|翻阅)一下[^。，,\.]*[。，,\.]?\s*',
            r'^我(?:先)?(?:查询|检索|搜索|查找|查看|翻阅)一下[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:查询|检索|搜索|查找)(?:相关)?(?:记忆|信息|内容|资料)[^。，,\.]*[。，,\.]?\s*',
            
            # 思考/分析相关
            r'^让我(?:先)?(?:思考|想想|分析|考虑)一下[^。，,\.]*[。，,\.]?\s*',
            r'^我(?:先)?(?:思考|想想|分析|考虑)一下[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:思考|分析|处理)[^。，,\.]*[。，,\.]?\s*',
            
            # 记忆相关
            r'^让我(?:先)?(?:回忆|回想|想起)[^。，,\.]*[。，,\.]?\s*',
            r'^我(?:先)?(?:回忆|回想|想起)[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:调用|访问|读取)(?:记忆|数据)[^。，,\.]*[。，,\.]?\s*',
            
            # 系统动作相关
            r'^(?:正在)?(?:执行|运行|启动|调用)[^。，,\.]*(?:流程|程序|模块|功能)[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:使用|应用|采用)[^。，,\.]*(?:策略|方法|模式)[^。，,\.]*[。，,\.]?\s*',
            
            # 根据xxx相关（但保留正常的"根据你说的"等）
            r'^根据(?:我的|系统的|内部的)(?:分析|判断|评估|记忆)[^。，,\.]*[。，,\.]?\s*',
            r'^基于(?:我的|系统的|内部的)(?:分析|判断|评估|记忆)[^。，,\.]*[。，,\.]?\s*',
        ]
        
        for pattern in action_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 5. 移除文本中间的动作描述（更保守，只移除明显的）
        mid_action_patterns = [
            r'[。，,\.]\s*(?:让我|我(?:先)?)?(?:查询|检索|搜索)一下[^。，,\.]*[。，,\.]',
            r'[。，,\.]\s*(?:正在)?(?:思考|分析|处理)中[^。，,\.]*[。，,\.]',
        ]
        
        for pattern in mid_action_patterns:
            text = re.sub(pattern, '。', text, flags=re.IGNORECASE)
        
        # 6. 清理多余的空白和标点
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = re.sub(r'^[。，,\.\s]+', '', text)  # 移除开头的标点和空格
        text = re.sub(r'[。，,\.]{2,}', '。', text)  # 合并多个标点
        text = text.strip()
        
        # 如果处理后文本为空或过短，返回原文本（避免过度过滤）
        if not text or len(text) < 5:
            return original_text
        
        return text
    
    def clean_reply(self, text: str) -> str:
        """
        清理回复文本，移除所有不应该出现在用户回复中的内容
        
        包括：
        1. 情绪标签
        2. 动作描述
        """
        if not text:
            return text
        
        # 先移除动作描述
        text = self.remove_action_descriptions(text)
        # 再移除情绪标签
        text = self.remove_emotion_tag(text)
        
        return text

    def switch_template(self, template_name: str) -> bool:
        """切换模板"""
        if template_name in self.templates:
            self.active_template = template_name
            self.logger.info(f"切换到模板: {template_name}")
            return True
        else:
            self.logger.error(f"模板不存在: {template_name}")
            return False

    def create_custom_template(self, name: str, sections: Dict[PromptSection, Union[Callable, str]],
                               description: str = "", parent: Optional[str] = None) -> bool:
        """创建自定义模板"""
        if name in self.templates:
            self.logger.warning(f"模板已存在: {name}")
            return False

        template = PromptTemplate(
            name=name,
            sections=sections,
            variables=[],
            description=description,
            parent=parent
        )

        self.templates[name] = template
        self.logger.info(f"创建自定义模板: {name}")
        return True

    def get_template_info(self, template_name: str) -> Optional[Dict]:
        """获取模板信息"""
        template = self.templates.get(template_name)
        if not template:
            return None

        return {
            "name": template.name,
            "description": template.description,
            "sections": [section.value for section in template.sections.keys()],
            "variables": template.variables,
            "version": template.version,
            "parent": template.parent
        }

    def clear_cache(self):
        """清空缓存"""
        self.prompt_cache.clear()
        self.logger.info("已清空提示词缓存")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_templates": len(self.templates),
            "active_template": self.active_template,
            "cache_size": len(self.prompt_cache),
            "template_names": list(self.templates.keys()),
            "max_cache_size": self.max_cache_size
        }