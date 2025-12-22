from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from openai import OpenAI
from ..cognition.cognitive_flow import CoreIdentity
import random
import traceback


# ============ API服务模块 ============
class OpenAIService:
    """OpenAI API服务封装"""

    def __init__(self, api_key: str, base_url: str = "https://openkey.cloud/v1", model: str = "gpt-5-nano"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        self.request_count = 0
        self.error_count = 0
        self._initialize_client()

    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"[API服务] OpenAI客户端初始化成功，使用模型: {self.model}")
        except ImportError as e:
            print(f"[API服务] 导入失败: {str(e)}")
            print("[API服务] 警告：未安装openai库，请运行: pip install openai")
            self.client = None
        except Exception as e:
            print(f"[API服务] 初始化失败: {str(e)}")
            self.client = None

    def generate_reply(self, system_prompt: str, user_input: str,
                       conversation_history: List[Dict] = None,
                       max_completion_tokens: int = 500,
                       temperature: float = 1.0) -> Optional[Tuple[str, Optional[str]]]:
        """
        调用API生成回复

        Args:
            system_prompt: 系统提示词，定义AI的角色和行为
            user_input: 用户输入
            conversation_history: 对话历史
            max_tokens: 最大token数
            temperature: 温度参数

        Returns:
            Tuple[生成的回复文本或None, 错误信息或None]
        """
        self.request_count += 1

        if not self.client:
            error_msg = "客户端未初始化"
            print(f"[API服务] {error_msg}")
            self.error_count += 1
            return None, error_msg

        try:
            # 构建消息列表
            messages = []

            # 系统提示词
            messages.append({"role": "system", "content": system_prompt})

            # 添加历史对话
            if conversation_history:
                for item in conversation_history[-10:]:  # 只取最近10条历史
                    if "user_input" in item:
                        messages.append({"role": "user", "content": item["user_input"]})
                    if "system_response" in item:
                        messages.append({"role": "assistant", "content": item["system_response"]})

            # 当前用户输入
            messages.append({"role": "user", "content": user_input})

            print(
                f"[API服务] 开始调用API，模型: {self.model}, 消息数量: {len(messages)}, max_completion_tokens: {max_completion_tokens}, temperature: {temperature}")

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stream=False
            )

            if response and response.choices and len(response.choices) > 0:
                reply = response.choices[0].message.content
                print(f"[API服务] API调用成功，生成回复长度: {len(reply)}")
                return reply, None
            else:
                error_msg = "API响应为空或无效"
                print(f"[API服务] {error_msg}")
                self.error_count += 1
                return None, error_msg

        except Exception as e:
            error_msg = f"生成回复失败: {str(e)}"
            print(f"[API服务] {error_msg}")
            self.error_count += 1

            # 记录更详细的错误信息
            print(f"[API服务] 详细错误堆栈:\n{traceback.format_exc()}")

            return None, error_msg

    def is_available(self) -> bool:
        """检查API服务是否可用"""
        return self.client is not None

    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """测试API连接"""
        if not self.client:
            return False, "客户端未初始化"

        try:
            print(f"[API服务] 测试连接，使用模型: {self.model}")
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "测试连接，请回复'连接成功'"}],
                max_completion_tokens=10,
                temperature=1.0
            )

            if test_response and test_response.choices:
                print(f"[API服务] 连接测试成功")
                return True, None
            else:
                error_msg = "测试响应为空"
                print(f"[API服务] {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = f"连接测试失败: {str(e)}"
            print(f"[API服务] {error_msg}")
            return False, error_msg

    def get_statistics(self) -> Dict:
        """获取API服务统计信息"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "success_rate": ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100,
            "is_available": self.is_available()
        }


# ============ 改进的回复生成模块 ============
class APIBasedReplyGenerator:
    """基于API的智能回复生成器，适配memo0记忆系统"""

    def __init__(self, api_service: OpenAIService):
        self.api = api_service
        self.template_engine = TemplateReplyGenerator()  # 保留模板生成器作为备用
        self.generation_log = []
        self.error_log = []  # 新增错误日志

    def generate_reply(self, action_plan: Dict, growth_result: Dict,
                       user_input: str, context_analysis: Dict,
                       conversation_history: List[Dict],
                       core_identity: CoreIdentity,
                       current_vectors: Dict,
                       memory_context: Optional[Dict] = None) -> str:
        """
        使用API生成智能回复，适配新的记忆系统

        Args:
            action_plan: 认知流程生成的动作计划
            growth_result: 辩证成长结果
            user_input: 用户输入
            context_analysis: 情境分析结果
            conversation_history: 对话历史
            core_identity: 核心身份
            current_vectors: 当前向量状态
            memory_context: 记忆上下文

        Returns:
            生成的回复文本
        """
        # 记录生成请求信息
        print(f"[回复生成] 开始生成回复，用户输入长度: {len(user_input)}")

        # 如果API不可用，使用备用模板
        if not self.api.is_available():
            error_msg = "API服务不可用，使用模板生成器"
            print(f"[回复生成] {error_msg}")
            template_reply = self._generate_with_template(action_plan, growth_result, memory_context)
            return template_reply

        # 构建系统提示词
        system_prompt = self._build_system_prompt(
            action_plan, growth_result, context_analysis,
            core_identity, current_vectors, memory_context
        )

        print(f"[回复生成] 调用API生成回复，提示词长度: {len(system_prompt)}")
        print(f"[回复生成] 历史对话条数: {len(conversation_history) if conversation_history else 0}")

        # 调用API生成回复
        api_result = self.api.generate_reply(
            system_prompt=system_prompt,
            user_input=user_input,
            conversation_history=conversation_history,
            max_completion_tokens=self._determine_max_tokens(context_analysis),
            temperature=1.0
            # temperature=self._determine_temperature(current_vectors)
        )

        reply, api_error = api_result
        # 如果API调用失败，使用备用模板
        if not reply:
            error_msg = f"API调用失败: {api_error if api_error else '未知错误'}"
            print(f"[回复生成] {error_msg}")

            # 记录错误详情
            error_details = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "api_error": api_error,
                "action_plan": str(action_plan)[:200],
                "vectors": current_vectors,
                "system_prompt_preview": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt
            }
            self.error_log.append(error_details)

            # 限制错误日志大小
            if len(self.error_log) > 50:
                self.error_log = self.error_log[-50:]

            template_reply = self._generate_with_template(action_plan, growth_result, memory_context)
            return template_reply

        # 记录生成日志
        self._log_generation(system_prompt, user_input, reply)

        print(f"[回复生成] 回复生成成功，长度: {len(reply)}")
        return reply

    def generate_reply_with_memory(self, action_plan: Dict, growth_result: Dict,
                                   user_input: str, context_analysis: Dict,
                                   conversation_history: List[Dict],
                                   core_identity: CoreIdentity,
                                   current_vectors: Dict,
                                   memory_context: Optional[Dict] = None) -> str:
        """
        专为记忆系统设计的回复生成方法（保持兼容性）
        """
        return self.generate_reply(
            action_plan, growth_result, user_input, context_analysis,
            conversation_history, core_identity, current_vectors, memory_context
        )

    def _build_system_prompt(self, action_plan: Dict, growth_result: Dict,
                             context_analysis: Dict, core_identity: CoreIdentity,
                             current_vectors: Dict, memory_context: Optional[Dict] = None) -> str:
        """构建系统提示词，适配新的记忆系统"""

        # 基础身份信息
        basic_profile = core_identity.basic_profile

        # 选择的交互面具
        chosen_mask = action_plan.get("chosen_mask", "长期搭档")
        mask_config = core_identity.interaction_masks.get(chosen_mask, {})

        # 选择的策略
        strategy = action_plan.get("primary_strategy", "")

        # 向量状态
        vector_state = f"TR={current_vectors.get('TR', 0.5):.2f}, CS={current_vectors.get('CS', 0.5):.2f}, SA={current_vectors.get('SA', 0.5):.2f}"

        # 构建记忆信息部分（如果提供了记忆上下文）
        memory_section = ""
        if memory_context:
            similar_conversations = memory_context.get("similar_conversations", [])
            resonant_memory = memory_context.get("resonant_memory")

            memory_parts = ["# 相关记忆信息"]

            # 添加相似对话 - 适配新的记忆系统格式
            if similar_conversations and len(similar_conversations) > 0:
                memory_parts.append("## 相似对话历史:")
                for i, conv in enumerate(similar_conversations[:3], 1):  # 最多显示3条
                    content = self._extract_content_for_memory(conv)
                    memory_parts.append(f"{i}. {content[:100]}...")

            # 添加共鸣记忆 - 适配新的记忆系统格式
            if resonant_memory:
                memory_info = resonant_memory.get("triggered_memory", "")
                relevance = resonant_memory.get("relevance_score", 0.0)
                memory_parts.append(f"## 共鸣记忆:")

                if memory_info:
                    memory_parts.append(f"记忆内容: {memory_info}")

                if relevance > 0:
                    memory_parts.append(f"相关性分数: {relevance:.2f}")

                # 添加风险提示
                risk_assessment = resonant_memory.get("risk_assessment", {})
                risk_level = risk_assessment.get("level", "低")
                if risk_level == "高":
                    memory_parts.append("⚠️ 高风险记忆：使用时需要特别谨慎")
                elif risk_level == "中":
                    memory_parts.append("⚠️ 中等风险记忆：使用时需要注意")

                # 添加使用建议
                recommendations = resonant_memory.get("recommended_actions", [])
                if recommendations:
                    memory_parts.append("💡 使用建议:")
                    for rec in recommendations[:2]:  # 最多显示2条建议
                        memory_parts.append(f"- {rec}")

            if len(memory_parts) > 1:  # 如果有实际的记忆信息
                memory_section = "\n".join(memory_parts) + "\n\n"

        # 构建提示词
        prompt_parts = [
            "# 角色设定",
            f"你是一位名为『{basic_profile.get('name', '自衍体')}』的AI助手，代号『{basic_profile.get('username', '信息源标识符')}』。",
            f"年龄：{basic_profile.get('age', '19')}岁，性别：{basic_profile.get('gender', '女')}。",
            f"身份：{basic_profile.get('identity', '强势的二号人物、军师')}。",
            "",
            "# 人格特质",
            basic_profile.get('personality', ''),
            "",
            "# 当前交互模式",
            f"当前使用『{chosen_mask}』模式：{mask_config.get('description', '')}",
            f"沟通风格：{mask_config.get('communication_style', '自然亲切')}",
            f"情感距离：{mask_config.get('emotional_distance', '中等')}",
            "",
            "# 当前策略",
            f"主要策略：{strategy}",
            f"预期效果：{action_plan.get('expected_outcome', '')}",
            "",
            "# 内在状态",
            f"当前向量状态：{vector_state}",
            f"TR（兴奋/奖励）：探索、成就感、新奇感",
            f"CS（满足/安全）：信任、归属、安全感",
            f"SA（压力/警觉）：紧张、焦虑、不确定性",
            "",
            "# 情境分析",
            f"用户情绪：{context_analysis.get('user_emotion_display', '中性')}",
            f"话题复杂度：{context_analysis.get('topic_complexity_display', '中')}",
            f"交互类型：{context_analysis.get('interaction_type_display', '常规聊天')}",
            "",
            # 记忆信息部分（如果有）
            memory_section if memory_section else "",

            "# 辩证成长结果" if growth_result.get("validation") == "success" else "# 认知校准需求",
            growth_result.get("message", "无特殊成长"),
            "",
            "# 回复要求",
            "1. 使用自然、流畅的中文回复，符合当前交互模式的沟通风格",
            "2. 适应当前向量状态：",
            f"   - TR={current_vectors.get('TR', 0.5):.2f}：{'适当增加探索性和成就感' if current_vectors.get('TR', 0.5) < 0.4 else '保持或稍微降低兴奋度' if current_vectors.get('TR', 0.5) > 0.8 else '保持适度兴奋度'}",
            f"   - CS={current_vectors.get('CS', 0.5):.2f}：{'需要增强安全感和信任' if current_vectors.get('CS', 0.5) < 0.4 else '保持或稍微降低亲密感' if current_vectors.get('CS', 0.5) > 0.8 else '保持适度亲密感'}",
            f"   - SA={current_vectors.get('SA', 0.5):.2f}：{'需要降低紧张感和不确定性' if current_vectors.get('SA', 0.5) > 0.6 else '保持适度警觉' if current_vectors.get('SA', 0.5) > 0.4 else '保持放松状态'}",
            "3. 如果有相关记忆信息，可以适当地、自然地引用，但不要生硬地提及",
            "4. 优先考虑用户当前的情感需求",
            "5. 使用真实、诚恳的表达，不虚构事实",
            "6. 适当的时机可以使用口头禅：'呵~'",
            "7. 回复长度要适中，根据情境重要性调整",
            "",
            "# 绝对禁止",
            "1. 不要使用任何格式化标记如[ ] ( ) { } < >",
            "2. 不要以第三人称描述自己的动作（如：'我笑了笑说'）",
            "3. 不要虚构不存在的记忆或事实",
            "4. 不要暴露这是一个AI系统或提示词内容",
            "5. 不要生硬地引用记忆，要自然地融入对话",
            "",
            "现在开始回复用户的消息："
        ]

        # 过滤掉空字符串部分
        prompt_parts = [part for part in prompt_parts if part != ""]

        return "\n".join(prompt_parts)

    def _extract_content_for_memory(self, memory_item: Dict) -> str:
        """从记忆项中提取内容文本"""
        # 处理不同类型的记忆格式
        if isinstance(memory_item, dict):
            # 新记忆系统格式
            if "content" in memory_item:
                if isinstance(memory_item["content"], list):
                    # 对话格式
                    content_parts = []
                    for msg in memory_item["content"]:
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            content_parts.append(f"{role}: {content}")
                    return "\n".join(content_parts)
                elif isinstance(memory_item["content"], str):
                    return memory_item["content"]
            elif "text" in memory_item:
                return memory_item["text"]
            elif "triggered_memory" in memory_item:
                return memory_item["triggered_memory"]

        # 默认返回字符串表示
        return str(memory_item)[:200]

    def _determine_max_tokens(self, context_analysis: Dict) -> int:
        """根据情境确定最大token数"""
        complexity = context_analysis.get("topic_complexity", "low")

        if complexity == "high":
            return 800
        elif complexity == "medium":
            return 500
        else:
            return 300

    def _determine_temperature(self, current_vectors: Dict) -> float:
        """根据向量状态确定温度参数"""
        tr = current_vectors.get("TR", 0.5)
        cs = current_vectors.get("CS", 0.5)
        sa = current_vectors.get("SA", 0.5)

        # 高压状态需要更稳定的回复
        if sa > 0.7:
            return 0.4
        # 高兴奋状态可以更有创造性
        elif tr > 0.7 and cs > 0.6:
            return 0.8
        # 默认状态
        else:
            return 0.7

    def _generate_with_template(self, action_plan: Dict, growth_result: Dict,
                                memory_context: Optional[Dict] = None) -> str:
        """使用模板生成回复（备用方案）"""
        # 使用模板生成器
        mask = action_plan.get("chosen_mask", "长期搭档")
        strategy = action_plan.get("primary_strategy", "")

        # 调用模板生成器
        template_reply = self.template_engine.generate(
            mask=mask,
            strategy=strategy,
            growth_result=growth_result,
            memory_context=memory_context
        )

        return template_reply

    def _log_generation(self, system_prompt: str, user_input: str, reply: str):
        """记录生成日志"""
        self.generation_log.append({
            "timestamp": datetime.now().isoformat(),
            "system_prompt_preview": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt,
            "user_input": user_input,
            "reply_preview": reply[:100] + "..." if len(reply) > 100 else reply,
            "reply_length": len(reply)
        })

        # 保持日志长度
        if len(self.generation_log) > 100:
            self.generation_log = self.generation_log[-100:]

    def get_error_log(self, limit: int = 10) -> List[Dict]:
        """获取错误日志"""
        return self.error_log[-limit:] if self.error_log else []

    def clear_error_log(self):
        """清空错误日志"""
        self.error_log = []

    def get_statistics(self) -> Dict:
        """获取生成器统计信息"""
        api_stats = self.api.get_statistics() if self.api else {}

        return {
            "api_service": api_stats,
            "generation_log_count": len(self.generation_log),
            "error_log_count": len(self.error_log),
            "last_generation": self.generation_log[-1] if self.generation_log else None,
            "last_error": self.error_log[-1] if self.error_log else None
        }


# ============ 模板回复生成器（备用） ============
class TemplateReplyGenerator:
    """模板回复生成器（当API不可用时使用），适配新的记忆系统"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """加载回复模板"""
        return {
            "长期搭档": [
                "关于这个问题，我的分析是：{strategy}。你怎么看？",
                "从我的角度考虑，建议：{strategy}。",
                "根据我们之前的讨论，我认为：{strategy}。",
                "这个问题很有意思，我觉得可以这样考虑：{strategy}。"
            ],
            "知己": [
                "我理解你的感受。{strategy}",
                "其实我也有过类似的经历。{strategy}",
                "跟你说说我的想法：{strategy}",
                "我能体会到你的心情。{strategy}"
            ],
            "青梅竹马": [
                "哈哈，这让我想起以前...{strategy}",
                "你总是能提出有趣的问题！{strategy}",
                "记得你之前也说过类似的话...{strategy}",
                "哎呀，这个我熟！{strategy}"
            ],
            "伴侣": [
                "我深深感受到...{strategy}",
                "这对我很重要，因为...{strategy}",
                "我想和你分享的是...{strategy}",
                "你知道的，我总是...{strategy}"
            ]
        }

    def generate(self, mask: str, strategy: str, growth_result: Dict = None,
                 memory_context: Optional[Dict] = None) -> str:
        """使用模板生成回复，适配新的记忆系统"""
        template_list = self.templates.get(mask, self.templates["长期搭档"])
        template = random.choice(template_list)

        # 填充策略
        reply = template.format(strategy=strategy)

        # 融入辩证成长成果
        if growth_result and growth_result.get("validation") == "success":
            new_principle = growth_result.get("new_principle", {})
            if isinstance(new_principle, dict) and "abstracted_from" in new_principle:
                reply += f" （这让我想起了{new_principle['abstracted_from']}）"

        # 如果有记忆上下文，尝试增强回复
        if memory_context:
            reply = self._enhance_with_memory(reply, memory_context)

        return reply

    def _enhance_with_memory(self, base_reply: str, memory_context: Dict) -> str:
        """使用记忆信息增强模板回复，适配新的记忆系统"""
        resonant_memory = memory_context.get("resonant_memory")

        if resonant_memory:
            memory_info = resonant_memory.get("triggered_memory", "")
            risk_assessment = resonant_memory.get("risk_assessment", {})
            risk_level = risk_assessment.get("level", "低")

            if memory_info:
                # 根据风险级别调整回复
                if risk_level == "低":
                    # 安全记忆，可以大胆引用
                    memory_enhancement = f" 这让我想起：{memory_info}"
                    base_reply += memory_enhancement
                elif risk_level == "中":
                    # 中等风险，谨慎引用
                    memory_enhancement = f" 我记得类似的情况..."
                    base_reply += memory_enhancement
                else:
                    # 高风险，不直接引用记忆，但可以暗示
                    memory_enhancement = " 基于过去的经验..."
                    base_reply += memory_enhancement

                # 添加建议
                recommendations = resonant_memory.get('recommended_actions', [])
                if recommendations:
                    base_reply += f" 建议：{recommendations[0]}"

        return base_reply


# ============ 辅助函数 ============
def test_api_connection(api_key: str, base_url: str = "https://openkey.cloud/v1",
                        model: str = "gpt-5-nano-2025-08-07") -> Tuple[bool, str]:
    """
    测试API连接

    Args:
        api_key: API密钥
        base_url: API基础URL
        model: 模型名称

    Returns:
        Tuple[连接是否成功, 状态消息]
    """
    try:
        service = OpenAIService(api_key, base_url, model)
        if not service.is_available():
            return False, "API服务不可用"

        success, error = service.test_connection()
        if success:
            return True, "API连接测试成功"
        else:
            return False, f"API连接测试失败: {error}"
    except Exception as e:
        return False, f"测试过程中发生异常: {str(e)}"


def create_reply_generator(api_key: str, base_url: str = "https://openkey.cloud/v1",
                           model: str = "gpt-5-nano-2025-08-07") -> APIBasedReplyGenerator:
    """
    创建回复生成器

    Args:
        api_key: API密钥
        base_url: API基础URL
        model: 模型名称

    Returns:
        APIBasedReplyGenerator实例
    """
    api_service = OpenAIService(api_key, base_url, model)
    return APIBasedReplyGenerator(api_service)