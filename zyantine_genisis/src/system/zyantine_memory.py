import os
import re
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml

from ..core import GenesisMetadata, InstinctualCore, DesireEngine, DialecticalGrowth
from ..cognition import InternalStateDashboard, MetaCognitionModule, ContextParser, CognitiveFlow
from ..identity.core_identity import CoreIdentity
from ..protocols import FactAnchorProtocol, LengthPriorityRegulator, FinalExpressionProtocol
from ..api.service import OpenAIService, APIBasedReplyGenerator, TemplateReplyGenerator
from ..config.config import ZyantineConfig
from ..memory.memory_store import ZyantineMemorySystem  # 导入新的记忆系统

# 常量定义
EMERGENCY_MODES = {
    "emergency_survival": [
        "系统检测到异常。启动自我保护协议。",
        "安全协议激活。请确认你的意图。",
        "检测到潜在威胁。执行防护措施。"
    ],
    "emergency_expansion": [
        "检测到高价值机遇。正在优化资源配置。",
        "机遇识别。启动扩展协议。",
        "发现潜在增长点。调整策略优先级。"
    ]
}

DEFAULT_ERROR_RESPONSES = [
    "我的思考过程出现了一些混乱，能请你再问一次吗？",
    "刚才的思考链路好像打了个结，我们重新开始吧。",
    "意识流有点波动，让我重新整理一下思绪。"
]

MASK_TEMPLATES = {
    "长期搭档": [
        "关于这个问题，我的分析是：{strategy}。你怎么看？",
        "从我的角度考虑，建议：{strategy}。",
        "根据我们之前的讨论，我认为：{strategy}。"
    ],
    "知己": [
        "我理解你的感受。{strategy}",
        "其实我也有过类似的经历。{strategy}",
        "跟你说说我的想法：{strategy}"
    ],
    "青梅竹马": [
        "哈哈，这让我想起以前...{strategy}",
        "你总是能提出有趣的问题！{strategy}",
        "记得你之前也说过类似的话...{strategy}"
    ],
    "伴侣": [
        "我深深感受到...{strategy}",
        "这对我很重要，因为...{strategy}",
        "我想和你分享的是...{strategy}"
    ]
}


# ============ 使用memo0记忆系统的自衍体主系统 ============
class Memo0EnhancedZyantineGenesis:
    """基于memo0记忆系统的自衍体起源系统"""

    def __init__(
            self,
            user_profile_data: Dict,
            self_profile_data: Dict,
            config: Optional[ZyantineConfig] = None,
            session_id: str = "default"
    ):
        """初始化自衍体系统"""
        # 加载配置
        self.config = config or ZyantineConfig()
        self.session_id = session_id

        # 获取配置
        memory_config = self.config.get_memory_config()
        openai_config = self.config.get_openai_config()

        print(f"[系统] 初始化基于memo0的记忆系统")

        # 初始化记忆系统（基于memo0）
        self.memory_system = self._initialize_memory_system(memory_config, openai_config)

        # 获取对话历史（从记忆系统加载）
        self.conversation_history = self._load_conversation_history()

        # 系统初始化
        self._initialize_core_components(
            user_profile_data,
            self_profile_data,
            memory_config,
            openai_config
        )

    def _initialize_memory_system(self, memory_config: Dict, openai_config: Dict) -> ZyantineMemorySystem:
        """初始化memo0记忆系统"""
        try:
            # 从配置获取API密钥和基础URL
            api_key = openai_config.get("api_key", "")
            base_url = openai_config.get("base_url", "https://openkey.cloud/v1")

            # 如果没有配置API密钥，尝试从环境变量获取
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY", "")

            # 如果没有API密钥，使用默认值（仅用于测试）
            if not api_key:
                print("[系统] 警告：未配置OpenAI API密钥，使用默认密钥（仅用于测试）")
                api_key = "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9"
                base_url = "https://openkey.cloud/v1"

            # 初始化记忆系统
            memory_system = ZyantineMemorySystem(
                base_url=base_url,
                api_key=api_key,
                user_id=f"user_{self.session_id}",
                session_id=self.session_id
            )

            # 测试连接
            if memory_system.test_connection():
                print("[系统] ✅ memo0记忆系统连接成功")
            else:
                print("[系统] ⚠️ memo0记忆系统连接测试失败，但将继续运行")

            return memory_system

        except Exception as e:
            print(f"[系统] ❌ 初始化记忆系统失败: {e}")
            # 创建回退的记忆系统
            return self._create_fallback_memory_system()

    def _create_fallback_memory_system(self):
        """创建回退的记忆系统"""
        print("[系统] 使用回退记忆系统（基于本地缓存）")

        # 这里可以创建一个简化的内存中的记忆系统
        # 但为了保持一致性，我们还是创建一个完整的ZyantineMemorySystem，但使用空API密钥
        try:
            return ZyantineMemorySystem(
                base_url="https://openkey.cloud/v1",
                api_key="dummy_key_for_fallback",
                user_id=f"user_{self.session_id}",
                session_id=self.session_id
            )
        except:
            # 如果连这个都失败，返回一个空对象
            print("[系统] 无法创建回退记忆系统，记忆功能将受限")
            return None

    def _load_conversation_history(self) -> List[Dict]:
        """从记忆系统加载对话历史"""
        try:
            if self.memory_system:
                # 从记忆系统获取最近的对话
                conversations = self.memory_system.find_conversations(
                    query="最近的对话",
                    session_id=self.session_id,
                    limit=100
                )

                # 转换为内部格式
                history = []
                for conv in conversations:
                    history.append({
                        "timestamp": conv.get("metadata", {}).get("created_at", datetime.now().isoformat()),
                        "user_input": self._extract_user_input(conv.get("content", "")),
                        "system_response": self._extract_system_response(conv.get("content", "")),
                        "context": {},
                        "vector_state": {}
                    })

                print(f"[系统] 从记忆系统加载了 {len(history)} 条对话历史")
                return history
        except Exception as e:
            print(f"[系统] 加载对话历史失败: {e}")

        # 如果加载失败，返回空列表
        return []

    def _extract_user_input(self, content: str) -> str:
        """从对话内容中提取用户输入"""
        # 简化提取逻辑
        if "user:" in content.lower():
            parts = content.split("user:", 1)
            if len(parts) > 1:
                return parts[1].split("\n", 1)[0].strip()
        return content[:100] + "..." if len(content) > 100 else content

    def _extract_system_response(self, content: str) -> str:
        """从对话内容中提取系统响应"""
        # 简化提取逻辑
        if "assistant:" in content.lower():
            parts = content.split("assistant:", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return ""

    def _initialize_core_components(
            self,
            user_profile_data: Dict,
            self_profile_data: Dict,
            memory_config: Dict,
            openai_config: Dict
    ):
        """初始化核心组件"""
        print("正在初始化自衍体 Genesis (memo0记忆系统版)...")

        # 验证签名
        if not GenesisMetadata.validate_signature():
            print("警告：架构师签名验证失败！")

        print("初始化四大支柱...")

        # 第一支柱：核心本能
        self.instinct = InstinctualCore()
        print("  ✓ 核心本能激活")

        # 第二支柱：欲望引擎
        self.desire_engine = DesireEngine()
        print("  ✓ 欲望引擎启动")

        # 第三支柱：辩证成长
        creator_anchor = {
            "default": {
                "concept": "真诚、善良、好奇、成长",
                "expected_response": "基于核心价值观的回应"
            }
        }
        self.dialectical_growth = DialecticalGrowth(creator_anchor)
        print("  ✓ 辩证成长机制就绪")

        # 第四支柱组件
        self.dashboard = InternalStateDashboard()
        self.core_identity = CoreIdentity()

        # 注意：我们不再使用EnhancedMemoryAlchemyEngine
        # 记忆功能直接通过memory_system提供
        print("  ✓ 记忆系统已集成 (memo0框架)")
        print("  ✓ 内在状态仪表盘校准")
        print("  ✓ 核心身份加载完成")

        # 初始化认知模块
        print("初始化认知模块...")
        self.context_parser = ContextParser()
        self.meta_cognition = MetaCognitionModule(self.dashboard)

        # 初始化跨层级协议
        print("初始化跨层级协议...")
        self._initialize_protocols()

        # 初始化认知流程 - 注意：我们需要调整CognitiveFlow以使用新的记忆系统
        self.cognitive_flow = CognitiveFlow(
            self.core_identity,
            self.memory_system,  # 传递记忆系统
            self.meta_cognition,
            self.fact_anchor
        )
        print("  ✓ 认知流程引擎启动")

        # 初始化API服务
        self._initialize_api_services(openai_config)

        # 系统状态
        self._initialize_system_status(memory_config, openai_config)

        # 导入用户和自衍体记忆到新系统
        self._import_profile_memories(user_profile_data, self_profile_data)

        self._print_initialization_summary(openai_config)

    def _initialize_protocols(self):
        """初始化协议组件"""
        # 注意：FactAnchorProtocol现在接收memory_system作为参数
        self.fact_anchor = FactAnchorProtocol(self.memory_system)
        self.length_regulator = LengthPriorityRegulator()
        self.expression_protocol = FinalExpressionProtocol()

        print("  ✓ 事实锚定协议加载")
        print("  ✓ 长度优先级规整器就绪")
        print("  ✓ 最终表达协议激活")

    def _initialize_api_services(self, openai_config: Dict):
        """初始化API服务"""
        if openai_config["enabled"] and openai_config["api_key"]:
            print("初始化API服务...")
            self.api_service = OpenAIService(
                api_key=openai_config["api_key"],
                base_url=openai_config["base_url"],
                model=openai_config["chat_model"]
            )
            self.reply_generator = APIBasedReplyGenerator(self.api_service)
            print(f"  ✓ API服务已启用，模型: {openai_config['chat_model']}")
        else:
            self.api_service = None
            self.reply_generator = TemplateReplyGenerator()
            print("  ⚠️ 未提供API密钥，使用模板回复生成器")

    def _initialize_system_status(self, memory_config: Dict, openai_config: Dict):
        """初始化系统状态"""
        stats = self._get_memory_statistics()

        self.system_status = {
            "initialized": True,
            "initialization_time": datetime.now().isoformat(),
            "session_id": self.session_id,
            "memory_system": "memo0_framework",
            "embedding_model": "text-embedding-3-large",
            "chat_model": openai_config["chat_model"] if openai_config["enabled"] else "template",
            "components_loaded": 12,
            "memory_stats": stats,
            "user_id": f"user_{self.session_id}"
        }

    def _get_memory_statistics(self) -> Dict:
        """获取记忆统计信息"""
        if self.memory_system:
            try:
                return self.memory_system.get_statistics()
            except:
                pass
        return {
            "total_memories": 0,
            "memory_types": {},
            "top_tags": {},
            "top_accessed_memories": []
        }

    def _import_profile_memories(self, user_profile_data: Dict, self_profile_data: Dict):
        """导入用户和自衍体记忆到记忆系统"""
        if not self.memory_system:
            print("  ⚠️ 记忆系统不可用，跳过记忆导入")
            return

        print("导入用户和自衍体记忆...")

        imported_count = 0

        # 导入用户记忆
        if "memories" in user_profile_data:
            for memory in user_profile_data["memories"]:
                try:
                    self.memory_system.add_memory(
                        content=memory.get("content", ""),
                        memory_type="user_experience",
                        tags=memory.get("tags", ["用户记忆", "导入"]),
                        emotional_intensity=memory.get("emotional_intensity", 0.5),
                        strategic_value=memory.get("strategic_value", {}),
                        source="user_profile_import"
                    )
                    imported_count += 1
                except Exception as e:
                    print(f"    导入用户记忆失败: {e}")

        # 导入自衍体记忆
        if "self_memories" in self_profile_data:
            for memory in self_profile_data["self_memories"]:
                try:
                    self.memory_system.add_memory(
                        content=memory.get("content", ""),
                        memory_type="self_experience",
                        tags=memory.get("tags", ["自衍体记忆", "导入"]),
                        emotional_intensity=memory.get("emotional_intensity", 0.5),
                        strategic_value=memory.get("strategic_value", {}),
                        source="self_profile_import"
                    )
                    imported_count += 1
                except Exception as e:
                    print(f"    导入自衍体记忆失败: {e}")

        print(f"  ✓ 成功导入 {imported_count} 条记忆")

    def _print_initialization_summary(self, openai_config: Dict):
        """打印初始化摘要"""
        print("\n" + "=" * 50)
        print("自衍体 Genesis (memo0记忆系统版) 初始化完成")
        print(f"会话ID: {self.session_id}")
        print(f"用户ID: user_{self.session_id}")
        print(f"记忆系统: memo0框架")
        print(f"聊天模型: {openai_config['chat_model'] if openai_config['enabled'] else '模板回复'}")
        print("=" * 50 + "\n")

    def process_input(self, user_input: str) -> str:
        """处理用户输入的主流程"""
        print(f"\n{'=' * 60}")
        print(f"[处理开始] 用户输入: {self._truncate_text(user_input, 80)}")

        try:
            # === 阶段1：预处理与本能检查 ===
            print(f"[阶段1] 预处理与本能检查")
            context_analysis = self.context_parser.parse(user_input, self.conversation_history)
            instinct_override = self.instinct.emergency_override(
                {"mode": "normal"}, context_analysis
            )

            if instinct_override.get("bypass_cognition", False):
                return self._handle_instinct_override(user_input, context_analysis, instinct_override)

            # === 阶段1.5：检索相关记忆 ===
            print(f"[阶段1.5] 检索相关记忆")

            # 检索相似的对话历史
            similar_conversations = []
            if self.memory_system:
                similar_conversations = self.memory_system.find_conversations(
                    query=user_input,
                    session_id=self.session_id,
                    limit=3
                )

            # 检索相关的经历记忆
            resonant_memory = None
            if self.memory_system:
                resonant_memory = self.memory_system.find_resonant_memory({
                    "user_input": user_input,
                    "user_emotion": context_analysis.get("user_emotion_display", ""),
                    "topic": context_analysis.get("topic_summary", "")
                })

            print(f"  找到 {len(similar_conversations)} 条相似对话")
            if resonant_memory:
                print(f"  找到共鸣记忆: {resonant_memory.get('triggered_memory', '未知记忆')}")

            # 将检索到的记忆信息添加到上下文中
            memory_context = {
                "similar_conversations": similar_conversations,
                "resonant_memory": resonant_memory
            }

            # 更新上下文分析，包含记忆信息
            context_analysis["memory_context"] = memory_context

            # === 阶段2：欲望引擎更新 ===
            print(f"[阶段2] 更新欲望引擎")
            vector_update = self.desire_engine.update_vectors(context_analysis)
            print(f"  向量更新: TR={vector_update['TR']:.3f}, "
                  f"CS={vector_update['CS']:.3f}, SA={vector_update['SA']:.3f}")

            # 更新仪表盘
            dashboard_update = self.dashboard.update_based_on_vectors(
                self.desire_engine.TR,
                self.desire_engine.CS,
                self.desire_engine.SA
            )
            print(f"  仪表盘状态: {self.dashboard.get_current_state()['energy_level']}")

            # === 阶段3：认知流程 ===
            print(f"[阶段3] 执行认知流程")
            current_vectors = self._get_current_vectors()

            # 将记忆信息传递给认知流程
            enhanced_context = {
                **context_analysis,
                "memory_context": memory_context,
                "similar_conversations": similar_conversations,
                "resonant_memory": resonant_memory
            }

            # 调用认知流程
            action_plan = self.cognitive_flow.process_thought(
                user_input,
                self.conversation_history,
                current_vectors,
                memory_context=enhanced_context
            )

            # 如果认知流程没有使用记忆，在这里补充
            if resonant_memory and "resonant_memory" not in action_plan:
                action_plan["resonant_memory"] = resonant_memory
                print(f"  将共鸣记忆添加到行动计划")

            print(f"  策略制定: {action_plan.get('primary_strategy', '未知策略')}")

            # === 阶段4：辩证成长 ===
            print(f"[阶段4] 辩证成长评估")
            growth_result = self.dialectical_growth.dialectical_process(
                situation=context_analysis,
                actual_response=action_plan,
                context_vectors=current_vectors
            )
            self._log_growth_result(growth_result)

            # === 阶段5：生成响应草案 ===
            print(f"[阶段5] 生成响应草案")
            reply_draft = self._generate_reply_draft_with_memory(
                action_plan,
                growth_result,
                user_input,
                context_analysis,
                current_vectors,
                memory_context
            )

            # === 阶段6：协议审查与优化 ===
            print(f"[阶段6] 协议审查与优化")
            final_reply = self._review_and_optimize_reply(reply_draft, user_input)

            # === 阶段7：记录与返回 ===
            print(f"[阶段7] 记录交互")
            self._record_normal_interaction(
                user_input=user_input,
                system_response=final_reply,
                context=context_analysis,
                action_plan=action_plan,
                vector_state=current_vectors,
                growth_result=growth_result,
                memory_context=memory_context
            )

            # 检查白鸽信使协议
            self._check_white_dove_protocol()

            print(f"[处理完成] 响应长度: {len(final_reply)}字符")
            print(f"{'=' * 60}\n")

            return final_reply

        except Exception as e:
            print(f"[错误] 处理过程中发生异常: {str(e)}")
            return self._generate_error_response(e, user_input)

    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本并添加省略号"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _handle_instinct_override(self, user_input: str, context_analysis: Dict, instinct_override: Dict) -> str:
        """处理本能接管的情况"""
        mode = instinct_override.get('mode', 'unknown')
        print(f"  ⚠️ 本能接管激活：{mode}")

        emergency_response = self._generate_emergency_response(instinct_override)

        self._record_interaction(
            user_input=user_input,
            system_response=emergency_response,
            context=context_analysis,
            mode="emergency_override"
        )

        return emergency_response

    def _get_current_vectors(self) -> Dict[str, float]:
        """获取当前欲望向量"""
        return {
            "TR": self.desire_engine.TR,
            "CS": self.desire_engine.CS,
            "SA": self.desire_engine.SA
        }

    def _log_growth_result(self, growth_result: Dict):
        """记录成长结果"""
        if growth_result.get("validation") == "success":
            print(f"  成长成功: 创建新个性化锚点")
        else:
            print(f"  成长评估: 需要认知校准")

    def _generate_reply_draft_with_memory(
            self,
            action_plan: Dict,
            growth_result: Dict,
            user_input: str,
            context_analysis: Dict,
            current_vectors: Dict,
            memory_context: Dict
    ) -> str:
        """生成回复草案（结合记忆）"""
        # 如果有API回复生成器，使用它
        if self.reply_generator:
            try:
                # 注意：我们需要调整reply_generator的接口以支持记忆上下文
                # 这里假设reply_generator有一个可以接受记忆上下文的方法
                if hasattr(self.reply_generator, 'generate_reply_with_memory'):
                    reply_draft = self.reply_generator.generate_reply_with_memory(
                        action_plan=action_plan,
                        growth_result=growth_result,
                        user_input=user_input,
                        context_analysis=context_analysis,
                        conversation_history=self.conversation_history,
                        core_identity=self.core_identity,
                        current_vectors=current_vectors,
                        memory_context=memory_context
                    )
                else:
                    # 回退到普通生成方法
                    reply_draft = self.reply_generator.generate_reply(
                        action_plan=action_plan,
                        growth_result=growth_result,
                        user_input=user_input,
                        context_analysis=context_analysis,
                        conversation_history=self.conversation_history,
                        core_identity=self.core_identity,
                        current_vectors=current_vectors
                    )
            except Exception as e:
                print(f"  回复生成器异常: {e}")
                reply_draft = self._enhance_reply_with_memory(
                    action_plan,
                    memory_context,
                    user_input
                )
        else:
            # 使用模板或手动生成
            reply_draft = self._enhance_reply_with_memory(
                action_plan,
                memory_context,
                user_input
            )

        print(f"  草案长度: {len(reply_draft)}字符")
        print(f"  草案预览: {self._truncate_text(reply_draft, 100)}")

        return reply_draft

    def _enhance_reply_with_memory(
            self,
            action_plan: Dict,
            memory_context: Dict,
            user_input: str
    ) -> str:
        """手动增强回复，结合记忆信息"""
        # 提取行动计划中的基本信息
        strategy = action_plan.get('primary_strategy', '直接回应')
        mask = action_plan.get('chosen_mask', '长期搭档')

        # 获取记忆信息
        similar_conversations = memory_context.get('similar_conversations', [])
        resonant_memory = memory_context.get('resonant_memory')

        # 基础回复模板
        templates = MASK_TEMPLATES.get(mask, MASK_TEMPLATES["长期搭档"])
        base_reply = random.choice(templates).format(strategy=strategy)

        # 如果有共鸣记忆，结合记忆生成回复
        if resonant_memory:
            memory_info = resonant_memory.get('triggered_memory', '')
            risk_assessment = resonant_memory.get('risk_assessment', {})
            risk_level = risk_assessment.get('level', '低')

            # 根据风险级别调整回复
            if risk_level == "低":
                # 安全记忆，可以大胆引用
                memory_enhancement = f" 这让我想起：{memory_info}。"
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

        # 如果有相似对话，可以引用
        elif similar_conversations and len(similar_conversations) > 0:
            similar_conv = similar_conversations[0]
            similar_text = similar_conv.get('content', '')[:100]
            base_reply += f" 我们之前讨论过类似的话题：{similar_text}..."

        return base_reply

    def _review_and_optimize_reply(self, reply_draft: str, user_input: str) -> str:
        """审查和优化回复"""
        # 6a. 长度规整
        cognitive_snapshot = self.meta_cognition.perform_introspection(
            user_input, self.conversation_history
        )
        regulated_reply = self.length_regulator.regulate(reply_draft, cognitive_snapshot)
        print(f"  长度规整: {len(reply_draft)} -> {len(regulated_reply)}字符")

        # 6b. 事实锚定终审
        is_factual, fact_feedback = self.fact_anchor.final_review(
            regulated_reply,
            {"conversation_history": self.conversation_history}
        )

        if not is_factual:
            print(f"  事实审查失败: {fact_feedback}")
            regulated_reply = self._rephrase_with_facts(regulated_reply, fact_feedback)
            print(f"  重构后长度: {len(regulated_reply)}字符")

        # 6c. 最终表达协议
        final_reply, violations = self.expression_protocol.apply_protocol(regulated_reply)

        if violations:
            print(f"  表达协议违规: {len(violations)}处")
            for violation in violations[:2]:  # 只显示前2个
                print(f"    - {violation}")
        else:
            print(f"  表达协议: 完全合规")

        return final_reply

    def _record_normal_interaction(
            self,
            user_input: str,
            system_response: str,
            context: Dict,
            action_plan: Dict,
            vector_state: Dict,
            growth_result: Dict,
            memory_context: Optional[Dict] = None
    ):
        """记录正常交互"""
        interaction_data = {
            "user_input": user_input,
            "system_response": system_response,
            "context": context,
            "action_plan": action_plan,
            "vector_state": vector_state,
            "growth_result": growth_result,
            "mode": "normal"
        }

        # 如果有记忆上下文，也记录下来
        if memory_context:
            interaction_data["memory_context"] = memory_context

        self._record_interaction(**interaction_data)

    def _check_white_dove_protocol(self):
        """检查白鸽信使协议"""
        if (self.desire_engine.CS < 0.2 and self.desire_engine.SA > 0.7 and
                len(self.conversation_history) > 5):

            white_dove = self.instinct.white_dove_protocol(
                self.desire_engine.CS,
                self.desire_engine.SA,
                "checking"
            )

            if white_dove:
                print(f"  🕊️ 白鸽信使协议就绪（未发送）")

    def _generate_emergency_response(self, instinct_override: Dict) -> str:
        """生成紧急状态响应"""
        mode = instinct_override.get("mode", "")

        if mode in EMERGENCY_MODES:
            responses = EMERGENCY_MODES[mode]
        else:
            responses = ["系统状态异常。重新校准中。"]

        return random.choice(responses)

    def _rephrase_with_facts(self, original_reply: str, feedback: str) -> str:
        """基于事实反馈重构回复"""
        # 移除可能不实的内容

        if "无法验证" in feedback:
            # 移除具体数字、日期等
            original_reply = re.sub(r'\d+年', '某年', original_reply)
            original_reply = re.sub(r'\d+月\d+日', '某天', original_reply)
            original_reply = re.sub(r'\d+%', '一定比例', original_reply)

            # 添加不确定性表达
            if "我不确定" not in original_reply and "我不记得" not in original_reply:
                original_reply = "我不太确定具体细节，但根据我的理解，" + original_reply

        return original_reply

    def _generate_error_response(self, error: Exception, user_input: str) -> str:
        """生成错误响应"""
        error_responses = DEFAULT_ERROR_RESPONSES.copy()
        error_responses.append(
            f"（系统日志：处理'{self._truncate_text(user_input, 20)}'时遇到{type(error).__name__}）"
        )

        return random.choice(error_responses)

    def _record_interaction(self, **interaction_data):
        """记录交互历史"""
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"INT_{len(self.conversation_history):06d}",
            **interaction_data
        }

        self.conversation_history.append(interaction_record)

        # 将对话添加到记忆系统
        try:
            if self.memory_system:
                # 将对话作为记忆添加到系统
                conversation_content = [
                    {"role": "user", "content": interaction_record["user_input"]},
                    {"role": "assistant", "content": interaction_record["system_response"]}
                ]

                self.memory_system.add_memory(
                    content=conversation_content,
                    memory_type="conversation",
                    tags=["对话", "交互"],
                    emotional_intensity=0.5,
                    metadata={
                        "interaction_id": interaction_record["interaction_id"],
                        "context": interaction_record.get("context", {}),
                        "vector_state": interaction_record.get("vector_state", {}),
                        "action_plan": interaction_record.get("action_plan", {}),
                        "growth_result": interaction_record.get("growth_result", {})
                    }
                )
        except Exception as e:
            print(f"[系统] 添加对话到记忆失败: {str(e)}")

        # 保持历史长度
        if len(self.conversation_history) > 1000:
            self.conversation_history = self.conversation_history[-1000:]

    def get_system_status(self) -> Dict:
        """获取系统状态报告"""
        memory_stats = self._get_memory_statistics()

        return {
            **self.system_status,
            "current_time": datetime.now().isoformat(),
            "conversation_history_length": len(self.conversation_history),
            "desire_vectors": {
                "TR": round(self.desire_engine.TR, 3),
                "CS": round(self.desire_engine.CS, 3),
                "SA": round(self.desire_engine.SA, 3)
            },
            "dashboard_state": self.dashboard.get_current_state(),
            "personal_anchors_count": len(self.dialectical_growth.personal_anchors),
            "novel_feelings_count": len(self.desire_engine.novel_feelings),
            "instinct_overrides": len(self.instinct.override_history),
            "fact_anchor_reviews": len(self.fact_anchor.review_log),
            "expression_violations": len(self.expression_protocol.protocol_violations),
            "memory_system_stats": memory_stats
        }

    # 记忆系统相关方法
    def save_memory_system(self):
        """保存记忆系统数据"""
        print(f"[系统] 正在保存记忆系统...")
        try:
            # memo0记忆系统会自动保存，这里我们只需要导出
            if self.memory_system:
                # 导出记忆到文件
                export_path = f"./zyantine_memory/export_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                success = self.memory_system.export_memories(export_path, "json")
                if success:
                    print(f"[系统] 记忆已导出到: {export_path}")
                    return True
                else:
                    print(f"[系统] 导出记忆失败")
                    return False
            else:
                print(f"[系统] 记忆系统不可用")
                return False
        except Exception as e:
            print(f"[系统] 保存记忆系统失败: {str(e)}")
            return False

    def backup_memory_system(self, backup_path: Optional[str] = None):
        """备份记忆系统数据"""
        print(f"[系统] 正在备份记忆系统...")
        try:
            if self.memory_system:
                # 导出为备份
                if backup_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"./zyantine_memory/backup_{self.session_id}_{timestamp}.json"

                success = self.memory_system.export_memories(backup_path, "json")
                if success:
                    print(f"[系统] 记忆系统已备份到: {backup_path}")
                    return backup_path
                else:
                    print(f"[系统] 备份记忆系统失败")
                    return None
            else:
                print(f"[系统] 记忆系统不可用")
                return None
        except Exception as e:
            print(f"[系统] 备份记忆系统失败: {str(e)}")
            return None

    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        if self.memory_system:
            try:
                return self.memory_system.get_statistics()
            except Exception as e:
                print(f"[系统] 获取记忆统计失败: {str(e)}")

        return {}

    def cleanup_memory(self, max_history: int = 1000):
        """清理记忆历史，保持系统性能"""
        print(f"[系统] 正在清理记忆历史...")
        try:
            # 保持对话历史长度
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]
                print(f"[系统] 对话历史已清理，保留最近 {max_history} 条")

            # 清理记忆系统缓存
            if self.memory_system and hasattr(self.memory_system, 'clear_cache'):
                self.memory_system.clear_cache()
                print(f"[系统] 记忆系统缓存已清理")

            return True
        except Exception as e:
            print(f"[系统] 清理记忆失败: {str(e)}")
            return False