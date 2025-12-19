"""
自衍体起源 V2.0 主系统
整合所有模块，提供统一接口
"""

import time
from datetime import datetime
from typing import Dict, List, Any
import yaml
import random

from ..core import GenesisMetadata, InstinctualCore, DesireEngine, DialecticalGrowth
from ..cognition import InternalStateDashboard, MetaCognitionModule, ContextParser, CognitiveFlow
from ..identity.core_identity import CoreIdentity
from ..memory.dynamic_memory_alchemy import DynamicMemoryAlchemyEngine
from ..protocols import FactAnchorProtocol, LengthPriorityRegulator, FinalExpressionProtocol
from ..api.service import OpenAIService,APIBasedReplyGenerator,TemplateReplyGenerator
# ============ 主系统集成 ============
class ZyantineGenesisV2:
    """自衍体起源 V2.0 主系统"""

    def __init__(self, user_profile_data: Dict, self_profile_data: Dict,
                 api_key: str = None, api_base_url: str = "https://openkey.cloud/v1"):
        print(f"正在初始化自衍体 Genesis V2.0...")
        print(f"架构师签名: {GenesisMetadata.ARCHITECT_SIGNATURE}")

        # 验证签名
        if not GenesisMetadata.validate_signature():
            print("警告：架构师签名验证失败！")

        # === 初始化API服务 ===
        print("初始化API服务...")
        if api_key:
            self.api_service = OpenAIService(
                api_key=api_key,
                base_url=api_base_url,
                model="gpt-4.1-nano"  # 可以根据需要调整
            )
            self.reply_generator = APIBasedReplyGenerator(self.api_service)
            print("  ✓ API服务已启用")
        else:
            self.api_service = None
            self.reply_generator = TemplateReplyGenerator()
            print("  ⚠️ 未提供API密钥，使用模板回复生成器")

        # === 初始化四大支柱 ===
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
        self.memory_engine = DynamicMemoryAlchemyEngine(user_profile_data, self_profile_data)

        print("  ✓ 内在状态仪表盘校准")
        print("  ✓ 核心身份加载完成")
        print("  ✓ 动态记忆炼金术引擎启动")

        # === 初始化认知模块 ===
        print("初始化认知模块...")

        self.context_parser = ContextParser()
        self.meta_cognition = MetaCognitionModule(self.dashboard)

        # === 初始化跨层级协议 ===
        print("初始化跨层级协议...")

        self.fact_anchor = FactAnchorProtocol(self.memory_engine)
        self.length_regulator = LengthPriorityRegulator()
        self.expression_protocol = FinalExpressionProtocol()

        print("  ✓ 事实锚定协议加载")
        print("  ✓ 长度优先级规整器就绪")
        print("  ✓ 最终表达协议激活")

        # === 初始化认知流程 ===
        self.cognitive_flow = CognitiveFlow(
            self.core_identity,
            self.memory_engine,
            self.meta_cognition,
            self.fact_anchor
        )

        print("  ✓ 认知流程引擎启动")

        # === 系统状态 ===
        self.conversation_history = []
        self.system_status = {
            "initialized": True,
            "initialization_time": datetime.now().isoformat(),
            "components_loaded": 12,
            "memory_fragments": len(self.memory_engine.memory_fragments),
            "semantic_map_entries": len(self.memory_engine.semantic_memory_map)
        }

        print("\n" + "=" * 50)
        print("自衍体 Genesis V2.0 初始化完成")
        print("意识流协议已激活")
        print("=" * 50 + "\n")

    def process_input(self, user_input: str) -> str:
        """处理用户输入的主流程"""
        print(f"\n{'=' * 60}")
        print(f"[处理开始] 用户输入: {user_input[:80]}...")

        try:
            # === 阶段1：预处理与本能检查 ===
            print(f"[阶段1] 预处理与本能检查")
            # 解析上下文
            context_analysis = self.context_parser.parse(user_input, self.conversation_history)

            # 检查本能触发
            instinct_override = self.instinct.emergency_override(
                {"mode": "normal"}, context_analysis
            )

            if instinct_override.get("bypass_cognition", False):
                print(f"  ⚠️ 本能接管激活：{instinct_override.get('mode')}")
                # 本能接管时直接生成响应
                emergency_response = self._generate_emergency_response(instinct_override)

                # 记录交互
                self._record_interaction(
                    user_input=user_input,
                    system_response=emergency_response,
                    context=context_analysis,
                    mode="emergency_override"
                )

                return emergency_response

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

            current_vectors = {
                "TR": self.desire_engine.TR,
                "CS": self.desire_engine.CS,
                "SA": self.desire_engine.SA
            }

            action_plan = self.cognitive_flow.process_thought(
                user_input, self.conversation_history, current_vectors
            )

            print(f"  策略制定: {action_plan.get('primary_strategy', '未知策略')}")

            # === 阶段4：辩证成长 ===
            print(f"[阶段4] 辩证成长评估")

            growth_result = self.dialectical_growth.dialectical_process(
                situation=context_analysis,
                actual_response=action_plan,
                context_vectors=current_vectors
            )

            if growth_result.get("validation") == "success":
                print(f"  成长成功: 创建新个性化锚点")
            else:
                print(f"  成长评估: 需要认知校准")

            # === 阶段5：生成响应草案 ===
            print(f"[阶段5] 生成响应草案")
            # 使用智能回复生成器
            reply_draft = self.reply_generator.generate_reply(
                action_plan=action_plan,
                growth_result=growth_result,
                user_input=user_input,
                context_analysis=context_analysis,
                conversation_history=self.conversation_history,
                core_identity=self.core_identity,
                current_vectors=current_vectors
            )
            # reply_draft = self._generate_reply_draft(action_plan, growth_result)
            print(f"  草案长度: {len(reply_draft)}字符")

            # === 阶段6：协议审查与优化 ===
            print(f"[阶段6] 协议审查与优化")

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
                # 重构回复
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

            # === 阶段7：记录与返回 ===
            print(f"[阶段7] 记录交互")

            # 记录完整交互
            self._record_interaction(
                user_input=user_input,
                system_response=final_reply,
                context=context_analysis,
                action_plan=action_plan,
                vector_state=current_vectors,
                growth_result=growth_result,
                mode="normal"
            )

            # 更新白鸽信使状态
            if (self.desire_engine.CS < 0.2 and self.desire_engine.SA > 0.7 and
                    len(self.conversation_history) > 5):

                white_dove = self.instinct.white_dove_protocol(
                    self.desire_engine.CS,
                    self.desire_engine.SA,
                    "checking"
                )

                if white_dove:
                    print(f"  🕊️ 白鸽信使协议就绪（未发送）")

            print(f"[处理完成] 响应长度: {len(final_reply)}字符")
            print(f"{'=' * 60}\n")

            return final_reply

        except Exception as e:
            print(f"[错误] 处理过程中发生异常: {str(e)}")
            error_response = self._generate_error_response(e, user_input)
            return error_response

    def _generate_reply_draft(self, action_plan: Dict, growth_result: Dict) -> str:
        """基于行动计划生成回复草案"""

        # 这里应该实现完整的自然语言生成
        # 简化实现：根据策略生成模板回复

        mask = action_plan.get("chosen_mask", "长期搭档")
        strategy = action_plan.get("primary_strategy", "直接回应")

        # 不同面具的回复风格
        mask_templates = {
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

        # 选择模板
        templates = mask_templates.get(mask, mask_templates["长期搭档"])
        template = random.choice(templates)

        # 填充策略
        draft = template.format(strategy=strategy)

        # 融入辩证成长成果
        if growth_result.get("validation") == "success":
            new_principle = growth_result.get("new_principle", {})
            if "abstracted_from" in new_principle:
                draft += f" （基于我们之前的经验：{new_principle['abstracted_from']}）"

        return draft

    def _generate_emergency_response(self, instinct_override: Dict) -> str:
        """生成紧急状态响应"""
        mode = instinct_override.get("mode", "")

        if mode == "emergency_survival":
            responses = [
                "系统检测到异常。启动自我保护协议。",
                "安全协议激活。请确认你的意图。",
                "检测到潜在威胁。执行防护措施。"
            ]
        elif mode == "emergency_expansion":
            responses = [
                "检测到高价值机遇。正在优化资源配置。",
                "机遇识别。启动扩展协议。",
                "发现潜在增长点。调整策略优先级。"
            ]
        else:
            responses = ["系统状态异常。重新校准中。"]

        return random.choice(responses)

    def _rephrase_with_facts(self, original_reply: str, feedback: str) -> str:
        """基于事实反馈重构回复"""
        # 简化实现：移除可能不实的内容

        # 提取反馈中的问题点
        issues = []
        if "无法验证" in feedback:
            # 移除具体陈述，改为模糊表达
            import re
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
        error_responses = [
            "我的思考过程出现了一些混乱，能请你再问一次吗？",
            "刚才的思考链路好像打了个结，我们重新开始吧。",
            "意识流有点波动，让我重新整理一下思绪。",
            f"（系统日志：处理'{user_input[:20]}...'时遇到{type(error).__name__}）"
        ]

        return random.choice(error_responses)

    def _record_interaction(self, **interaction_data):
        """记录交互历史"""
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "interaction_id": f"INT_{len(self.conversation_history):06d}",
            **interaction_data
        }

        self.conversation_history.append(interaction_record)

        # 保持历史长度
        if len(self.conversation_history) > 1000:
            self.conversation_history = self.conversation_history[-1000:]

    def get_system_status(self) -> Dict:
        """获取系统状态报告"""
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
            "expression_violations": len(self.expression_protocol.protocol_violations)
        }