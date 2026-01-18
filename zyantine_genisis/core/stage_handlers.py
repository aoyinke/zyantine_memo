"""
处理管道阶段处理器
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import re
import traceback

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
            # 检查 context 是否为 None
            if context is None:
                error_msg = "预处理失败: 上下文对象为 None"
                if self.logger:
                    self.logger.error(error_msg)
                # 创建一个新的空上下文对象
                from core.processing_pipeline import StageContext
                context = StageContext(
                    user_input="",
                    conversation_history=[],
                    system_components={}
                )
                context.add_error(error_msg)
                return context

            # 清理输入
            user_input = getattr(context, 'user_input', '')
            if self.logger:
                self.logger.debug(f"预处理开始: {user_input[:50]}...")
            
            cleaned_input = self._clean_input(user_input)

            # 提取上下文信息
            context_info = self._extract_context(cleaned_input, context)

            # 更新上下文
            context.user_input = cleaned_input
            # 确保context_info是字典类型
            if not isinstance(context_info, dict):
                if self.logger:
                    self.logger.error(f"上下文信息不是字典类型，而是{type(context_info)}，将使用默认值")
                context_info = {"raw_input": cleaned_input}
            context.context_info = context_info

            if self.logger:
                self.logger.debug(f"预处理完成: 输入长度 {len(cleaned_input)}, 上下文项 {len(context_info)}")

        except Exception as e:
            error_msg = f"预处理失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            if context:
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
        if self.logger:
            self.logger.debug(f"开始提取上下文: input_text={repr(input_text[:50])}...")
            self.logger.debug(f"当前context.context_info类型: {type(context.context_info)}, 值: {repr(context.context_info)[:200]}")
        
        if not self.context_parser:
            result = {"raw_input": input_text}
            if self.logger:
                self.logger.debug(f"无上下文解析器，返回默认字典: type={type(result)}, value={result}")
            return result

        try:
            # 尝试使用上下文解析器
            if hasattr(self.context_parser, 'parse'):
                result = self.context_parser.parse(input_text, context.conversation_history)
                if self.logger:
                    self.logger.debug(f"上下文解析器返回: type={type(result)}, value={repr(result)[:200]}")
                    # 如果返回的不是字典，记录详细信息
                    if not isinstance(result, dict):
                        self.logger.error(f"上下文解析器返回了非字典类型: {type(result)}, 值: {repr(result)[:500]}")
                return result
            else:
                # 简单的关键词提取
                keywords = self._extract_keywords(input_text)
                result = {
                    "keywords": keywords,
                    "has_question": "?" in input_text,
                    "has_emotion": self._detect_emotion(input_text),
                    "length_category": self._categorize_length(input_text)
                }
                if self.logger:
                    self.logger.debug(f"使用关键词提取，返回: type={type(result)}, value={repr(result)[:200]}")
                return result
        except Exception as e:
            if self.logger:
                self.logger.warning(f"上下文解析失败，使用默认解析: {e}")
                self.logger.error(traceback.format_exc())
            result = {
                "keywords": [],
                "has_question": "?" in input_text,
                "raw_input": input_text[:200]
            }
            if self.logger:
                self.logger.debug(f"异常后返回默认字典: type={type(result)}, value={repr(result)[:200]}")
            return result

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

            # 检查 context 是否为 None
            if context is None:
                error_msg = "记忆检索失败: 上下文对象为 None"
                if self.logger:
                    self.logger.error(error_msg)
                # 创建一个新的空上下文对象
                from core.processing_pipeline import StageContext
                context = StageContext(
                    user_input="",
                    conversation_history=[],
                    system_components={}
                )
                context.add_error(error_msg)
                return context

            # 在检索长期记忆之前，先检索短期记忆并更新对话历史
            # 这样可以确保对话历史包含最新的短期记忆
            try:
                conversation_id = self.memory_manager.session_id
                memory_system = self.memory_manager.memory_system
                
                if self.logger:
                    self.logger.debug(f"[记忆检索] 开始检索短期记忆，conversation_id: {conversation_id}")
                
                # 检索短期记忆
                short_term_memories = memory_system.search_short_term_memories(
                    conversation_id=conversation_id,
                    limit=50  # 获取更多短期记忆用于上下文
                )
                
                if self.logger:
                    self.logger.debug(f"[记忆检索] 检索到 {len(short_term_memories)} 条短期记忆")
                
                # 直接使用get_conversation_history获取完整的对话历史（包括短期和长期记忆）
                # 这样确保对话历史是最新的和完整的
                try:
                    updated_history = self.memory_manager.get_conversation_history(limit=50)
                    if updated_history:
                        context.conversation_history = updated_history
                        if self.logger:
                            self.logger.info(f"[记忆检索] 更新对话历史，现在有 {len(updated_history)} 条记录")
                            # 打印前3条用于调试
                            for idx, item in enumerate(updated_history[:3]):
                                user_input = item.get("user_input", "")[:30]
                                system_response = item.get("system_response", "")[:30]
                                self.logger.debug(f"[记忆检索] 历史 {idx}: user='{user_input}...', response='{system_response}...'")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"[记忆检索] 获取完整对话历史失败: {str(e)}")
                
                # 如果短期记忆存在，也单独更新（作为备用）
                if short_term_memories:
                    # 获取当前对话历史
                    conversation_history = list(getattr(context, 'conversation_history', []))
                    
                    # 从短期记忆创建对话历史项（如果不存在）
                    existing_pairs = {(h.get("user_input", ""), h.get("system_response", "")) 
                                     for h in conversation_history if h.get("user_input") and h.get("system_response")}
                    
                    added_count = 0
                    for stm in short_term_memories:
                        metadata = stm.metadata or {}
                        user_input = metadata.get("user_input", "")
                        system_response = metadata.get("system_response", "")
                        
                        # 如果这个对话对不在现有历史中，添加它
                        if user_input and system_response and (user_input, system_response) not in existing_pairs:
                            conversation_history.append({
                                "timestamp": metadata.get("timestamp", stm.created_at.isoformat()),
                                "user_input": user_input,
                                "system_response": system_response,
                                "context": {},
                                "vector_state": {}
                            })
                            existing_pairs.add((user_input, system_response))
                            added_count += 1
                    
                    if added_count > 0:
                        # 按时间排序（最新的在前）
                        conversation_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                        # 更新context的对话历史
                        context.conversation_history = conversation_history
                        
                        if self.logger:
                            self.logger.info(f"[记忆检索] 从短期记忆添加了 {added_count} 条新记录，对话历史现在有 {len(conversation_history)} 条")
            except Exception as e:
                # 短期记忆检索失败不应影响整体流程
                if self.logger:
                    self.logger.error(f"[记忆检索] 检索短期记忆失败: {str(e)}, traceback: {traceback.format_exc()}")

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
                "user_input": getattr(context, 'user_input', ''),
                "context": getattr(context, 'context_info', {})
            })

            # 更新上下文
            context.retrieved_memories = memories
            context.resonant_memory = resonant_memory
            
            # 优化：确保context_info包含主题信息，以便后续阶段使用
            context_info = getattr(context, 'context_info', {})
            if isinstance(context_info, dict):
                # 如果检索到了记忆，提取记忆中的主题信息来增强当前主题
                if memories:
                    memory_topics = []
                    for mem in memories[:3]:  # 只看前3条最相关的记忆
                        if isinstance(mem, dict):
                            mem_topics = mem.get("metadata", {}).get("topics", [])
                            if mem_topics:
                                memory_topics.extend(mem_topics)
                    
                    # 如果从记忆中提取到了主题，更新context_info
                    if memory_topics:
                        # 统计主题出现频率
                        from collections import Counter
                        topic_counts = Counter(memory_topics)
                        most_common_topic = topic_counts.most_common(1)
                        if most_common_topic:
                            # 如果当前没有主题或主题置信度低，使用记忆中的主题
                            current_topic = context_info.get("current_topic", "unknown")
                            current_confidence = context_info.get("topic_confidence", 0.0)
                            if current_topic == "unknown" or current_confidence < 0.5:
                                context_info["memory_suggested_topic"] = most_common_topic[0][0]
                        
                        context.context_info = context_info

            if self.logger:
                self.logger.debug(f"记忆检索完成: 找到 {len(memories)} 条相关记忆")

        except Exception as e:
            error_msg = f"记忆检索失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            if context:
                context.add_error(error_msg)

        return context

    def _build_search_query(self, context: StageContext) -> str:
        """
        构建搜索查询（优化版：利用对话历史中的关键实体和信息）
        """
        # 检查 context 是否为 None
        if context is None:
            return ""
        
        # 确保context_info是字典
        context_info = getattr(context, 'context_info', {})
        if not isinstance(context_info, dict):
            if self.logger:
                self.logger.warning(f"context_info不是字典类型: {type(context_info)}")
            user_input = getattr(context, 'user_input', '')
            return user_input[:200]
        
        # 提取用户输入的关键词
        user_input = getattr(context, 'user_input', '')
        keywords = context_info.get("keywords", [])
        
        # 构建基础查询
        if keywords:
            base_query = " ".join(keywords[:5])
        else:
            base_query = user_input
        
        # 提取对话历史中的关键信息
        conversation_history = getattr(context, 'conversation_history', [])
        enhanced_query_parts = [base_query]
        
        if conversation_history:
            # 1. 提取最近对话的主题
            recent_themes = self._extract_recent_themes(conversation_history)
            if recent_themes:
                enhanced_query_parts.append(recent_themes)
            
            # 2. 提取对话历史中的关键实体
            key_entities = self._extract_entities_from_history(conversation_history)
            if key_entities:
                enhanced_query_parts.extend(key_entities[:3])  # 最多使用3个实体
            
            # 3. 提取对话主题和上下文
            dialogue_topics = self._extract_dialogue_topics(conversation_history)
            if dialogue_topics:
                enhanced_query_parts.append(dialogue_topics)
            
            # 4. 提取最近对话中的关键信息片段
            key_snippets = self._extract_key_snippets(conversation_history)
            if key_snippets:
                enhanced_query_parts.extend(key_snippets[:2])  # 最多使用2个关键片段
        
        # 合并查询部分，去重并限制长度
        query = " ".join(set(enhanced_query_parts))
        return query[:300]  # 增加到300字符以包含更多上下文

    def _extract_recent_themes(self, history: List[Dict], limit: int = 3) -> str:
        """
        提取最近对话的主题（优化版：更智能的主题提取）
        """
        if not history:
            return ""

        recent_items = history[-limit:]
        themes = []
        
        # 主题关键词库
        topic_keywords = {
            "fitness": ["体测", "体检", "健身", "运动", "锻炼", "测试", "成绩", "跑步", "跳远", "肺活量"],
            "work": ["工作", "上班", "任务", "项目", "会议", "报告", "邮件", "客户", "同事"],
            "study": ["学习", "学校", "考试", "作业", "课程", "复习", "考试", "成绩", "论文"],
            "life": ["生活", "日常", "周末", "假期", "旅行", "美食", "电影", "音乐"]
        }

        for item in recent_items:
            user_input = item.get("user_input", "")
            if not user_input:
                continue
            
            # 1. 提取关键词（过滤停用词）
            words = self._extract_meaningful_words(user_input)
            themes.extend(words[:3])
            
            # 2. 检测主题类别
            user_input_lower = user_input.lower()
            for topic, keywords in topic_keywords.items():
                if any(kw in user_input_lower for kw in keywords):
                    themes.append(topic)
                    break

        return " ".join(set(themes))
    
    def _extract_meaningful_words(self, text: str, max_words: int = 10) -> List[str]:
        """
        提取有意义的关键词（过滤停用词）
        """
        import re
        
        # 中文停用词
        stopwords = {'我', '你', '他', '她', '它', '的', '了', '在', '是', '有', '和', '与', 
                     '就', '都', '而', '及', '或', '这', '那', '哪', '吗', '呢', '啊', '吧',
                     '了', '的', '着', '过', '地', '得', '也', '还', '又', '只', '很', '更',
                     '都', '要', '想', '会', '能', '应该', '可以', '可能', '已经', '还是'}
        
        # 提取词汇
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词和短词
        meaningful = [w for w in words if w not in stopwords and len(w) > 1]
        
        # 统计频率并返回最常见的词
        from collections import Counter
        word_counts = Counter(meaningful)
        
        return [word for word, _ in word_counts.most_common(max_words)]
    
    def _extract_entities_from_history(self, history: List[Dict], limit: int = 5) -> List[str]:
        """
        从对话历史中提取关键实体（人物、地点、事件等）
        """
        import re
        
        entities = []
        recent_items = history[-limit:] if len(history) > limit else history
        
        for item in recent_items:
            user_input = item.get("user_input", "")
            if not user_input:
                continue
            
            # 1. 提取日期和时间
            date_patterns = [
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
                r'\d{1,2}月\d{1,2}日',
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'昨天|今天|明天|上周|本周|下周'
            ]
            for pattern in date_patterns:
                matches = re.findall(pattern, user_input)
                entities.extend(matches[:2])  # 每条最多2个日期
            
            # 2. 提取数字和金额
            number_patterns = [
                r'¥\d+\.?\d*',
                r'\$\d+\.?\d*',
                r'\d+%',
                r'第\d+',
            ]
            for pattern in number_patterns:
                matches = re.findall(pattern, user_input)
                entities.extend(matches[:2])  # 每条最多2个数字
            
            # 3. 提取可能的专有名词（首字母大写或中文）
            # 这里使用简单启发式：长度大于1且不在停用词中的词
            words = self._extract_meaningful_words(user_input, max_words=5)
            entities.extend(words)
        
        # 去重并返回最常见的实体
        from collections import Counter
        entity_counts = Counter(entities)
        
        # 返回出现频率较高的实体（至少出现1次）
        return [entity for entity, count in entity_counts.most_common(10) if count >= 1]
    
    def _extract_dialogue_topics(self, history: List[Dict], limit: int = 5) -> str:
        """
        提取对话主题（更深入的语义分析）
        """
        if not history:
            return ""
        
        recent_items = history[-limit:] if len(history) > limit else history
        topic_words = []
        
        # 主题关键词映射
        topic_keywords = {
            "fitness": ["体测", "体检", "健身", "运动", "锻炼", "测试", "成绩"],
            "work": ["工作", "上班", "任务", "项目", "会议", "报告"],
            "study": ["学习", "学校", "考试", "作业", "课程"],
            "life": ["生活", "日常", "周末", "假期", "旅行"]
        }
        
        detected_topics = []
        for item in recent_items:
            user_input = item.get("user_input", "")
            if not user_input:
                continue
            
            user_input_lower = user_input.lower()
            
            # 检测主题
            for topic, keywords in topic_keywords.items():
                if any(kw in user_input_lower for kw in keywords):
                    if topic not in detected_topics:
                        detected_topics.append(topic)
            
            # 提取关键词
            words = self._extract_meaningful_words(user_input, max_words=3)
            topic_words.extend(words)
        
        # 合并主题和关键词
        all_topics = detected_topics + topic_words
        return " ".join(set(all_topics))
    
    def _extract_key_snippets(self, history: List[Dict], limit: int = 3, snippet_length: int = 20) -> List[str]:
        """
        提取最近对话中的关键信息片段
        """
        if not history:
            return []
        
        recent_items = history[-limit:] if len(history) > limit else history
        snippets = []
        
        # 关键词指示器
        key_indicators = ["重要", "关键", "记住", "注意", "需要", "应该", "必须", "务必"]
        
        for item in recent_items:
            user_input = item.get("user_input", "")
            if not user_input:
                continue
            
            # 检查是否包含关键指示器
            user_input_lower = user_input.lower()
            if any(indicator in user_input_lower for indicator in key_indicators):
                # 提取包含关键指示器的句子片段
                words = user_input.split()
                for i, word in enumerate(words):
                    if any(ind in word.lower() for ind in key_indicators):
                        # 提取该词周围的上下文
                        start = max(0, i - snippet_length // 2)
                        end = min(len(words), i + snippet_length // 2)
                        snippet = " ".join(words[start:end])
                        snippets.append(snippet)
                        break
        
        return snippets


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
        # 确保context_info是字典
        if not isinstance(context.context_info, dict):
            self.logger.warning(f"context_info不是字典类型: {type(context.context_info)}")
            emotion = "neutral"
        else:
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

            # 更新上下文 - 存储认知快照和最终行动方案
            context.cognitive_snapshot = cognitive_result.get("cognitive_snapshot")
            final_action_plan = cognitive_result.get("final_action_plan")
            context.cognitive_result = final_action_plan

            # 提取策略和情感
            if final_action_plan:
                context.strategy = final_action_plan.get("primary_strategy")
                context.emotional_context = final_action_plan.get("emotional_context", {})

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
                desire_vectors=context.desire_vectors,
                context_info=context.context_info
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
        """生成回复 - 优化版：减少日志和不必要的处理"""
        try:
            # 检查 context 是否为 None
            if context is None:
                from core.processing_pipeline import StageContext
                context = StageContext(
                    user_input="",
                    conversation_history=[],
                    system_components={}
                )
                context.add_error("回复生成失败: 上下文对象为 None")
                return context

            # 1. 确定面具并保存到 context
            chosen_mask = self._determine_mask(context)
            context.mask_type = chosen_mask

            # 构建回复生成参数 - 简化版
            generation_params = self._build_generation_params_fast(context, chosen_mask)

            # 直接生成回复，跳过复杂的日志记录
            if context.cognitive_result:
                reply_result = self.reply_generator.generate_from_cognitive_flow(generation_params)
            else:
                reply_result = self.reply_generator.generate_reply(**generation_params)
            
            # 处理返回值
            if isinstance(reply_result, tuple):
                reply_text, emotion = reply_result
                context.emotional_context = context.emotional_context or {}
                context.emotional_context['reply_emotion'] = emotion
            else:
                reply_text = reply_result
            
            # 检测是否是降级回复
            context.is_fallback_reply = self._is_fallback_reply(reply_text)
            
            # 更新上下文
            context.final_reply = reply_text

        except Exception as e:
            if self.logger:
                self.logger.error(f"回复生成失败: {str(e)}")
            if context:
                context.add_error(f"回复生成失败: {str(e)}")

        return context
    
    def _build_generation_params_fast(self, context: StageContext, chosen_mask: str) -> Dict:
        """
        快速构建回复生成参数 - 优化版：增强对话历史处理以保持话题连贯性
        
        关键改进：
        1. 增加对话历史数量从5条到10条
        2. 格式化对话历史，确保user_input和system_response清晰分离
        3. 添加当前话题信息到生成参数
        """
        if context is None:
            return {
                "action_plan": {"chosen_mask": chosen_mask, "primary_strategy": None},
                "memory_context": {"retrieved_memories": [], "resonant_memory": None},
                "user_input": "",
                "conversation_history": [],
                "growth_result": None,
                "context_analysis": {},
                "current_vectors": {}
            }
        
        # 简化的记忆上下文处理
        memory_context = {
            "retrieved_memories": [],
            "resonant_memory": None
        }
        
        # 只在有记忆时才处理
        retrieved_memories = getattr(context, 'retrieved_memories', [])
        if retrieved_memories:
            memory_context["retrieved_memories"] = [
                self._convert_memory_to_dict(mem) for mem in retrieved_memories[:5]  # 增加到5条
            ]
        
        resonant_memory = getattr(context, 'resonant_memory', None)
        if resonant_memory:
            memory_context["resonant_memory"] = self._convert_memory_to_dict(resonant_memory)
        
        # 优化：增加对话历史数量到10条，并格式化
        raw_history = getattr(context, 'conversation_history', [])
        formatted_history = self._format_conversation_history(raw_history[-10:])  # 增加到10条
        
        # 获取上下文信息，确保是字典类型
        context_info = getattr(context, 'context_info', {})
        if not isinstance(context_info, dict):
            context_info = {}
        
        # 提取当前主题信息
        current_topic = context_info.get("current_topic", "")
        topic_confidence = context_info.get("topic_confidence", 0.0)
        referential_analysis = context_info.get("referential_analysis", {})
        
        # 提取前文承诺和上下文链接信息（关键：解决"不知道指的是什么"问题）
        context_links = context_info.get("context_links", {})
        pending_promises = context_info.get("pending_promises", [])
        likely_reference = context_info.get("likely_reference")
        has_unresolved_context = context_info.get("has_unresolved_context", False)
        
        # 构建增强的上下文分析
        enhanced_context_analysis = {
            **context_info,
            "current_topic": current_topic,
            "topic_confidence": topic_confidence,
            "referential_analysis": referential_analysis,
            "context_links": context_links,
            "pending_promises": pending_promises,
            "likely_reference": likely_reference,
            "has_unresolved_context": has_unresolved_context,
            "conversation_turn_count": len(formatted_history)  # 添加对话轮数信息
        }
        
        return {
            "action_plan": {
                "chosen_mask": chosen_mask,
                "primary_strategy": getattr(context, 'strategy', None)
            },
            "memory_context": memory_context,
            "user_input": getattr(context, 'user_input', ''),
            "conversation_history": formatted_history,
            "growth_result": getattr(context, 'growth_result', None),
            "context_analysis": enhanced_context_analysis,
            "current_vectors": getattr(context, 'desire_vectors', {})
        }
    
    def _format_conversation_history(self, history: List[Dict]) -> List[Dict]:
        """
        格式化对话历史，确保格式统一且包含完整信息
        
        Args:
            history: 原始对话历史列表
            
        Returns:
            格式化后的对话历史列表
        """
        formatted = []
        
        for conv in history:
            if not isinstance(conv, dict):
                continue
            
            # 提取用户输入和系统回复
            user_input = conv.get("user_input", "")
            system_response = conv.get("system_response", "")
            timestamp = conv.get("timestamp", "")
            
            # 如果没有直接的user_input/system_response，尝试从content解析
            if not user_input and not system_response:
                content = conv.get("content", "")
                if isinstance(content, str):
                    # 尝试解析格式化的对话内容
                    if "用户:" in content or "用户：" in content:
                        lines = content.split("\n")
                        for line in lines:
                            line = line.strip()
                            if line.startswith("用户:") or line.startswith("用户："):
                                user_input = line.replace("用户:", "").replace("用户：", "").strip()
                            elif line.startswith("AI:") or line.startswith("AI："):
                                system_response = line.replace("AI:", "").replace("AI：", "").strip()
                    else:
                        # 如果是纯文本，假设是用户输入
                        user_input = content
            
            # 只添加有效的对话记录
            if user_input or system_response:
                formatted.append({
                    "user_input": user_input,
                    "system_response": system_response,
                    "timestamp": timestamp
                })
        
        return formatted
    
    def _convert_memory_to_dict(self, mem) -> Dict:
        """将记忆对象转换为字典"""
        if isinstance(mem, dict):
            return mem
        
        return {
            'memory_id': getattr(mem, 'memory_id', 'unknown'),
            'content': getattr(mem, 'content', ''),
            'memory_type': str(getattr(mem, 'memory_type', 'unknown')),
            'metadata': getattr(mem, 'metadata', {}),
            'relevance_score': getattr(mem, 'relevance_score', 0.0)
        }

    def _build_generation_params(self, context: StageContext, chosen_mask: str) -> Dict:
        """构建回复生成参数"""
        # 检查 context 是否为 None
        if context is None:
            if self.logger:
                self.logger.error("_build_generation_params: context is None")
            return {
                "action_plan": {
                    "chosen_mask": chosen_mask,
                    "primary_strategy": None
                },
                "memory_context": {
                    "retrieved_memories": [],
                    "resonant_memory": None
                },
                "user_input": "",
                "conversation_history": [],
                "growth_result": None,
                "context_analysis": {},
                "current_vectors": {}
            }
        
        if self.logger:
            self.logger.debug(f"_build_generation_params: context type is {type(context)}")
            self.logger.debug(f"_build_generation_params: context has user_input attribute: {hasattr(context, 'user_input')}")
            if hasattr(context, 'user_input'):
                self.logger.debug(f"_build_generation_params: context.user_input is {context.user_input}")
        
        return {
            "action_plan": {
                "chosen_mask": chosen_mask,
                "primary_strategy": getattr(context, 'strategy', None)
            },
            "memory_context": {
                "retrieved_memories": getattr(context, 'retrieved_memories', []),
                "resonant_memory": getattr(context, 'resonant_memory', None)
            },
            "user_input": getattr(context, 'user_input', ''),
            "conversation_history": getattr(context, 'conversation_history', []),
            "growth_result": getattr(context, 'growth_result', None),
            "context_analysis": getattr(context, 'context_info', {}),
            "current_vectors": getattr(context, 'desire_vectors', {})
        }

    def _is_fallback_reply(self, reply_text: str) -> bool:
        """
        检测是否是降级回复
        
        Args:
            reply_text: 回复文本
            
        Returns:
            是否是降级回复
        """
        if not reply_text or not isinstance(reply_text, str):
            return True
        
        # 检查长度（降级回复通常很短）
        if len(reply_text) < 20:
            return True
        
        # 检查是否是常见的降级回复模式
        fallback_patterns = [
            "我收到了你的消息",
            "我收到了",
            "收到了",
            "我思考了一下",
            "能请你再问一次吗",
            "我们重新开始吧",
            "让我重新整理一下",
            "刚才的思考",
            "意识流有点波动",
            "思考过程出现了一些混乱",
        ]
        
        reply_lower = reply_text.lower()
        for pattern in fallback_patterns:
            if pattern in reply_lower:
                return True
        
        return False
    
    def _determine_mask(self, context: StageContext) -> str:
        """确定使用哪个面具（角色）"""
        default_mask = "长期搭档"

        # 检查 context 是否为 None
        if context is None:
            return default_mask

        # 安全地获取 desire_vectors 属性
        desire_vectors = getattr(context, 'desire_vectors', None)
        if not desire_vectors:
            return default_mask

        connection_strength = desire_vectors.get("connection", 0)
        if connection_strength > 0.7:
            return "知己" if random.random() > 0.5 else "伴侣"

        growth_strength = desire_vectors.get("growth", 0)
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

        # 检查 context 是否为 None
        if context is None:
            return random.choice(templates)

        # 安全地获取 emotional_context 属性
        emotional_context = getattr(context, 'emotional_context', {})
        emotion = emotional_context.get("dominant_emotion", "neutral")

        if emotion == "positive" and len(templates) > 1:
            return templates[0]
        elif emotion == "negative" and len(templates) > 2:
            return templates[1]
        else:
            return random.choice(templates)

    def _fill_template(self, template: str, context: StageContext) -> str:
        """填充模板"""
        if not template:
            # 检查 context 是否为 None
            if context is None:
                return "我思考了一下，但还没有形成完整的回复。"
            strategy = getattr(context, 'strategy', None)
            return strategy or "我思考了一下，但还没有形成完整的回复。"

        reply = template

        # 检查 context 是否为 None
        if context is not None:
            strategy = getattr(context, 'strategy', None)
            if "{strategy}" in reply and strategy:
                reply = reply.replace("{strategy}", strategy)
            elif strategy:
                reply = f"{reply} {strategy}"

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

            # 检查 context 是否为 None
            if context is None:
                error_msg = "协议审查失败: 上下文对象为 None"
                if self.logger:
                    self.logger.error(error_msg)
                # 创建一个新的空上下文对象
                from core.processing_pipeline import StageContext
                context = StageContext(
                    user_input="",
                    conversation_history=[],
                    system_components={}
                )
                context.add_error(error_msg)
                return context

            if not getattr(context, 'final_reply', None):
                error_msg = "没有可审查的回复"
                if self.logger:
                    self.logger.error(error_msg)
                context.add_error(error_msg)
                return context

            review_results = []
            
            # 检查是否是降级回复
            is_fallback = getattr(context, 'is_fallback_reply', False)
            
            if is_fallback:
                # 如果是降级回复，跳过API事实审查，只做文本层面的处理
                if self.logger:
                    self.logger.debug("检测到降级回复，跳过API事实审查，只进行文本处理")
                
                # 只进行文本层面的协议处理（长度、表达验证）
                if self.protocol_engine and hasattr(self.protocol_engine, 'apply_text_protocols'):
                    # 如果协议引擎有文本协议方法，只调用文本协议
                    protocol_context = {
                        "user_input": getattr(context, 'user_input', ''),
                        "conversation_history": getattr(context, 'conversation_history', []),
                        "cognitive_snapshot": context.cognitive_snapshot if hasattr(context, 'cognitive_snapshot') else None,
                        "core_identity": context.core_identity if hasattr(context, 'core_identity') else None,
                        "skip_fact_check": True  # 标记跳过事实检查
                    }
                    
                    try:
                        final_text, protocol_summary = self.protocol_engine.apply_all_protocols(
                            draft=context.final_reply,
                            context=protocol_context
                        )
                        
                        if final_text:
                            context.final_reply = final_text
                        
                        # 提取文本协议结果
                        if protocol_summary and isinstance(protocol_summary, dict):
                            protocol_steps = protocol_summary.get("protocol_steps", {})
                            for step_name, step_result in protocol_steps.items():
                                # 跳过事实检查结果
                                if step_name == "fact_check":
                                    continue
                                review_results.append({
                                    "type": step_name,
                                    "result": {
                                        "status": step_result.get("status") if isinstance(step_result, dict) else "unknown",
                                        "feedback": step_result.get("feedback", "") if isinstance(step_result, dict) else "",
                                        "needs_fix": False
                                    }
                                })
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"降级回复的文本协议处理失败: {e}")
                
                # 添加降级标记到审查结果
                review_results.append({
                    "type": "fallback_notice",
                    "result": {
                        "status": "skipped",
                        "feedback": "降级回复，跳过完整协议审查",
                        "needs_fix": False
                    }
                })
            else:
                # 正常回复，执行完整协议审查
                if self.protocol_engine and hasattr(self.protocol_engine, 'apply_all_protocols'):
                    # 构建上下文信息
                    protocol_context = {
                        "user_input": getattr(context, 'user_input', ''),
                        "conversation_history": getattr(context, 'conversation_history', []),
                        "cognitive_snapshot": context.cognitive_snapshot if hasattr(context, 'cognitive_snapshot') else None,
                        "core_identity": context.core_identity if hasattr(context, 'core_identity') else None
                    }

                    # 应用所有协议
                    try:
                        final_text, protocol_summary = self.protocol_engine.apply_all_protocols(
                            draft=context.final_reply,
                            context=protocol_context
                        )
                        
                        # 确保 protocol_summary 是有效字典
                        if not protocol_summary or not isinstance(protocol_summary, dict):
                            if self.logger:
                                self.logger.warning("协议摘要为空或无效，使用默认摘要")
                            protocol_summary = {
                                "protocol_steps": {},
                                "conflicts_detected": [],
                                "conflicts_resolved": [],
                                "original_draft_length": len(context.final_reply),
                                "final_text_length": len(final_text) if final_text else len(context.final_reply),
                                "total_reduction": 0,
                                "total_processing_time": 0
                            }

                        # 更新最终回复
                        if final_text:
                            context.final_reply = final_text

                        # 提取协议步骤结果
                        protocol_steps = protocol_summary.get("protocol_steps", {})
                        if protocol_steps and isinstance(protocol_steps, dict):
                            for step_name, step_result in protocol_steps.items():
                                if isinstance(step_result, dict):
                                    status = step_result.get("status")
                                    review_results.append({
                                        "type": step_name,
                                        "result": {
                                            "status": status,
                                            "feedback": step_result.get("feedback", ""),
                                            "needs_fix": status in ["failed", "violations_found"] if status else False
                                        }
                                    })

                        # 添加冲突信息
                        conflicts = protocol_summary.get("conflicts_detected", [])
                        if conflicts and isinstance(conflicts, list) and len(conflicts) > 0:
                            resolved = protocol_summary.get("conflicts_resolved", [])
                            if not isinstance(resolved, list):
                                resolved = []
                            review_results.append({
                                "type": "conflicts",
                                "result": {
                                    "conflicts": conflicts,
                                    "resolved": resolved,
                                    "needs_fix": len(conflicts) > 0
                                }
                            })

                        # 添加摘要信息
                        review_results.append({
                            "type": "summary",
                            "result": {
                                "original_length": protocol_summary.get("original_draft_length", len(context.final_reply)),
                                "final_length": protocol_summary.get("final_text_length", len(final_text) if final_text else len(context.final_reply)),
                                "reduction": protocol_summary.get("total_reduction", 0),
                                "processing_time": protocol_summary.get("total_processing_time", 0)
                            }
                        })
                    except Exception as e:
                        error_msg = f"协议审查执行失败: {str(e)}"
                        if self.logger:
                            self.logger.error(error_msg)
                        context.add_error(error_msg)
                        # 添加默认审查结果
                        review_results.append({
                            "type": "error",
                            "result": {
                                "status": "failed",
                                "feedback": error_msg,
                                "needs_fix": False
                            }
                        })

            # 元认知评估
            if self.meta_cognition and hasattr(self.meta_cognition, 'evaluate'):
                meta_evaluation = self.meta_cognition.evaluate(
                    reply=context.final_reply,
                    context=context
                )
                review_results.append({"type": "meta_cognition", "result": meta_evaluation})

            # 更新上下文
            context.review_results = review_results

            if self.logger:
                self.logger.debug(f"协议审查完成: 执行 {len(review_results)} 项检查")

        except Exception as e:
            error_msg = f"协议审查失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            if context:
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
            # 确保context_info是字典
            if not isinstance(context.context_info, dict):
                self.logger.warning(f"context_info不是字典类型: {type(context.context_info)}")
                interaction_data["emotional_intensity"] = 0.5
            else:
                interaction_data["emotional_intensity"] = context.context_info.get("emotional_intensity", 0.5)

            # 添加 interaction_id
            import hashlib
            import json
            interaction_id = hashlib.md5(
                json.dumps(interaction_data, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            interaction_data["interaction_id"] = interaction_id

            self.logger.debug(f"打印交互数据我看一下: {interaction_data}")
            
            # 先添加到短期记忆（无论长期记忆是否成功都要保存）
            # 这样确保对话历史能够立即被检索到
            short_term_memory_id = None
            try:
                # 使用session_id作为conversation_id
                conversation_id = self.memory_manager.session_id
                memory_system = self.memory_manager.memory_system
                
                # 确保final_reply不为空
                system_response = context.final_reply or ""
                if not system_response:
                    if self.logger:
                        self.logger.warning("final_reply为空，无法保存到短期记忆")
                else:
                    # 构建短期记忆内容
                    short_term_content = f"用户: {context.user_input}\nAI: {system_response}"
                    
                    # 添加到短期记忆
                    short_term_memory_id = memory_system.add_short_term_memory(
                        content=short_term_content,
                        conversation_id=conversation_id,
                        metadata={
                            "user_input": context.user_input,
                            "system_response": system_response,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    if self.logger:
                        self.logger.info(
                            f"[短期记忆] 添加成功，memory_id: {short_term_memory_id}, "
                            f"conversation_id: {conversation_id}, "
                            f"user_input: '{context.user_input[:50]}...', "
                            f"system_response: '{system_response[:50]}...'"
                        )
                        # 立即验证短期记忆是否可检索
                        try:
                            verify_memories = memory_system.search_short_term_memories(
                                conversation_id=conversation_id,
                                limit=5
                            )
                            self.logger.debug(f"[短期记忆] 验证：检索到 {len(verify_memories)} 条短期记忆")
                        except Exception as e:
                            self.logger.warning(f"[短期记忆] 验证失败: {e}")
            except Exception as e:
                # 短期记忆添加失败不应该影响整体流程
                if self.logger:
                    self.logger.error(f"[短期记忆] 添加失败: {str(e)}, traceback: {traceback.format_exc()}")
            
            # 记录到长期记忆系统
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