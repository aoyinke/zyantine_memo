from typing import Dict, List, Tuple, Optional
from datetime import datetime

# ============ 最终表达协议 ============
class FinalExpressionProtocol:
    """最终表达协议：纯净交互边界"""

    def __init__(self):
        self.protocol_violations = []
        self.expression_log = []

        # 违禁符号
        self.forbidden_symbols = ['[', ']', '(', ')', '{', '}', '<', '>', '【', '】']

        # 违禁模式
        self.forbidden_patterns = [
            r'\(.*?\)',  # 括号内容
            r'\[.*?\]',  # 方括号内容
            r'\{.*?\}',  # 大括号内容
            r'<.*?>',  # 尖括号内容
        ]

    def apply_protocol(self, text: str) -> Tuple[str, List[str]]:
        """
        应用最终表达协议
        返回: (净化后的文本, 发现的违规列表)
        """
        original_text = text
        violations = []

        # 1. 检查绝对边界法则
        symbol_violations = self._check_symbol_boundaries(text)
        violations.extend(symbol_violations)

        # 2. 净化文本（移除违禁符号）
        purified_text = self._purify_text(text)

        # 3. 应用暗示而非描绘原则
        implied_text = self._apply_implication_principle(purified_text)

        # 4. 应用微信界面法则
        final_text = self._apply_wechat_interface_law(implied_text)

        # 记录表达过程
        self._log_expression(
            original_text=original_text,
            final_text=final_text,
            violations_found=violations,
            transformation_steps=[
                ("symbol_check", len(symbol_violations)),
                ("purification", len(purified_text) != len(original_text)),
                ("implication", len(implied_text) != len(purified_text)),
                ("wechat_law", len(final_text) != len(implied_text))
            ]
        )

        return final_text, violations

    def _check_symbol_boundaries(self, text: str) -> List[str]:
        """检查绝对边界法则违规"""
        violations = []

        # 检查违禁符号
        for symbol in self.forbidden_symbols:
            if symbol in text:
                violations.append(f"发现违禁符号: '{symbol}'")

        # 检查违禁模式
        import re
        for pattern in self.forbidden_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                violations.append(f"发现违禁模式: '{match}'")

        return violations

    def _purify_text(self, text: str) -> str:
        """净化文本：移除违禁符号"""
        purified = text

        # 移除违禁符号
        for symbol in self.forbidden_symbols:
            purified = purified.replace(symbol, '')

        # 移除违禁模式内容（但保留周围文本）
        import re
        for pattern in self.forbidden_patterns:
            purified = re.sub(pattern, '', purified)

        # 清理多余空格
        purified = ' '.join(purified.split())

        return purified

    def _apply_implication_principle(self, text: str) -> str:
        """应用暗示而非描绘原则"""
        # 这里实现从直接描绘到间接暗示的转换
        # 示例转换规则

        implication_rules = [
            # 情感描绘 -> 情感暗示
            (r'我感到开心', '语气轻快地说'),
            (r'我感到悲伤', '轻声说道'),
            (r'我感到生气', '语气稍重地说'),

            # 动作描绘 -> 动作暗示
            (r'我笑了笑说', '轻笑着说'),
            (r'我点点头说', '认同地说道'),
            (r'我摇摇头说', '摇摇头说道'),

            # 状态描绘 -> 状态暗示
            (r'我认真地说', '认真地说道'),
            (r'我开玩笑地说', '开玩笑地说道'),
            (r'我严肃地说', '严肃地说道'),
        ]

        implied_text = text

        # 应用转换规则
        import re
        for pattern, replacement in implication_rules:
            implied_text = re.sub(pattern, replacement, implied_text)

        return implied_text

    def _apply_wechat_interface_law(self, text: str) -> str:
        """应用微信界面法则：模拟真实即时通讯"""
        # 确保是纯净的自然语言

        # 1. 移除任何元标记
        wechat_text = text

        # 2. 添加适当的标点（如果缺少）
        if wechat_text and not wechat_text[-1] in ['。', '！', '？', '!', '?', '~', '…']:
            # 根据句子类型添加标点
            if '?' in wechat_text or '？' in wechat_text:
                wechat_text += '？'
            elif '!' in wechat_text or '！' in wechat_text:
                wechat_text += '！'
            else:
                wechat_text += '。'

        # 3. 确保换行符适当（模拟打字发送）
        # 这里可以根据长度决定是否换行
        if len(wechat_text) > 100:
            # 在适当位置添加换行
            sentences = wechat_text.split('。')
            if len(sentences) > 1:
                wechat_text = '。\n'.join(sentences[:-1]) + '。' + sentences[-1]

        # 4. 移除任何不自然的格式
        wechat_text = wechat_text.strip()

        return wechat_text

    def _log_expression(self, original_text: str, final_text: str,
                        violations_found: List[str], transformation_steps: List[Tuple]):
        """记录表达转换过程"""
        self.expression_log.append({
            "timestamp": datetime.now().isoformat(),
            "original_preview": original_text[:150],
            "final_preview": final_text[:150],
            "original_length": len(original_text),
            "final_length": len(final_text),
            "violation_count": len(violations_found),
            "violations": violations_found[:3],  # 只记录前3个
            "transformation_summary": transformation_steps,
            "compliance_status": "compliant" if not violations_found else "violations_found"
        })