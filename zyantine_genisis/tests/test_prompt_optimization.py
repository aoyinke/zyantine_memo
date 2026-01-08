"""
测试提示词优化效果
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.prompt_engine import PromptEngine
from cognition.core_identity import CoreIdentity
from utils.logger import SystemLogger

def test_prompt_optimization():
    """测试提示词优化效果"""
    logger = SystemLogger().get_logger("test_prompt")
    
    # 创建配置
    config = {
        "prompt_templates": {
            "standard": {
                "description": "标准模板"
            }
        }
    }
    
    # 创建提示词引擎
    prompt_engine = PromptEngine(config)
    
    # 清空缓存
    prompt_engine.clear_cache()
    logger.info("已清空缓存")
    
    # 创建核心身份
    core_identity = CoreIdentity()
    core_identity.basic_profile["name"] = "测试助手"
    core_identity.basic_profile["username"] = "test_assistant"
    core_identity.basic_profile["race"] = "人类"
    core_identity.basic_profile["age"] = "19"
    core_identity.basic_profile["gender"] = "女"
    core_identity.basic_profile["citizenship"] = "中国"
    core_identity.basic_profile["perspective"] = "我"
    core_identity.basic_profile["identity"] = "测试助手"
    
    # 构建测试上下文
    context = {
        "action_plan": {
            "chosen_mask": "长期搭档",
            "primary_strategy": "支持与陪伴"
        },
        "growth_result": {
            "TR": 0.6,
            "CS": 0.7,
            "SA": 0.3
        },
        "context_analysis": {
            "emotion": "平静",
            "intent": "聊天"
        },
        "core_identity": core_identity,
        "current_vectors": {
            "TR": 0.6,
            "CS": 0.7,
            "SA": 0.3
        },
        "memory_context": {
            "resonant_memory": None
        }
    }
    
    # 构建提示词
    prompt = prompt_engine.build_prompt(**context)
    
    # 检查优化内容
    checks = {
        "AI味表达禁止": "AI味表达绝对禁止" in prompt,
        "禁止身份暴露": "作为一个AI助手" in prompt,
        "禁止学术化表达": "从某种意义上来说" in prompt,
        "禁止总结性开头": "总的来说" in prompt,
        "推荐自然表达": "推荐的自然表达" in prompt,
        "使用我觉得": "我觉得" in prompt,
        "使用咱们": "咱们" in prompt,
        "核心原则去AI化": "核心原则：去AI化表达" in prompt,
        "禁止的AI味表达": "禁止的AI味表达" in prompt,
        "绝对禁止部分": "# 绝对禁止" in prompt
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("提示词优化效果检查")
    print("="*60)
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:30s} {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ 所有优化内容都已生效！")
    else:
        print("\n❌ 部分优化内容未生效，请检查代码")
    
    # 打印提示词长度
    print(f"\n提示词总长度: {len(prompt)} 字符")
    
    # 打印关键部分
    print("\n" + "="*60)
    print("绝对禁止部分预览:")
    print("="*60)
    absolute_prohibitions_start = prompt.find("# 绝对禁止")
    if absolute_prohibitions_start != -1:
        absolute_prohibitions_end = prompt.find("\n\n", absolute_prohibitions_start + 500)
        if absolute_prohibitions_end == -1:
            absolute_prohibitions_end = len(prompt)
        print(prompt[absolute_prohibitions_start:absolute_prohibitions_end])
    
    print("\n" + "="*60)
    print("回复要求部分预览:")
    print("="*60)
    reply_requirements_start = prompt.find("# 回复要求")
    if reply_requirements_start != -1:
        reply_requirements_end = prompt.find("\n\n", reply_requirements_start + 500)
        if reply_requirements_end == -1:
            reply_requirements_end = len(prompt)
        print(prompt[reply_requirements_start:reply_requirements_end])
    
    return all_passed

if __name__ == "__main__":
    success = test_prompt_optimization()
    sys.exit(0 if success else 1)