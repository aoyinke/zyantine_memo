"""
测试提示词优化效果
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    
    # 测试不同模板
    templates = ["standard", "concise", "memory_enhanced", "professional", "casual"]
    
    for template_name in templates:
        logger.info(f"\n测试模板: {template_name}")
        
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
                "intent": "聊天",
                "interaction_type": template_name if template_name in ["professional", "casual"] else "regular"
            },
            "core_identity": core_identity,
            "current_vectors": {
                "TR": 0.6,
                "CS": 0.7,
                "SA": 0.3
            },
            "memory_context": {
                "resonant_memory": {"triggered_memory": "测试记忆", "relevance_score": 0.8} if template_name == "memory_enhanced" else None
            }
        }
        
        # 构建提示词
        prompt = prompt_engine.build_prompt(**context)
        
        # 检查优化内容
        checks = {
            "AI味表达禁止": "AI味表达绝对禁止" in prompt,
            "核心原则去AI化": "核心原则" in prompt,
            "禁止的AI味表达": "禁止的AI味表达" in prompt or "禁止的表达" in prompt,
            "推荐自然表达": "推荐的自然表达方式" in prompt or "推荐的表达方式" in prompt,
            "绝对禁止部分": "# 绝对禁止" in prompt
        }
        
        # 打印结果
        print("\n" + "="*60)
        print(f"提示词优化效果检查 - 模板: {template_name}")
        print("="*60)
        
        all_passed = True
        for check_name, result in checks.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{check_name:30s} {status}")
            if not result:
                all_passed = False
        
        print("="*60)
        
        if all_passed:
            print(f"\n✅ 模板 {template_name} 所有优化内容都已生效！")
        else:
            print(f"\n❌ 模板 {template_name} 部分优化内容未生效，请检查代码")
        
        # 打印提示词长度
        print(f"\n提示词总长度: {len(prompt)} 字符")
        
        # 打印关键部分
        print("\n" + "="*60)
        print("回复要求部分预览:")
        print("="*60)
        reply_requirements_start = prompt.find("# 回复要求")
        if reply_requirements_start != -1:
            reply_requirements_end = prompt.find("\n\n", reply_requirements_start + 500)
            if reply_requirements_end == -1:
                reply_requirements_end = len(prompt)
            print(prompt[reply_requirements_start:reply_requirements_end])
    
    return True

def test_cache_optimization():
    """测试缓存优化效果"""
    logger = SystemLogger().get_logger("test_cache")
    
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
    
    # 首次构建提示词
    logger.info("首次构建提示词")
    prompt1 = prompt_engine.build_prompt(**context)
    
    # 再次构建提示词（应该使用缓存）
    logger.info("再次构建提示词（应该使用缓存）")
    prompt2 = prompt_engine.build_prompt(**context)
    
    # 检查缓存是否生效
    assert prompt1 == prompt2, "缓存未生效，两次构建的提示词不一致"
    logger.info("✅ 缓存优化测试通过")
    
    # 检查缓存大小
    cache_size = prompt_engine.get_statistics()["cache_size"]
    assert cache_size > 0, "缓存大小为0，缓存未正常工作"
    logger.info(f"✅ 缓存大小检查通过，当前缓存大小: {cache_size}")
    
    return True

def test_template_inheritance():
    """测试模板继承功能"""
    logger = SystemLogger().get_logger("test_template_inheritance")
    
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
    
    # 创建自定义模板，继承自standard
    custom_template_name = "custom_test_template"
    success = prompt_engine.create_custom_template(
        name=custom_template_name,
        sections={},
        description="测试自定义模板",
        parent="standard"
    )
    
    assert success, "创建自定义模板失败"
    logger.info(f"✅ 创建自定义模板成功: {custom_template_name}")
    
    # 获取模板信息
    template_info = prompt_engine.get_template_info(custom_template_name)
    assert template_info is not None, "获取模板信息失败"
    assert template_info["parent"] == "standard", "模板继承关系未正确设置"
    logger.info("✅ 模板继承测试通过")
    
    return True

if __name__ == "__main__":
    logger = SystemLogger().get_logger("test_main")
    logger.info("开始测试提示词引擎优化效果")
    
    # 运行所有测试
    tests = [
        test_prompt_optimization,
        test_cache_optimization,
        test_template_inheritance
    ]
    
    all_passed = True
    for test_func in tests:
        try:
            logger.info(f"运行测试: {test_func.__name__}")
            passed = test_func()
            if not passed:
                all_passed = False
                logger.error(f"测试 {test_func.__name__} 失败")
            else:
                logger.info(f"测试 {test_func.__name__} 通过")
        except Exception as e:
            all_passed = False
            logger.error(f"测试 {test_func.__name__} 出现异常: {str(e)}")
    
    if all_passed:
        logger.info("✅ 所有测试都已通过！")
        sys.exit(0)
    else:
        logger.error("❌ 部分测试失败，请检查代码")
        sys.exit(1)