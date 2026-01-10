import os
import sys
import json

# 测试配置文件路径
CONFIG_PATH = "config/llm_config.json"

# 1. 测试配置文件是否存在
if not os.path.exists(CONFIG_PATH):
    print(f"✗ 配置文件 {CONFIG_PATH} 不存在")
    sys.exit(1)

print(f"✓ 配置文件 {CONFIG_PATH} 存在")

# 2. 测试读取配置文件
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 获取当前启用的provider
    current_provider = config["api"]["provider"]
    if current_provider in config["api"]["providers"]:
        provider_config = config["api"]["providers"][current_provider]
        api_key = provider_config["api_key"]
        base_url = provider_config["base_url"]
        
        print(f"✓ 从配置文件读取到当前provider: {current_provider}")
        print(f"✓ API密钥: {api_key[:8]}..." if api_key else "✓ API密钥为空")
        print(f"✓ Base URL: {base_url}")
    else:
        print(f"✗ 当前provider {current_provider} 不在配置中")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ 读取配置文件失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试create_zyantine函数是否能正确读取配置文件
try:
    from zyantine_facade import create_zyantine
    
    # 使用配置文件创建实例
    facade = create_zyantine(config_file=CONFIG_PATH)
    print("✓ create_zyantine函数成功使用配置文件创建实例")
    
except Exception as e:
    print(f"✗ create_zyantine函数使用配置文件失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试默认配置文件路径
try:
    # 不传递config_file参数，应该使用默认配置文件路径
    facade = create_zyantine()
    print("✓ create_zyantine函数成功使用默认配置文件路径")
    
except Exception as e:
    print(f"✗ create_zyantine函数使用默认配置文件路径失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n所有测试通过！")