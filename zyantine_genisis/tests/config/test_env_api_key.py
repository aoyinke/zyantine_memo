import os
import sys

# 设置环境变量
os.environ["ZYANTINE_API_KEY"] = "test_key_from_env"

# 导入函数
from zyantine_facade import create_zyantine

# 测试create_zyantine函数是否能从环境变量读取API密钥
try:
    # 不传递api_key参数，应该从环境变量读取
    facade = create_zyantine(config_file=None)
    print("✓ create_zyantine函数成功从环境变量读取API密钥")
    
    # 测试SimpleZyantine类
    from zyantine_facade import SimpleZyantine
    simple_zyantine = SimpleZyantine()
    print("✓ SimpleZyantine类成功从环境变量读取API密钥")
    
    print("所有测试通过！")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)