#!/usr/bin/env python3
"""
测试组件初始化功能，特别是meta_cognition组件
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append('/Users/gyc/Desktop/GYC_coding/zyantine/zyantine_memo')

from zyantine_genisis.core.component_manager import ComponentManager
from zyantine_genisis.config.config_manager import ConfigManager

def test_component_initialization():
    """测试组件初始化"""
    print("=== 测试组件初始化 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get()
    
    try:
        # 创建组件管理器
        component_manager = ComponentManager(config)
        
        # 初始化组件
        components = component_manager.initialize_components()
        
        print(f"✅ 成功初始化 {len(components)} 个组件")
        
        # 检查meta_cognition组件是否成功初始化
        if 'meta_cognition' in components:
            print("✅ meta_cognition组件初始化成功")
            return True
        else:
            print("❌ meta_cognition组件初始化失败")
            # 打印所有组件状态
            print("组件状态:")
            status = component_manager.get_component_status()
            for name, stat in status.items():
                print(f"  {name}: {stat['status']} - {stat['error'] if stat['error'] else '无错误'}")
            return False
    
    except Exception as e:
        print(f"❌ 组件初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_component_initialization()
    sys.exit(0 if success else 1)
