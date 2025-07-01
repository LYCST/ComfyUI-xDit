#!/usr/bin/env python3
"""
测试numpy导入修复
"""

import sys
import os

def test_numpy_import():
    """测试numpy导入"""
    try:
        import numpy as np
        print("✅ numpy导入成功")
        
        # 测试numpy数组检测
        test_array = np.array([1, 2, 3])
        if isinstance(test_array, np.ndarray):
            print("✅ numpy数组检测正常")
        else:
            print("❌ numpy数组检测失败")
            
        return True
        
    except ImportError as e:
        print(f"❌ numpy导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_dispatcher_import():
    """测试dispatcher导入"""
    try:
        # 添加项目路径
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_nodes', 'comfyui_xdit_multigpu'))
        
        from xdit_runtime.dispatcher import XDiTDispatcher, SchedulingStrategy
        print("✅ dispatcher导入成功")
        
        # 测试创建dispatcher实例
        dispatcher = XDiTDispatcher(
            gpu_devices=[0, 1],
            model_path="/tmp/test.safetensors",
            strategy="Hybrid",
            scheduling_strategy=SchedulingStrategy.ROUND_ROBIN
        )
        print("✅ dispatcher实例创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ dispatcher测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 开始测试numpy导入修复...")
    
    success1 = test_numpy_import()
    success2 = test_dispatcher_import()
    
    if success1 and success2:
        print("\n🎉 numpy导入修复成功！")
    else:
        print("\n❌ numpy导入修复失败。")
        sys.exit(1) 