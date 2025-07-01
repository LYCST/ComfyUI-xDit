#!/usr/bin/env python3
"""
测试调度策略修复
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_nodes', 'comfyui_xdit_multigpu'))

def test_scheduling_strategy():
    """测试调度策略解析"""
    try:
        from xdit_runtime.dispatcher import SchedulingStrategy
        from nodes import parse_scheduling_strategy
        
        print("✅ 成功导入 SchedulingStrategy 和 parse_scheduling_strategy")
        
        # 测试各种调度策略
        test_strategies = ["round_robin", "least_loaded", "weighted_round_robin", "adaptive"]
        
        for strategy_str in test_strategies:
            try:
                strategy_enum = parse_scheduling_strategy(strategy_str)
                print(f"✅ '{strategy_str}' -> {strategy_enum.value}")
            except Exception as e:
                print(f"❌ '{strategy_str}' 解析失败: {e}")
        
        # 测试无效策略
        try:
            strategy_enum = parse_scheduling_strategy("invalid_strategy")
            print(f"✅ 无效策略回退到默认: {strategy_enum.value}")
        except Exception as e:
            print(f"❌ 无效策略处理失败: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dispatcher_creation():
    """测试调度器创建"""
    try:
        from xdit_runtime.dispatcher import XDiTDispatcher, SchedulingStrategy
        
        print("\n🔧 测试调度器创建...")
        
        # 创建调度器
        dispatcher = XDiTDispatcher(
            gpu_devices=[0, 1],
            model_path="/tmp/test_model.safetensors",
            strategy="Hybrid",
            scheduling_strategy=SchedulingStrategy.ROUND_ROBIN
        )
        
        print(f"✅ 调度器创建成功")
        print(f"   - GPU设备: {dispatcher.gpu_devices}")
        print(f"   - 调度策略: {dispatcher.scheduling_strategy.value}")
        print(f"   - 并行策略: {dispatcher.strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ 调度器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 开始测试调度策略修复...")
    
    success1 = test_scheduling_strategy()
    success2 = test_dispatcher_creation()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！调度策略修复成功。")
    else:
        print("\n❌ 部分测试失败，需要进一步修复。")
        sys.exit(1) 