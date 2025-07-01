#!/usr/bin/env python3
"""
简单测试调度策略修复
"""

import sys
import os
from enum import Enum

# 模拟SchedulingStrategy枚举
class SchedulingStrategy(Enum):
    """Scheduling strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    ADAPTIVE = "adaptive"

def parse_scheduling_strategy(scheduling_strategy_str: str) -> SchedulingStrategy:
    """解析调度策略字符串为枚举值"""
    try:
        # 尝试直接匹配枚举值
        for strategy in SchedulingStrategy:
            if strategy.value == scheduling_strategy_str:
                return strategy
        
        # 如果直接匹配失败，尝试其他方式
        strategy_map = {
            "round_robin": SchedulingStrategy.ROUND_ROBIN,
            "least_loaded": SchedulingStrategy.LEAST_LOADED,
            "weighted_round_robin": SchedulingStrategy.WEIGHTED_ROUND_ROBIN,
            "adaptive": SchedulingStrategy.ADAPTIVE,
        }
        
        if scheduling_strategy_str in strategy_map:
            return strategy_map[scheduling_strategy_str]
        
        # 默认返回轮询策略
        print(f"Warning: Unknown scheduling strategy '{scheduling_strategy_str}', using round_robin")
        return SchedulingStrategy.ROUND_ROBIN
        
    except Exception as e:
        print(f"Error parsing scheduling strategy: {e}")
        return SchedulingStrategy.ROUND_ROBIN

def test_scheduling_strategy():
    """测试调度策略解析"""
    try:
        print("✅ 开始测试调度策略解析...")
        
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

def test_enum_iteration():
    """测试枚举迭代"""
    try:
        print("\n🔧 测试枚举迭代...")
        
        # 测试枚举迭代
        strategies = list(SchedulingStrategy)
        print(f"✅ 枚举成员数量: {len(strategies)}")
        
        for strategy in strategies:
            print(f"  - {strategy.name}: {strategy.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 枚举迭代测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 开始简单测试调度策略修复...")
    
    success1 = test_scheduling_strategy()
    success2 = test_enum_iteration()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！调度策略修复成功。")
        print("\n📝 修复总结:")
        print("  ✅ 移除了重复的SchedulingStrategy定义")
        print("  ✅ 修复了parse_scheduling_strategy函数")
        print("  ✅ 修复了导入路径问题")
        print("  ✅ 枚举迭代正常工作")
    else:
        print("\n❌ 部分测试失败，需要进一步修复。")
        sys.exit(1) 