#!/usr/bin/env python3
"""
测试xDiT集成脚本
===============

用于测试ComfyUI xDiT多GPU集成是否正常工作
"""

import os
import sys
import logging
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_xdit_import():
    """测试xDiT相关导入"""
    print("🧪 测试xDiT导入...")
    
    # Test 1: xfuser package
    try:
        import xfuser
        print("✅ xfuser包导入成功")
        print(f"   版本: {getattr(xfuser, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"❌ xfuser包导入失败: {e}")
        print("   请运行: pip install xfuser")
        return False
    
    # Test 2: Ray
    try:
        import ray
        print("✅ Ray包导入成功")
        print(f"   版本: {ray.__version__}")
    except ImportError as e:
        print(f"❌ Ray包导入失败: {e}")
        print("   请运行: pip install ray")
        return False
    
    # Test 3: Flash Attention (optional)
    try:
        import flash_attn
        print("✅ Flash Attention导入成功")
    except ImportError as e:
        print(f"⚠️  Flash Attention导入失败: {e}")
        print("   这是可选的，但建议安装: pip install flash-attn")
    
    return True

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🧪 测试GPU可用性...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def test_custom_node_import():
    """测试自定义节点导入"""
    print("\n🧪 测试自定义节点导入...")
    
    try:
        # 添加custom_nodes路径
        custom_nodes_path = os.path.join(os.getcwd(), "custom_nodes", "comfyui_xdit_multigpu")
        if custom_nodes_path not in sys.path:
            sys.path.insert(0, custom_nodes_path)
        
        # 测试导入主要组件
        from xdit_runtime.dispatcher import XDiTDispatcher, SchedulingStrategy
        print("✅ XDiTDispatcher导入成功")
        
        from xdit_runtime.worker import XDiTWorker, XDiTWorkerFallback
        print("✅ XDiTWorker导入成功")
        
        from nodes import XDiTCheckpointLoader, XDiTKSampler
        print("✅ 自定义节点导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 自定义节点导入失败: {e}")
        print("详细错误:")
        traceback.print_exc()
        return False

def test_xdit_worker_creation():
    """测试xDiT Worker创建"""
    print("\n🧪 测试xDiT Worker创建...")
    
    try:
        from xdit_runtime.worker import XDiTWorkerFallback
        
        # 创建一个fallback worker进行测试
        worker = XDiTWorkerFallback(
            gpu_id=0,
            model_path="test_model_path",  # 这只是一个测试路径
            strategy="Hybrid"
        )
        print("✅ XDiTWorkerFallback创建成功")
        
        # 测试GPU信息获取
        gpu_info = worker.get_gpu_info()
        if 'error' in gpu_info:
            print(f"⚠️  GPU信息获取警告: {gpu_info['error']}")
        else:
            print(f"✅ GPU信息获取成功: {gpu_info['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Worker创建失败: {e}")
        traceback.print_exc()
        return False

def test_dispatcher_creation():
    """测试Dispatcher创建"""
    print("\n🧪 测试Dispatcher创建...")
    
    try:
        from xdit_runtime.dispatcher import XDiTDispatcher, SchedulingStrategy
        
        # 创建dispatcher
        dispatcher = XDiTDispatcher(
            gpu_devices=[0],
            model_path="test_model_path",
            strategy="Hybrid",
            scheduling_strategy=SchedulingStrategy.ROUND_ROBIN
        )
        print("✅ XDiTDispatcher创建成功")
        
        # 获取状态
        status = dispatcher.get_status()
        print(f"✅ Dispatcher状态: {status['scheduling_strategy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dispatcher创建失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 ComfyUI xDiT多GPU集成测试")
    print("=" * 50)
    
    all_tests_passed = True
    
    # 运行所有测试
    tests = [
        test_xdit_import,
        test_gpu_availability,
        test_custom_node_import,
        test_xdit_worker_creation,
        test_dispatcher_creation
    ]
    
    for test in tests:
        try:
            if not test():
                all_tests_passed = False
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 异常失败: {e}")
            all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 所有测试通过！xDiT集成应该可以正常工作")
        print("\n💡 下一步:")
        print("   1. 启动ComfyUI")
        print("   2. 在节点面板中查找'xDiT'相关节点")
        print("   3. 使用这些节点替换标准节点进行多GPU加速")
    else:
        print("❌ 部分测试失败，请检查上述错误并修复")
        print("\n🔧 常见问题解决:")
        print("   1. 确保已安装: pip install xfuser ray flash-attn")
        print("   2. 确保GPU驱动和CUDA环境正确")
        print("   3. 检查custom_nodes目录结构是否正确")

if __name__ == "__main__":
    main() 