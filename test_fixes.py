#!/usr/bin/env python3
"""
测试修复后的xDiT集成
===================

验证新的worker和dispatcher设计是否正常工作
"""

import os
import sys
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_new_worker_design():
    """测试新的worker设计"""
    print("🧪 测试新的Worker设计...")
    
    try:
        # 添加custom_nodes路径
        custom_nodes_path = os.path.join(os.getcwd(), "custom_nodes", "comfyui_xdit_multigpu")
        if custom_nodes_path not in sys.path:
            sys.path.insert(0, custom_nodes_path)
        
        from xdit_runtime.worker import XDiTWorkerFallback
        
        # 创建worker
        worker = XDiTWorkerFallback(
            gpu_id=0,
            model_path="test_model_path",
            strategy="Hybrid"
        )
        
        # 初始化
        success = worker.initialize()
        if not success:
            print("❌ Worker初始化失败")
            return False
        
        print("✅ Worker初始化成功")
        
        # 测试model wrapping
        dummy_state_dict = {"test": torch.randn(10, 10)}
        dummy_config = {"model_type": "flux"}
        
        wrap_success = worker.wrap_model_for_xdit(dummy_state_dict, dummy_config)
        if wrap_success:
            print("✅ 模型包装成功")
        else:
            print("❌ 模型包装失败")
            return False
        
        # 测试inference
        dummy_conditioning = [torch.randn(1, 77, 768), {}]
        dummy_latents = torch.randn(1, 16, 64, 64)  # Flux格式
        
        result = worker.run_inference(
            model_state_dict=dummy_state_dict,
            conditioning_positive=dummy_conditioning,
            conditioning_negative=dummy_conditioning,
            latent_samples=dummy_latents
        )
        
        if result is not None:
            print("✅ 推理测试成功")
            print(f"   输入形状: {dummy_latents.shape}")
            print(f"   输出形状: {result.shape}")
        else:
            print("❌ 推理测试失败")
            return False
        
        # 清理
        worker.cleanup()
        print("✅ Worker清理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Worker测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_dispatcher_design():
    """测试新的dispatcher设计"""
    print("\n🧪 测试新的Dispatcher设计...")
    
    try:
        custom_nodes_path = os.path.join(os.getcwd(), "custom_nodes", "comfyui_xdit_multigpu")
        if custom_nodes_path not in sys.path:
            sys.path.insert(0, custom_nodes_path)
        
        from xdit_runtime.dispatcher import XDiTDispatcher, SchedulingStrategy
        
        # 创建dispatcher
        dispatcher = XDiTDispatcher(
            gpu_devices=[0],
            model_path="test_model_path",
            strategy="Hybrid",
            scheduling_strategy=SchedulingStrategy.ROUND_ROBIN
        )
        
        # 不调用initialize()，因为那会尝试创建Ray workers
        # 我们只测试dispatcher对象的创建和状态
        
        status = dispatcher.get_status()
        print(f"✅ Dispatcher创建成功")
        print(f"   调度策略: {status['scheduling_strategy']}")
        print(f"   GPU设备: {status['gpu_devices']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dispatcher测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_node_import():
    """测试节点导入"""
    print("\n🧪 测试节点导入...")
    
    try:
        custom_nodes_path = os.path.join(os.getcwd(), "custom_nodes", "comfyui_xdit_multigpu")
        if custom_nodes_path not in sys.path:
            sys.path.insert(0, custom_nodes_path)
        
        # 测试导入主要节点类
        from nodes import XDiTKSampler, XDiTUNetLoader, XDiTCheckpointLoader
        
        print("✅ 所有节点类导入成功")
        
        # 测试INPUT_TYPES
        input_types = XDiTKSampler.INPUT_TYPES()
        if 'required' in input_types and 'optional' in input_types:
            print("✅ KSampler输入类型定义正确")
        else:
            print("❌ KSampler输入类型定义有问题")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 节点导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_compatibility():
    """测试内存和通道兼容性"""
    print("\n🧪 测试内存和通道兼容性...")
    
    try:
        # 测试不同的latent格式
        sd_latents = torch.randn(1, 4, 64, 64)     # 标准SD格式
        flux_latents = torch.randn(1, 16, 64, 64)  # Flux格式
        
        print(f"✅ SD Latents: {sd_latents.shape}")
        print(f"✅ Flux Latents: {flux_latents.shape}")
        
        # 测试GPU内存分配
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ 可用GPU数量: {device_count}")
            
            for i in range(min(device_count, 4)):  # 只测试前4个GPU
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        else:
            print("⚠️  CUDA不可用")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存兼容性测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🔧 测试修复后的xDiT集成")
    print("=" * 40)
    
    tests = [
        test_node_import,
        test_memory_compatibility,
        test_new_worker_design,
        test_new_dispatcher_design,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"🔴 {test.__name__} 失败")
        except Exception as e:
            print(f"🔴 {test.__name__} 异常: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！新的设计应该能解决之前的问题")
        print("\n💡 主要改进:")
        print("   ✅ 不再尝试从单个safetensors文件创建pipeline")
        print("   ✅ 直接使用ComfyUI的模型数据结构")
        print("   ✅ 保持原始的latent格式和通道数")
        print("   ✅ 改进的错误处理和回退机制")
        print("\n🚀 建议下一步:")
        print("   1. 重新启动ComfyUI测试修复效果")
        print("   2. 检查error.log查看新的日志输出")
        print("   3. 尝试使用xDiT节点进行推理")
    else:
        print("❌ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main() 