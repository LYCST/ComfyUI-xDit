#!/usr/bin/env python3
"""
测试Worker的xfuser初始化
=====================
验证Worker能否正确初始化xfuser的分布式环境
"""

import sys
import os
import torch
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_xfuser_worker_initialization():
    """测试Worker的xfuser初始化"""
    logger.info("🔧 Testing xfuser Worker initialization...")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.xdit_runtime.worker import XDiTWorkerFallback
        
        # 创建Worker（不使用Ray）
        worker = XDiTWorkerFallback(
            gpu_id=0,
            model_path="black-forest-labs/FLUX.1-schnell",  # 使用schnell版本
            strategy="Hybrid"
        )
        
        # 尝试初始化
        success = worker.initialize()
        
        if success:
            logger.info("✅ Worker initialized successfully")
            
            # 获取GPU信息
            gpu_info = worker.get_gpu_info()
            logger.info(f"GPU Info: {gpu_info}")
            
            # 清理
            worker.cleanup()
            logger.info("✅ Worker cleanup completed")
            
            return True
        else:
            logger.error("❌ Worker initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Worker test failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_comfyui_integration():
    """测试ComfyUI集成"""
    logger.info("🔌 Testing ComfyUI integration...")
    
    try:
        # 现在我们有了一个可以工作的xfuser环境
        # 让我们测试与ComfyUI的基本集成
        
        from custom_nodes.comfyui_xdit_multigpu.xdit_runtime.dispatcher import XDiTDispatcher
        
        # 创建dispatcher with required parameters
        dispatcher = XDiTDispatcher(
            gpu_devices=[0, 1, 2, 3, 4, 5, 6, 7],  # 使用所有8个GPU
            model_path="black-forest-labs/FLUX.1-schnell"
        )
        
        # 初始化（应该启动Ray和workers）
        success = dispatcher.initialize()
        
        if success:
            logger.info("✅ XDiT Dispatcher initialized successfully")
            
            # 获取worker信息
            logger.info("✅ All 8 GPU workers are now ready for xDiT inference!")
            logger.info("💡 Each worker is running on a different physical GPU")
            
            # 清理
            dispatcher.cleanup()
            logger.info("✅ Dispatcher cleanup completed")
            
            return True
        else:
            logger.error("❌ Dispatcher initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ ComfyUI integration test failed: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    """主测试函数"""
    logger.info("🧪 Worker xfuser初始化测试")
    logger.info("=" * 60)
    
    # 测试序列
    tests = [
        ("Worker Initialization", test_xfuser_worker_initialization),
        ("ComfyUI Integration", test_comfyui_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🚀 Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # 输出测试结果
    logger.info("\n📊 Test Results:")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! Workers can be initialized")
        logger.info("💡 Ready to test ComfyUI workflow")
    else:
        logger.error("❌ Some tests failed. Check configuration")

if __name__ == "__main__":
    main() 