#!/usr/bin/env python3
"""
TACO-DiT 基础功能测试脚本

测试TACO-DiT的核心组件是否正常工作
"""

import sys
import os
import logging

# 添加ComfyUI路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试模块导入"""
    logger.info("Testing TACO-DiT imports...")
    
    try:
        # 测试xDit导入
        import xfuser
        logger.info("✅ xDit imported successfully")
        
        from xfuser.core.distributed import get_world_group
        logger.info("✅ xDit distributed module imported successfully")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import xDit: {e}")
        return False
    
    try:
        # 测试TACO-DiT导入
        from comfy.taco_dit import TACODiTConfig, TACODiTConfigManager
        logger.info("✅ TACO-DiT config modules imported successfully")
        
        from comfy.taco_dit import TACODiTDistributedManager
        logger.info("✅ TACO-DiT distributed manager imported successfully")
        
        from comfy.taco_dit import TACODiTModelPatcher
        logger.info("✅ TACO-DiT model wrapper imported successfully")
        
        from comfy.taco_dit import TACODiTExecutionEngine
        logger.info("✅ TACO-DiT execution engine imported successfully")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import TACO-DiT modules: {e}")
        return False
    
    return True

def test_config_manager():
    """测试配置管理器"""
    logger.info("Testing TACO-DiT config manager...")
    
    try:
        from comfy.taco_dit import TACODiTConfigManager
        
        # 创建配置管理器
        config_manager = TACODiTConfigManager()
        
        # 测试自动检测
        config = config_manager.auto_detect_config()
        logger.info(f"✅ Auto-detected config: {config}")
        
        # 测试配置验证
        is_valid = config_manager.validate_config()
        logger.info(f"✅ Config validation: {is_valid}")
        
        # 测试并行信息
        parallel_info = config_manager.get_parallel_info()
        logger.info(f"✅ Parallel info: {parallel_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config manager test failed: {e}")
        return False

def test_distributed_manager():
    """测试分布式管理器"""
    logger.info("Testing TACO-DiT distributed manager...")
    
    try:
        from comfy.taco_dit import TACODiTDistributedManager, TACODiTConfig
        
        # 创建配置
        config = TACODiTConfig(enabled=True, auto_detect=False)
        
        # 创建分布式管理器
        dist_manager = TACODiTDistributedManager()
        
        # 测试初始化（单GPU模式）
        success = dist_manager.initialize(config)
        logger.info(f"✅ Distributed manager initialization: {success}")
        
        # 测试设备获取
        device = dist_manager.get_device()
        logger.info(f"✅ Device: {device}")
        
        # 测试世界信息
        world_info = dist_manager.get_world_info()
        logger.info(f"✅ World info: {world_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Distributed manager test failed: {e}")
        return False

def test_execution_engine():
    """测试执行引擎"""
    logger.info("Testing TACO-DiT execution engine...")
    
    try:
        from comfy.taco_dit import TACODiTExecutionEngine, TACODiTConfig
        
        # 创建配置
        config = TACODiTConfig(enabled=True, auto_detect=False)
        
        # 创建执行引擎
        engine = TACODiTExecutionEngine(config)
        
        # 测试执行信息
        exec_info = engine.get_execution_info()
        logger.info(f"✅ Execution info: {exec_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Execution engine test failed: {e}")
        return False

def test_gpu_detection():
    """测试GPU检测"""
    logger.info("Testing GPU detection...")
    
    try:
        import torch
        
        # 检测GPU数量
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ GPU count: {gpu_count}")
        
        if gpu_count > 0:
            # 检测GPU信息
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"✅ GPU {i}: {props.name}, {memory_gb:.1f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU detection test failed: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("Starting TACO-DiT basic functionality tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Detection", test_gpu_detection),
        ("Config Manager", test_config_manager),
        ("Distributed Manager", test_distributed_manager),
        ("Execution Engine", test_execution_engine),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAILED - {e}")
            results.append((test_name, False))
    
    # 总结结果
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TACO-DiT is ready for use.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 