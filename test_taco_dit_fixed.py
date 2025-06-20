#!/usr/bin/env python3
"""
测试修复后的TACO-DiT节点
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_taco_dit_import():
    """测试TACO-DiT模块导入"""
    try:
        # 检查torch
        import torch
        logger.info("✓ PyTorch available")
        
        # 检查TACO-DiT模块
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from comfy.taco_dit import (
            TACODiTConfig,
            TACODiTConfigManager,
            TACODiTDistributedManager,
            TACODiTModelPatcher,
            TACODiTExecutionEngine
        )
        logger.info("✓ TACO-DiT modules imported successfully")
        
        # 测试配置
        config = TACODiTConfig(enabled=True, auto_detect=True)
        logger.info(f"✓ TACO-DiT config created: {config}")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False

def test_comfyui_integration():
    """测试ComfyUI集成"""
    try:
        # 检查ComfyUI节点
        import nodes
        from nodes import common_ksampler
        logger.info("✓ ComfyUI nodes imported successfully")
        
        # 检查采样器
        import comfy.samplers
        logger.info("✓ ComfyUI samplers imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ ComfyUI import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ ComfyUI integration error: {e}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✓ CUDA available with {gpu_count} GPUs")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {gpu_name}")
            
            return True
        else:
            logger.warning("⚠ CUDA not available, using CPU")
            return True
            
    except Exception as e:
        logger.error(f"✗ GPU test error: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("="*50)
    logger.info("TACO-DiT Fixed Nodes Test")
    logger.info("="*50)
    
    tests = [
        ("TACO-DiT Import", test_taco_dit_import),
        ("ComfyUI Integration", test_comfyui_integration),
        ("GPU Availability", test_gpu_availability),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} passed")
            else:
                logger.error(f"✗ {test_name} failed")
        except Exception as e:
            logger.error(f"✗ {test_name} error: {e}")
            results.append((test_name, False))
    
    # 总结
    logger.info("\n" + "="*50)
    logger.info("Test Summary:")
    logger.info("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TACO-DiT Fixed Nodes are ready to use.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 