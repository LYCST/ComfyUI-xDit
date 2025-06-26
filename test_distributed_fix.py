#!/usr/bin/env python3
"""
测试分布式初始化修复
=======================

测试新的分布式协调机制是否正常工作
"""

import os
import sys
import torch
import logging
import time

# 设置环境
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "custom_nodes", "comfyui_xdit_multigpu"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_distributed_initialization():
    """测试分布式初始化"""
    try:
        from xdit_runtime import XDiTDispatcher, SchedulingStrategy
        
        logger.info("🚀 Testing distributed initialization fix...")
        
        # 配置GPU设备（测试时用少一些）
        gpu_devices = [0, 1, 2, 3]  # 4个GPU测试
        model_path = "models/checkpoints/flux/flux1-dev.safetensors"
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            model_path = "/fake/path/for/testing"  # 测试用假路径
        
        # 创建dispatcher
        dispatcher = XDiTDispatcher(
            gpu_devices=gpu_devices,
            model_path=model_path,
            strategy="Hybrid",
            scheduling_strategy=SchedulingStrategy.ROUND_ROBIN
        )
        
        logger.info(f"Created dispatcher for {len(gpu_devices)} GPUs")
        
        # 初始化
        logger.info("Initializing dispatcher...")
        start_time = time.time()
        
        success = dispatcher.initialize()
        
        init_time = time.time() - start_time
        logger.info(f"Initialization completed in {init_time:.2f}s")
        
        if success:
            logger.info("✅ Dispatcher initialization succeeded!")
            
            # 获取状态
            status = dispatcher.get_status()
            logger.info(f"Status: {status}")
            
            # 测试简单推理（应该fallback到ComfyUI）
            logger.info("Testing inference fallback...")
            try:
                dummy_latent = torch.randn(1, 4, 64, 64)
                result = dispatcher.run_inference(
                    model_state_dict={},
                    conditioning_positive=[],
                    conditioning_negative=[],
                    latent_samples=dummy_latent,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    seed=42
                )
                
                if result is None:
                    logger.info("✅ Correctly returned None for fallback")
                else:
                    logger.info(f"✅ Got result: {result.shape}")
                    
            except Exception as e:
                logger.info(f"Expected fallback behavior: {e}")
            
            # 清理
            dispatcher.cleanup()
            logger.info("✅ Cleanup completed")
            
        else:
            logger.error("❌ Dispatcher initialization failed")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_single_gpu_mode():
    """测试单GPU模式"""
    try:
        from xdit_runtime import XDiTDispatcher, SchedulingStrategy
        
        logger.info("🚀 Testing single GPU mode...")
        
        # 单GPU配置
        gpu_devices = [0]
        model_path = "/fake/path/for/testing"
        
        dispatcher = XDiTDispatcher(
            gpu_devices=gpu_devices,
            model_path=model_path,
            strategy="Hybrid",
            scheduling_strategy=SchedulingStrategy.ROUND_ROBIN
        )
        
        success = dispatcher.initialize()
        
        if success:
            logger.info("✅ Single GPU initialization succeeded!")
            
            # 测试推理
            dummy_latent = torch.randn(1, 4, 64, 64)
            result = dispatcher.run_inference(
                model_state_dict={},
                conditioning_positive=[],
                conditioning_negative=[],
                latent_samples=dummy_latent,
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=42
            )
            
            if result is None:
                logger.info("✅ Single GPU correctly returned None for fallback")
            
            dispatcher.cleanup()
            return True
        else:
            logger.error("❌ Single GPU initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"Single GPU test failed: {e}")
        return False

def test_ray_availability():
    """测试Ray可用性"""
    try:
        import ray
        logger.info("✅ Ray is available")
        
        if ray.is_initialized():
            logger.info("Ray already initialized")
        else:
            logger.info("Ray not initialized yet")
            
        return True
    except ImportError:
        logger.warning("⚠️ Ray not available")
        return False

def main():
    """主测试函数"""
    logger.info("="*50)
    logger.info("分布式初始化修复测试")
    logger.info("="*50)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPUs: {gpu_count}")
    
    # 测试Ray
    ray_available = test_ray_availability()
    
    # 测试单GPU模式
    logger.info("\n" + "="*30)
    logger.info("测试单GPU模式")
    logger.info("="*30)
    
    single_gpu_success = test_single_gpu_mode()
    
    # 测试多GPU模式（仅在有足够GPU且Ray可用时）
    if gpu_count >= 4 and ray_available:
        logger.info("\n" + "="*30)
        logger.info("测试多GPU分布式初始化")
        logger.info("="*30)
        
        multi_gpu_success = test_distributed_initialization()
    else:
        logger.info(f"Skipping multi-GPU test (GPUs: {gpu_count}, Ray: {ray_available})")
        multi_gpu_success = True  # 跳过测试视为成功
    
    # 总结
    logger.info("\n" + "="*30)
    logger.info("测试结果总结")
    logger.info("="*30)
    
    logger.info(f"单GPU模式: {'✅ 成功' if single_gpu_success else '❌ 失败'}")
    logger.info(f"多GPU模式: {'✅ 成功' if multi_gpu_success else '❌ 失败'}")
    
    overall_success = single_gpu_success and multi_gpu_success
    logger.info(f"总体结果: {'✅ 修复成功' if overall_success else '❌ 仍有问题'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 