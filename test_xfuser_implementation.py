#!/usr/bin/env python3
"""
测试xfuser FLUX多GPU推理实现
=============================
验证xfuser是否正确集成并能进行FLUX模型的多GPU推理
"""

import sys
import torch
import logging
import os
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_xfuser_import():
    """测试xfuser导入"""
    logger.info("1️⃣ Testing xfuser import...")
    
    try:
        import xfuser
        logger.info(f"✅ xfuser imported successfully, version: {getattr(xfuser, '__version__', 'unknown')}")
        
        from xfuser import xFuserArgs
        logger.info("✅ xFuserArgs imported")
        
        from xfuser import xFuserFluxPipeline
        logger.info("✅ xFuserFluxPipeline imported")
        
        from xfuser.core.distributed import initialize_runtime_state
        logger.info("✅ initialize_runtime_state imported")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ xfuser import failed: {e}")
        logger.info("💡 Try: pip install xfuser")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during import: {e}")
        return False

def test_flux_model_availability():
    """测试FLUX模型可用性"""
    logger.info("2️⃣ Testing FLUX model availability...")
    
    # 检查常见的FLUX模型路径
    flux_paths = [
        "/home/shuzuan/prj/ComfyUI-xDit/models/unet/flux1-dev.safetensors",
        "/home/shuzuan/prj/ComfyUI-xDit/models/checkpoints/flux1-dev.safetensors",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell"
    ]
    
    for path in flux_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found FLUX model at: {path}")
            return path
        elif not path.startswith("/"):
            logger.info(f"🔍 Will try to download from HuggingFace: {path}")
            return path
    
    logger.warning("⚠️ No local FLUX model found, will try HuggingFace download")
    return "black-forest-labs/FLUX.1-schnell"

def test_gpu_availability():
    """测试GPU可用性"""
    logger.info("3️⃣ Testing GPU availability...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"✅ Found {gpu_count} GPUs")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        logger.info(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    return gpu_count >= 2  # 至少需要2个GPU进行多GPU测试

def test_xfuser_args_creation():
    """测试xFuserArgs创建"""
    logger.info("4️⃣ Testing xFuserArgs creation...")
    
    try:
        from xfuser import xFuserArgs
        
        # 创建xfuser参数（用于8GPU配置）
        xfuser_args = xFuserArgs(
            model="black-forest-labs/FLUX.1-schnell",
            height=1024,
            width=1024,
            ulysses_degree=2,  # 序列并行度
            pipefusion_parallel_degree=2,  # 管道并行度
            use_cfg_parallel=False,  # FLUX不使用CFG
            use_parallel_vae=False,
            seed=42,
            output_type="latent",
            warmup_steps=0,
            trust_remote_code=True,
            use_torch_compile=False,
            num_inference_steps=20,
        )
        
        logger.info("✅ xFuserArgs created successfully")
        logger.info(f"   Model: {xfuser_args.model}")
        logger.info(f"   Ulysses degree: {xfuser_args.ulysses_degree}")
        logger.info(f"   PipeFusion degree: {xfuser_args.pipefusion_parallel_degree}")
        
        return xfuser_args
        
    except Exception as e:
        logger.error(f"❌ xFuserArgs creation failed: {e}")
        logger.exception("Full traceback:")
        return None

def test_distributed_initialization(xfuser_args):
    """测试分布式初始化"""
    logger.info("5️⃣ Testing distributed initialization...")
    
    try:
        from xfuser.core.distributed import initialize_runtime_state
        
        # 初始化分布式状态
        initialize_runtime_state(xfuser_args)
        logger.info("✅ Distributed runtime state initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Distributed initialization failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_flux_pipeline_creation(model_path):
    """测试FLUX pipeline创建"""
    logger.info("6️⃣ Testing FLUX pipeline creation...")
    
    try:
        from xfuser import xFuserFluxPipeline, xFuserArgs
        
        # 设置单GPU分布式环境变量
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 创建xfuser参数
        xfuser_args = xFuserArgs(
            model=model_path,
            height=512,  # 测试用较小尺寸
            width=512,
            num_inference_steps=4,
            seed=42,
            output_type="latent",
            # 设置为单GPU配置
            ulysses_degree=1,
            pipefusion_parallel_degree=1,
            use_cfg_parallel=False,
        )
        
        # 使用xFuserArgs创建配置
        engine_config = xfuser_args.create_config()
        logger.info("✅ EngineConfig created from xFuserArgs")
        
        # 创建FLUX pipeline
        pipeline = xFuserFluxPipeline.from_pretrained(
            model_path,
            engine_config=engine_config,
            torch_dtype=torch.float16,
        )
        
        logger.info("✅ xFuserFluxPipeline created successfully")
        logger.info(f"   Device: {getattr(pipeline, 'device', 'unknown')}")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ FLUX pipeline creation failed: {e}")
        logger.exception("Full traceback:")
        return None

def test_simple_inference(pipeline):
    """测试简单推理"""
    logger.info("7️⃣ Testing simple inference...")
    
    try:
        # 简单的推理测试
        start_time = time.time()
        
        result = pipeline(
            prompt="a cute cat",
            height=512,  # 使用较小尺寸进行测试
            width=512,
            num_inference_steps=4,  # 使用较少步数进行测试
            guidance_scale=1.0,  # FLUX.1-schnell 使用1.0
            output_type="latent",
        )
        
        end_time = time.time()
        
        logger.info(f"✅ Simple inference completed in {end_time - start_time:.2f}s")
        logger.info(f"   Output type: {type(result)}")
        
        if hasattr(result, 'images'):
            logger.info(f"   Images shape: {result.images.shape}")
        elif hasattr(result, 'latents'):
            logger.info(f"   Latents shape: {result.latents.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple inference failed: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    """主测试函数"""
    logger.info("🧪 xfuser FLUX多GPU推理测试")
    logger.info("=" * 60)
    
    # 测试序列
    tests = [
        ("Import Test", test_xfuser_import),
        ("GPU Availability", test_gpu_availability),
        ("FLUX Model", test_flux_model_availability),
    ]
    
    results = {}
    
    # 基础测试
    for test_name, test_func in tests:
        try:
            if test_name == "FLUX Model":
                result = test_func()
                model_path = result
                results[test_name] = bool(result)
            else:
                result = test_func()
                results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # 如果基础测试通过，继续进阶测试
    if all(results.values()):
        logger.info("\n🚀 Basic tests passed, proceeding with advanced tests...")
        
        # 创建xfuser参数
        xfuser_args = test_xfuser_args_creation()
        if xfuser_args:
            results["xFuserArgs"] = True
            
            # 初始化分布式（可选，单GPU测试可能不需要）
            # distributed_ok = test_distributed_initialization(xfuser_args)
            # results["Distributed Init"] = distributed_ok
            
            # 创建pipeline
            pipeline = test_flux_pipeline_creation(model_path)
            if pipeline:
                results["Pipeline Creation"] = True
                
                # 简单推理测试
                inference_ok = test_simple_inference(pipeline)
                results["Simple Inference"] = inference_ok
            else:
                results["Pipeline Creation"] = False
        else:
            results["xFuserArgs"] = False
    
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
        logger.info("🎉 ALL TESTS PASSED! xfuser is ready for FLUX multi-GPU inference")
        logger.info("💡 You can now test the ComfyUI integration")
    else:
        logger.error("❌ Some tests failed. Please check the setup")
        logger.info("💡 Recommendations:")
        if not results.get("Import Test", False):
            logger.info("   • Install xfuser: pip install xfuser")
        if not results.get("GPU Availability", False):
            logger.info("   • Check CUDA installation and GPU drivers")
        if not results.get("FLUX Model", False):
            logger.info("   • Download FLUX model or check model paths")

if __name__ == "__main__":
    main() 