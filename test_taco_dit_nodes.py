#!/usr/bin/env python3
"""
TACO-DiT 节点测试脚本

测试TACO-DiT节点是否正常工作
"""

import sys
import os
import logging

# 添加ComfyUI路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_taco_dit_availability():
    """测试TACO-DiT是否可用"""
    logger.info("Testing TACO-DiT availability...")
    
    try:
        # 测试xDit导入
        import xfuser
        logger.info("✅ xDit imported successfully")
        
        # 测试TACO-DiT模块导入
        from comfy.taco_dit import TACODiTConfigManager
        logger.info("✅ TACO-DiT modules imported successfully")
        
        # 测试配置管理器
        config_manager = TACODiTConfigManager()
        config = config_manager.config
        logger.info(f"✅ TACO-DiT config: {config}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_custom_nodes():
    """测试自定义节点"""
    logger.info("Testing custom nodes...")
    
    try:
        # 测试节点文件是否存在
        nodes_file = "custom_nodes/TACO_DiT_Enhanced_Nodes.py"
        if not os.path.exists(nodes_file):
            logger.error(f"❌ Nodes file not found: {nodes_file}")
            return False
        
        logger.info("✅ TACO-DiT nodes file exists")
        
        # 测试节点导入
        import importlib.util
        spec = importlib.util.spec_from_file_location("taco_dit_nodes", nodes_file)
        nodes_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodes_module)
        
        logger.info("✅ TACO-DiT nodes imported successfully")
        
        # 检查节点映射
        if hasattr(nodes_module, 'NODE_CLASS_MAPPINGS'):
            node_count = len(nodes_module.NODE_CLASS_MAPPINGS)
            logger.info(f"✅ Found {node_count} TACO-DiT nodes")
            
            for node_name in nodes_module.NODE_CLASS_MAPPINGS.keys():
                logger.info(f"  - {node_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Custom nodes test failed: {e}")
        return False

def test_gpu_configuration():
    """测试GPU配置"""
    logger.info("Testing GPU configuration...")
    
    try:
        import torch
        
        # 检查CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
        
        # 检查GPU数量
        gpu_count = torch.cuda.device_count()
        logger.info(f"Available GPUs: {gpu_count}")
        
        if gpu_count > 0:
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"GPU {i}: {props.name}, {memory_gb:.1f} GB")
        
        # 检查是否配置了3张GPU
        if cuda_devices == '3,4,5':
            logger.info("✅ GPU configuration correct (using GPUs 3,4,5)")
        else:
            logger.warning(f"⚠️  GPU configuration may not be optimal: {cuda_devices}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU configuration test failed: {e}")
        return False

def test_model_patcher():
    """测试模型包装器"""
    logger.info("Testing model patcher...")
    
    try:
        from comfy.taco_dit import TACODiTModelPatcher, TACODiTConfig
        
        # 创建测试配置
        config = TACODiTConfig(
            enabled=True,
            sequence_parallel=True,
            sequence_parallel_degree=3,
            cfg_parallel=True,
            use_flash_attention=True
        )
        
        # 创建模拟模型
        class MockModel:
            def __init__(self):
                self.name = "MockDiTModel"
            
            def forward(self, **kwargs):
                import torch
                return torch.randn(1, 3, 1024, 1024)
        
        model = MockModel()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型包装器
        model_patcher = TACODiTModelPatcher(
            model=model,
            load_device=device,
            offload_device='cpu',
            parallel_config=config
        )
        
        # 获取模型信息
        model_info = model_patcher.get_model_info()
        logger.info(f"✅ Model patcher created successfully")
        logger.info(f"  - Parallel enabled: {model_info['parallel_enabled']}")
        logger.info(f"  - Model type: {model_info['model_type']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model patcher test failed: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("="*60)
    logger.info("🧪 TACO-DiT Nodes Test Suite")
    logger.info("="*60)
    
    tests = [
        ("TACO-DiT Availability", test_taco_dit_availability),
        ("Custom Nodes", test_custom_nodes),
        ("GPU Configuration", test_gpu_configuration),
        ("Model Patcher", test_model_patcher),
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
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TACO-DiT nodes are ready for use.")
        logger.info("\n📋 Next steps:")
        logger.info("1. Start ComfyUI: ./start_comfyui_gpu345.sh")
        logger.info("2. Open browser: http://0.0.0.0:12215")
        logger.info("3. Add TACO-DiT nodes to your workflow")
        logger.info("4. Check console for TACO-DiT logs")
    else:
        logger.warning("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 