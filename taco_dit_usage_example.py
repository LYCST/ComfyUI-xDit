#!/usr/bin/env python3
"""
TACO-DiT 使用示例

展示如何在ComfyUI中使用TACO-DiT进行多GPU并行推理
"""

import sys
import os
import logging
import torch

# 添加ComfyUI路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_auto_config():
    """示例1: 自动配置模式"""
    logger.info("="*60)
    logger.info("示例1: 自动配置模式")
    logger.info("="*60)
    
    from comfy.taco_dit import TACODiTConfigManager
    
    # 创建配置管理器，自动检测最优配置
    config_manager = TACODiTConfigManager()
    
    # 获取自动检测的配置
    config = config_manager.config
    
    logger.info(f"自动检测的配置:")
    logger.info(f"  - 启用状态: {config.enabled}")
    logger.info(f"  - 序列并行: {config.sequence_parallel} (度: {config.sequence_parallel_degree})")
    logger.info(f"  - 流水线并行: {config.pipeline_parallel} (度: {config.pipeline_parallel_degree})")
    logger.info(f"  - 张量并行: {config.tensor_parallel} (度: {config.tensor_parallel_degree})")
    logger.info(f"  - CFG并行: {config.cfg_parallel}")
    logger.info(f"  - Flash Attention: {config.use_flash_attention}")
    
    # 获取并行信息字符串
    parallel_info = config_manager.get_parallel_info()
    logger.info(f"并行配置: {parallel_info}")
    
    return config

def example_2_manual_config():
    """示例2: 手动配置模式"""
    logger.info("="*60)
    logger.info("示例2: 手动配置模式")
    logger.info("="*60)
    
    from comfy.taco_dit import TACODiTConfig
    
    # 手动创建配置
    config = TACODiTConfig(
        enabled=True,
        sequence_parallel=True,
        sequence_parallel_degree=4,  # 使用4个GPU进行序列并行
        pipeline_parallel=True,
        pipeline_parallel_degree=2,  # 使用2个GPU进行流水线并行
        cfg_parallel=True,
        use_flash_attention=True,
        use_cache=True
    )
    
    logger.info(f"手动配置:")
    logger.info(f"  - 总并行度: {config.sequence_parallel_degree * config.pipeline_parallel_degree}")
    logger.info(f"  - 序列并行: {config.sequence_parallel_degree}")
    logger.info(f"  - 流水线并行: {config.pipeline_parallel_degree}")
    
    return config

def example_3_distributed_setup():
    """示例3: 分布式环境设置"""
    logger.info("="*60)
    logger.info("示例3: 分布式环境设置")
    logger.info("="*60)
    
    from comfy.taco_dit import TACODiTDistributedManager, TACODiTConfig
    
    # 创建配置
    config = TACODiTConfig(enabled=True, auto_detect=False)
    
    # 创建分布式管理器
    dist_manager = TACODiTDistributedManager()
    
    # 初始化分布式环境
    success = dist_manager.initialize(config)
    logger.info(f"分布式环境初始化: {'成功' if success else '失败'}")
    
    if success:
        # 获取分布式信息
        world_info = dist_manager.get_world_info()
        logger.info(f"分布式信息: {world_info}")
        
        # 获取当前设备
        device = dist_manager.get_device()
        logger.info(f"当前设备: {device}")
    
    return dist_manager

def example_4_execution_engine():
    """示例4: 执行引擎使用"""
    logger.info("="*60)
    logger.info("示例4: 执行引擎使用")
    logger.info("="*60)
    
    from comfy.taco_dit import TACODiTExecutionEngine, TACODiTConfig
    
    # 创建配置
    config = TACODiTConfig(
        enabled=True,
        sequence_parallel=True,
        sequence_parallel_degree=2,
        cfg_parallel=True
    )
    
    # 创建执行引擎
    engine = TACODiTExecutionEngine(config)
    
    # 获取执行信息
    exec_info = engine.get_execution_info()
    logger.info(f"执行引擎信息:")
    logger.info(f"  - 并行配置: {exec_info['parallel_config']}")
    logger.info(f"  - xDit可用: {exec_info['xdit_available']}")
    logger.info(f"  - 分布式信息: {exec_info['distributed_info']}")
    
    return engine

def example_5_model_wrapper():
    """示例5: 模型包装器使用"""
    logger.info("="*60)
    logger.info("示例5: 模型包装器使用")
    logger.info("="*60)
    
    from comfy.taco_dit import TACODiTModelPatcher, TACODiTConfig
    
    # 创建配置
    config = TACODiTConfig(
        enabled=True,
        sequence_parallel=True,
        sequence_parallel_degree=2
    )
    
    # 创建一个模拟模型（实际使用时会是真实的DiT模型）
    class MockModel:
        def __init__(self):
            self.name = "MockDiTModel"
        
        def forward(self, **kwargs):
            return torch.randn(1, 3, 1024, 1024)  # 模拟输出
    
    # 创建模型包装器
    model = MockModel()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_patcher = TACODiTModelPatcher(
        model=model,
        load_device=device,
        offload_device='cpu',
        parallel_config=config
    )
    
    # 获取模型信息
    model_info = model_patcher.get_model_info()
    logger.info(f"模型包装器信息:")
    logger.info(f"  - 模型类型: {model_info['model_type']}")
    logger.info(f"  - 并行启用: {model_info['parallel_enabled']}")
    logger.info(f"  - xDit包装器: {model_info['xdit_wrapper']}")
    
    return model_patcher

def example_6_performance_monitoring():
    """示例6: 性能监控"""
    logger.info("="*60)
    logger.info("示例6: 性能监控")
    logger.info("="*60)
    
    from comfy.taco_dit import TACODiTExecutionEngine, TACODiTConfig
    import time
    
    # 创建配置
    config = TACODiTConfig(enabled=True, sequence_parallel=True, sequence_parallel_degree=2)
    
    # 创建执行引擎
    engine = TACODiTExecutionEngine(config)
    
    # 模拟多次执行
    for i in range(3):
        logger.info(f"执行 {i+1}/3...")
        
        # 模拟输入数据
        input_data = {
            'batch_size': 1,
            'sequence_length': 256,
            'height': 1024,
            'width': 1024,
            'x': torch.randn(1, 3, 1024, 1024)
        }
        
        # 模拟模型
        class MockModel:
            def forward(self, **kwargs):
                time.sleep(0.1)  # 模拟推理时间
                return torch.randn(1, 3, 1024, 1024)
        
        model = MockModel()
        
        # 执行推理
        start_time = time.time()
        output = engine.execute(model, input_data)
        execution_time = time.time() - start_time
        
        logger.info(f"  执行时间: {execution_time:.3f}s")
    
    # 获取执行统计
    exec_info = engine.get_execution_info()
    stats = exec_info['execution_stats']
    
    logger.info(f"执行统计:")
    logger.info(f"  - 总执行次数: {stats['total_executions']}")
    logger.info(f"  - 平均执行时间: {stats['avg_time']:.3f}s")
    logger.info(f"  - 并行执行次数: {stats['parallel_executions']}")
    logger.info(f"  - 回退执行次数: {stats['fallback_executions']}")

def main():
    """主函数"""
    logger.info("🚀 TACO-DiT 使用示例")
    logger.info("="*60)
    
    try:
        # 运行所有示例
        config1 = example_1_auto_config()
        config2 = example_2_manual_config()
        dist_manager = example_3_distributed_setup()
        engine = example_4_execution_engine()
        model_patcher = example_5_model_wrapper()
        example_6_performance_monitoring()
        
        logger.info("="*60)
        logger.info("🎉 所有示例运行成功！")
        logger.info("="*60)
        logger.info("TACO-DiT 已准备就绪，可以开始使用多GPU并行推理了！")
        
        # 提供使用建议
        logger.info("\n📋 使用建议:")
        logger.info("1. 对于6张RTX 4090，建议使用序列并行度4 + 流水线并行度2")
        logger.info("2. 启用Flash Attention以获得最佳性能")
        logger.info("3. 使用CFG并行来加速分类器引导")
        logger.info("4. 监控GPU内存使用情况，避免OOM")
        
    except Exception as e:
        logger.error(f"❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 