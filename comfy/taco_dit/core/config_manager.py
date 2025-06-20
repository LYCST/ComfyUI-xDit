"""
TACO-DiT Configuration Manager

Manages parallel inference configuration and auto-detection.
"""

import os
import yaml
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class TACODiTConfig:
    """TACO-DiT configuration class"""
    
    # 基本开关
    enabled: bool = True
    auto_detect: bool = True
    
    # 并行策略配置
    sequence_parallel: bool = False
    pipeline_parallel: bool = False
    tensor_parallel: bool = False
    cfg_parallel: bool = True  # 默认启用CFG并行
    
    # 并行度配置
    sequence_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    tensor_parallel_degree: int = 1
    
    # 性能优化配置
    use_flash_attention: bool = True
    use_torch_compile: bool = False
    use_cache: bool = True
    use_teacache: bool = False
    use_fbcache: bool = False
    
    # 内存管理配置
    enable_offload: bool = False
    offload_device: str = 'cpu'
    enable_sequential_cpu_offload: bool = False
    
    # 分布式配置
    backend: str = 'nccl'
    master_addr: str = 'localhost'
    master_port: int = 29500
    
    # 调试配置
    debug: bool = False
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """验证配置参数"""
        if self.sequence_parallel_degree < 1:
            raise ValueError("sequence_parallel_degree must be >= 1")
        if self.pipeline_parallel_degree < 1:
            raise ValueError("pipeline_parallel_degree must be >= 1")
        if self.tensor_parallel_degree < 1:
            raise ValueError("tensor_parallel_degree must be >= 1")

class TACODiTConfigManager:
    """TACO-DiT配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = TACODiTConfig()
        self.config_path = config_path
        
        # 加载配置文件
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # 自动检测配置
        if self.config.auto_detect:
            self.auto_detect_config()
    
    def _get_default_config(self) -> TACODiTConfig:
        """获取默认配置"""
        return TACODiTConfig(
            enabled=False,
            auto_detect=True
        )
    
    def auto_detect_config(self) -> TACODiTConfig:
        """自动检测最优配置"""
        try:
            # 检测GPU数量
            gpu_count = torch.cuda.device_count()
            logger.info(f"Detected {gpu_count} GPUs")
            
            # 针对3张GPU的优化配置
            if gpu_count >= 3:
                # 使用3张GPU的配置：序列并行度3
                config = TACODiTConfig(
                    enabled=True,
                    sequence_parallel=True,
                    sequence_parallel_degree=3,  # 3张GPU全部用于序列并行
                    pipeline_parallel=False,     # 不使用流水线并行
                    pipeline_parallel_degree=1,
                    tensor_parallel=False,       # 不使用张量并行
                    tensor_parallel_degree=1,
                    cfg_parallel=True,           # 启用CFG并行
                    use_flash_attention=True,    # 启用Flash Attention
                    use_cache=True,              # 启用缓存
                    auto_detect=True
                )
                logger.info("Auto-detected config for 3 GPUs: sequence_parallel_degree=3")
            elif gpu_count == 2:
                # 2张GPU配置
                config = TACODiTConfig(
                    enabled=True,
                    sequence_parallel=True,
                    sequence_parallel_degree=2,
                    pipeline_parallel=False,
                    pipeline_parallel_degree=1,
                    tensor_parallel=False,
                    tensor_parallel_degree=1,
                    cfg_parallel=True,
                    use_flash_attention=True,
                    use_cache=True,
                    auto_detect=True
                )
                logger.info("Auto-detected config for 2 GPUs: sequence_parallel_degree=2")
            elif gpu_count == 1:
                # 单GPU配置
                config = TACODiTConfig(
                    enabled=True,
                    sequence_parallel=False,
                    sequence_parallel_degree=1,
                    pipeline_parallel=False,
                    pipeline_parallel_degree=1,
                    tensor_parallel=False,
                    tensor_parallel_degree=1,
                    cfg_parallel=True,
                    use_flash_attention=True,
                    use_cache=True,
                    auto_detect=True
                )
                logger.info("Auto-detected config for 1 GPU: single GPU mode")
            else:
                # 无GPU配置
                config = TACODiTConfig(
                    enabled=False,
                    auto_detect=True
                )
                logger.warning("No GPU detected, TACO-DiT disabled")
            
            self.config = config
            return config
            
        except Exception as e:
            logger.error(f"Auto-detection failed: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def load_from_file(self, config_path: str) -> TACODiTConfig:
        """从配置文件加载配置"""
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 更新配置
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
        
        return self.config
    
    def save_to_file(self, config_path: str) -> None:
        """保存配置到文件"""
        logger.info(f"Saving configuration to {config_path}")
        
        try:
            config_data = {
                key: getattr(self.config, key) 
                for key in self.config.__dataclass_fields__.keys()
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_parallel_info(self) -> str:
        """获取并行配置信息字符串"""
        if not self.config.enabled:
            return "TACO-DiT disabled"
        
        parts = []
        if self.config.sequence_parallel:
            parts.append(f"SP{self.config.sequence_parallel_degree}")
        if self.config.pipeline_parallel:
            parts.append(f"PP{self.config.pipeline_parallel_degree}")
        if self.config.tensor_parallel:
            parts.append(f"TP{self.config.tensor_parallel_degree}")
        if self.config.cfg_parallel:
            parts.append("CFG")
        
        return "_".join(parts) if parts else "Single GPU"
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查并行度乘积是否合理
            total_parallel = (
                self.config.sequence_parallel_degree *
                self.config.pipeline_parallel_degree *
                self.config.tensor_parallel_degree
            )
            
            gpu_count = torch.cuda.device_count()
            if total_parallel > gpu_count:
                logger.warning(f"Total parallel degree ({total_parallel}) exceeds GPU count ({gpu_count})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def __str__(self) -> str:
        return f"TACODiTConfig(enabled={self.config.enabled}, parallel={self.get_parallel_info()})" 