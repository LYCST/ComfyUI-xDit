"""
TACO-DiT Distributed Manager

Manages distributed environment initialization and coordination.
"""

import os
import logging
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TACODiTDistributedManager:
    """TACO-DiT分布式环境管理器"""
    
    def __init__(self):
        self.initialized = False
        self.world_size = 0
        self.rank = 0
        self.local_rank = 0
        self.backend = 'nccl'
        
    def initialize(self, config: 'TACODiTConfig') -> bool:
        """初始化分布式环境"""
        if self.initialized:
            logger.info("Distributed environment already initialized")
            return True
        
        try:
            # 从环境变量获取分布式配置
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.rank = int(os.environ.get('RANK', 0))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            logger.info(f"Initializing distributed environment: world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}")
            
            if self.world_size > 1:
                # 设置环境变量
                os.environ['MASTER_ADDR'] = config.master_addr
                os.environ['MASTER_PORT'] = str(config.master_port)
                
                # 初始化进程组
                dist.init_process_group(
                    backend=self.backend,
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )
                
                # 设置当前设备
                torch.cuda.set_device(self.local_rank)
                
                logger.info(f"Process group initialized successfully on rank {self.rank}")
                
                # 初始化xDit分布式环境（如果可用）
                self._init_xdit_distributed()
                
            else:
                logger.info("Single GPU mode, skipping distributed initialization")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed environment: {e}")
            return False
    
    def _init_xdit_distributed(self):
        """初始化xDit分布式环境"""
        try:
            # 尝试导入xDit
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel
            )
            
            # 初始化xDit分布式环境
            init_distributed_environment()
            initialize_model_parallel()
            
            logger.info("xDit distributed environment initialized")
            
        except ImportError:
            logger.warning("xDit not available, skipping xDit distributed initialization")
        except Exception as e:
            logger.error(f"Failed to initialize xDit distributed environment: {e}")
    
    def cleanup(self):
        """清理分布式环境"""
        if self.initialized and self.world_size > 1:
            try:
                dist.destroy_process_group()
                logger.info("Distributed process group destroyed")
            except Exception as e:
                logger.error(f"Failed to destroy process group: {e}")
        
        self.initialized = False
    
    def is_master(self) -> bool:
        """检查是否为master进程"""
        return self.rank == 0
    
    def is_last_rank(self) -> bool:
        """检查是否为最后一个rank"""
        return self.rank == self.world_size - 1
    
    def get_device(self) -> torch.device:
        """获取当前设备"""
        if self.world_size > 1:
            return torch.device(f'cuda:{self.local_rank}')
        else:
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def barrier(self):
        """同步所有进程"""
        if self.initialized and self.world_size > 1:
            dist.barrier()
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """广播张量"""
        if self.initialized and self.world_size > 1:
            dist.broadcast(tensor, src)
        return tensor
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """全局规约"""
        if self.initialized and self.world_size > 1:
            dist.all_reduce(tensor, op)
        return tensor
    
    def gather(self, tensor: torch.Tensor, dst: int = 0):
        """收集张量"""
        if self.initialized and self.world_size > 1:
            gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gathered, dst)
            return gathered
        else:
            return [tensor]
    
    def scatter(self, tensor_list: list, src: int = 0):
        """分发张量"""
        if self.initialized and self.world_size > 1:
            tensor = torch.zeros_like(tensor_list[0])
            dist.scatter(tensor, tensor_list, src)
            return tensor
        else:
            return tensor_list[0] if tensor_list else None
    
    def get_world_info(self) -> Dict[str, Any]:
        """获取分布式环境信息"""
        return {
            'initialized': self.initialized,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'backend': self.backend,
            'device': str(self.get_device()),
            'is_master': self.is_master(),
            'is_last_rank': self.is_last_rank()
        }
    
    def __str__(self) -> str:
        return f"TACODiTDistributedManager(world_size={self.world_size}, rank={self.rank}, initialized={self.initialized})"

# 全局分布式管理器实例
_distributed_manager = None

def get_distributed_manager() -> TACODiTDistributedManager:
    """获取全局分布式管理器实例"""
    global _distributed_manager
    if _distributed_manager is None:
        _distributed_manager = TACODiTDistributedManager()
    return _distributed_manager

def initialize_distributed(config: 'TACODiTConfig') -> bool:
    """初始化分布式环境"""
    manager = get_distributed_manager()
    return manager.initialize(config)

def cleanup_distributed():
    """清理分布式环境"""
    global _distributed_manager
    if _distributed_manager is not None:
        _distributed_manager.cleanup()
        _distributed_manager = None 