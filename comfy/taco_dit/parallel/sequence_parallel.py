"""
Sequence Parallel Manager

Manages sequence parallelism for distributed inference.
"""

import logging
import torch
from typing import Optional, Dict, Any
from ..core.distributed_manager import get_distributed_manager

logger = logging.getLogger(__name__)

class SequenceParallelManager:
    """序列并行管理器"""
    
    def __init__(self, parallel_degree: int = 1):
        self.parallel_degree = parallel_degree
        self.distributed_manager = get_distributed_manager()
        self.initialized = False
        
    def initialize(self) -> bool:
        """初始化序列并行"""
        if self.initialized:
            return True
            
        try:
            if self.parallel_degree > 1:
                # 初始化xDit序列并行
                self._init_xdit_sequence_parallel()
                logger.info(f"Sequence parallel initialized with degree {self.parallel_degree}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize sequence parallel: {e}")
            return False
    
    def _init_xdit_sequence_parallel(self):
        """初始化xDit序列并行"""
        try:
            from xfuser.core.distributed import (
                get_sequence_parallel_world_size,
                get_sequence_parallel_rank
            )
            
            # 检查序列并行配置
            sp_world_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            
            logger.info(f"xDit sequence parallel: world_size={sp_world_size}, rank={sp_rank}")
            
        except ImportError:
            logger.warning("xDit sequence parallel not available")
        except Exception as e:
            logger.error(f"Failed to initialize xDit sequence parallel: {e}")
    
    def split_sequence(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """分割序列维度"""
        if self.parallel_degree <= 1:
            return tensor
            
        try:
            # 计算每个进程处理的序列长度
            seq_len = tensor.size(dim)
            chunk_size = seq_len // self.parallel_degree
            
            # 获取当前rank
            rank = self.distributed_manager.rank
            
            # 计算当前进程的序列范围
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < self.parallel_degree - 1 else seq_len
            
            # 分割张量
            if dim == 0:
                return tensor[start_idx:end_idx]
            elif dim == 1:
                return tensor[:, start_idx:end_idx]
            elif dim == 2:
                return tensor[:, :, start_idx:end_idx]
            else:
                return tensor[:, :, :, start_idx:end_idx]
                
        except Exception as e:
            logger.error(f"Failed to split sequence: {e}")
            return tensor
    
    def gather_sequence(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """收集序列维度"""
        if self.parallel_degree <= 1:
            return tensor
            
        try:
            # 收集所有进程的张量
            gathered = self.distributed_manager.gather(tensor)
            
            # 在master进程上拼接
            if self.distributed_manager.is_master():
                return torch.cat(gathered, dim=dim)
            else:
                return tensor
                
        except Exception as e:
            logger.error(f"Failed to gather sequence: {e}")
            return tensor
    
    def forward_sequence_parallel(self, model, input_data: Dict[str, Any]) -> Any:
        """序列并行前向传播"""
        if self.parallel_degree <= 1:
            return model(**input_data)
        
        try:
            # 分割输入序列
            if 'x' in input_data:
                input_data['x'] = self.split_sequence(input_data['x'])
            
            # 执行前向传播
            output = model(**input_data)
            
            # 收集输出序列
            if isinstance(output, torch.Tensor):
                output = self.gather_sequence(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Sequence parallel forward failed: {e}")
            # 回退到原始执行
            return model(**input_data)
    
    def __str__(self) -> str:
        return f"SequenceParallelManager(degree={self.parallel_degree}, initialized={self.initialized})" 