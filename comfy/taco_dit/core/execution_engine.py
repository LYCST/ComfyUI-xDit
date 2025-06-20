"""
TACO-DiT Execution Engine

Manages parallel inference execution and coordination.
"""

import logging
import time
import torch
from typing import Optional, Dict, Any, Union
from .config_manager import TACODiTConfig
from .distributed_manager import get_distributed_manager

logger = logging.getLogger(__name__)

class TACODiTExecutionEngine:
    """TACO-DiT执行引擎，管理并行推理的执行流程"""
    
    def __init__(self, parallel_config: TACODiTConfig):
        self.parallel_config = parallel_config
        self.distributed_manager = get_distributed_manager()
        self.runtime_state = None
        self.execution_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'parallel_executions': 0,
            'fallback_executions': 0
        }
        
        # 初始化xDit运行时状态（如果可用）
        self._init_xdit_runtime()
    
    def _init_xdit_runtime(self):
        """初始化xDit运行时状态"""
        try:
            from xfuser.core.distributed import get_runtime_state
            self.runtime_state = get_runtime_state()
            logger.info("xDit runtime state initialized")
        except ImportError:
            logger.warning("xDit runtime state not available")
        except Exception as e:
            logger.error(f"Failed to initialize xDit runtime state: {e}")
    
    def prepare_execution(self, model, input_data: Dict[str, Any]) -> bool:
        """准备执行环境"""
        try:
            # 设置运行时状态（如果使用xDit）
            if self.runtime_state and self.parallel_config.enabled:
                self._setup_runtime_state(input_data)
            
            # 初始化模型并行
            if self.parallel_config.sequence_parallel:
                self._setup_sequence_parallel(model)
                
            if self.parallel_config.pipeline_parallel:
                self._setup_pipeline_parallel(model)
                
            if self.parallel_config.tensor_parallel:
                self._setup_tensor_parallel(model)
            
            logger.info("Execution environment prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare execution environment: {e}")
            return False
    
    def _setup_runtime_state(self, input_data: Dict[str, Any]):
        """设置xDit运行时状态"""
        if not self.runtime_state:
            return
        
        try:
            # 提取输入参数
            batch_size = input_data.get('batch_size', 1)
            sequence_length = input_data.get('sequence_length', 256)
            height = input_data.get('height', 1024)
            width = input_data.get('width', 1024)
            
            # 设置运行时参数
            self.runtime_state.set_input_parameters(
                batch_size=batch_size,
                sequence_length=sequence_length,
                height=height,
                width=width
            )
            
            logger.debug(f"Runtime state set: batch_size={batch_size}, seq_len={sequence_length}, h={height}, w={width}")
            
        except Exception as e:
            logger.error(f"Failed to setup runtime state: {e}")
    
    def _setup_sequence_parallel(self, model):
        """设置序列并行"""
        try:
            if hasattr(model, 'setup_sequence_parallel'):
                model.setup_sequence_parallel(self.parallel_config.sequence_parallel_degree)
                logger.info(f"Sequence parallel setup: degree={self.parallel_config.sequence_parallel_degree}")
        except Exception as e:
            logger.warning(f"Failed to setup sequence parallel: {e}")
    
    def _setup_pipeline_parallel(self, model):
        """设置流水线并行"""
        try:
            if hasattr(model, 'setup_pipeline_parallel'):
                model.setup_pipeline_parallel(self.parallel_config.pipeline_parallel_degree)
                logger.info(f"Pipeline parallel setup: degree={self.parallel_config.pipeline_parallel_degree}")
        except Exception as e:
            logger.warning(f"Failed to setup pipeline parallel: {e}")
    
    def _setup_tensor_parallel(self, model):
        """设置张量并行"""
        try:
            if hasattr(model, 'setup_tensor_parallel'):
                model.setup_tensor_parallel(self.parallel_config.tensor_parallel_degree)
                logger.info(f"Tensor parallel setup: degree={self.parallel_config.tensor_parallel_degree}")
        except Exception as e:
            logger.warning(f"Failed to setup tensor parallel: {e}")
    
    def execute(self, model, input_data: Dict[str, Any], **kwargs) -> Any:
        """执行并行推理"""
        start_time = time.time()
        
        try:
            # 准备执行环境
            if not self.prepare_execution(model, input_data):
                logger.warning("Failed to prepare execution, falling back to original")
                return self._fallback_execute(model, input_data, **kwargs)
            
            # 执行推理
            if self.parallel_config.enabled and hasattr(model, 'xdit_wrapper') and model.xdit_wrapper:
                # 使用xDit并行推理
                output = self._parallel_execute(model, input_data, **kwargs)
                self.execution_stats['parallel_executions'] += 1
            else:
                # 使用原始推理
                output = self._fallback_execute(model, input_data, **kwargs)
                self.execution_stats['fallback_executions'] += 1
            
            # 更新统计信息
            execution_time = time.time() - start_time
            self._update_stats(execution_time)
            
            logger.info(f"Execution completed in {execution_time:.2f}s")
            return output
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            # 回退到原始执行
            return self._fallback_execute(model, input_data, **kwargs)
        finally:
            # 清理资源
            self._cleanup_execution()
    
    def _parallel_execute(self, model, input_data: Dict[str, Any], **kwargs) -> Any:
        """并行推理执行"""
        try:
            # 使用xDit包装器进行并行推理
            if hasattr(model, 'xdit_wrapper') and model.xdit_wrapper:
                return model.xdit_wrapper.forward(**input_data, **kwargs)
            else:
                raise ValueError("No xDit wrapper available")
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise
    
    def _fallback_execute(self, model, input_data: Dict[str, Any], **kwargs) -> Any:
        """回退到原始执行"""
        try:
            # 使用模型的原始forward方法
            if hasattr(model, 'forward'):
                return model.forward(**input_data, **kwargs)
            else:
                # 尝试直接调用模型
                return model(**input_data, **kwargs)
        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            raise
    
    def _update_stats(self, execution_time: float):
        """更新执行统计信息"""
        self.execution_stats['total_executions'] += 1
        self.execution_stats['total_time'] += execution_time
        self.execution_stats['avg_time'] = self.execution_stats['total_time'] / self.execution_stats['total_executions']
    
    def _cleanup_execution(self):
        """清理执行资源"""
        try:
            # 清理xDit运行时状态
            if self.runtime_state:
                self.runtime_state.destroy_distributed_env()
            
            # 同步分布式进程
            if self.distributed_manager.initialized:
                self.distributed_manager.barrier()
                
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def get_execution_info(self) -> Dict[str, Any]:
        """获取执行信息"""
        info = {
            'parallel_config': {
                'enabled': self.parallel_config.enabled,
                'sequence_parallel': self.parallel_config.sequence_parallel,
                'pipeline_parallel': self.parallel_config.pipeline_parallel,
                'tensor_parallel': self.parallel_config.tensor_parallel,
                'cfg_parallel': self.parallel_config.cfg_parallel,
            },
            'execution_stats': self.execution_stats.copy(),
            'distributed_info': self.distributed_manager.get_world_info(),
            'xdit_available': self.runtime_state is not None
        }
        
        return info
    
    def reset_stats(self):
        """重置执行统计信息"""
        self.execution_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'parallel_executions': 0,
            'fallback_executions': 0
        }
        logger.info("Execution stats reset")
    
    def __str__(self) -> str:
        return f"TACODiTExecutionEngine(enabled={self.parallel_config.enabled}, stats={self.execution_stats})"

# 全局执行引擎实例
_execution_engine = None

def get_execution_engine(config: TACODiTConfig) -> TACODiTExecutionEngine:
    """获取全局执行引擎实例"""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = TACODiTExecutionEngine(config)
    return _execution_engine

def reset_execution_engine():
    """重置全局执行引擎实例"""
    global _execution_engine
    _execution_engine = None 