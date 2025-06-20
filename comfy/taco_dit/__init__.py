"""
TACO-DiT: Tensor-parallel Accelerated ComfyUI with Distributed DiT

A multi-GPU parallel inference solution for ComfyUI using xDit technology.
"""

from .core.config_manager import TACODiTConfig, TACODiTConfigManager
from .core.distributed_manager import TACODiTDistributedManager
from .core.model_wrapper import TACODiTModelPatcher
from .core.execution_engine import TACODiTExecutionEngine

__version__ = "1.0.0"
__author__ = "TACO-DiT Team"

__all__ = [
    "TACODiTConfig",
    "TACODiTConfigManager", 
    "TACODiTDistributedManager",
    "TACODiTModelPatcher",
    "TACODiTExecutionEngine",
] 