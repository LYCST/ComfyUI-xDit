"""
ComfyUI xDiT Multi-GPU Plugin Configuration
===========================================

Configuration settings for the multi-GPU acceleration plugin.
"""

import os
import torch
from typing import List, Dict, Any

class PluginConfig:
    """Plugin configuration class"""
    
    # Default settings
    DEFAULT_GPU_DEVICES = "0,1,2,3"
    DEFAULT_PARALLEL_STRATEGY = "Hybrid"
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_MEMORY_FRACTION = 0.8
    
    # xDiT specific settings
    XDIT_ENABLE_FLASH_ATTENTION = True
    XDIT_ENABLE_CACHE = True
    XDIT_TORCH_DTYPE = torch.float16
    
    # Performance settings
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"
    
    # GPU management
    MAX_GPU_MEMORY_FRACTION = 0.95
    MIN_GPU_MEMORY_GB = 8
    
    @classmethod
    def get_available_gpus(cls) -> List[int]:
        """Get list of available GPU devices"""
        if not torch.cuda.is_available():
            return []
        
        gpu_count = torch.cuda.device_count()
        available_gpus = []
        
        for i in range(gpu_count):
            try:
                # Check if GPU is accessible
                torch.cuda.get_device_properties(i)
                available_gpus.append(i)
            except Exception:
                continue
        
        return available_gpus
    
    @classmethod
    def get_gpu_memory_info(cls) -> Dict[int, Dict[str, Any]]:
        """Get memory information for all available GPUs"""
        gpu_info = {}
        
        for gpu_id in cls.get_available_gpus():
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                memory_total = props.total_memory / (1024**3)  # Convert to GB
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                
                gpu_info[gpu_id] = {
                    "name": props.name,
                    "memory_total_gb": memory_total,
                    "memory_reserved_gb": memory_reserved,
                    "memory_allocated_gb": memory_allocated,
                    "memory_free_gb": memory_total - memory_reserved,
                    "compute_capability": f"{props.major}.{props.minor}"
                }
            except Exception as e:
                print(f"Warning: Could not get info for GPU {gpu_id}: {e}")
        
        return gpu_info
    
    @classmethod
    def validate_gpu_config(cls, gpu_devices: str) -> bool:
        """Validate GPU device configuration"""
        try:
            gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
            available_gpus = cls.get_available_gpus()
            
            for gpu_id in gpu_list:
                if gpu_id not in available_gpus:
                    print(f"Warning: GPU {gpu_id} is not available")
                    return False
            
            return True
        except Exception:
            return False
    
    @classmethod
    def get_optimal_batch_size(cls, gpu_devices: str, model_size_gb: float = 2.0) -> int:
        """Calculate optimal batch size based on GPU configuration"""
        try:
            gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
            gpu_info = cls.get_gpu_memory_info()
            
            min_free_memory = float('inf')
            for gpu_id in gpu_list:
                if gpu_id in gpu_info:
                    free_memory = gpu_info[gpu_id]["memory_free_gb"]
                    min_free_memory = min(min_free_memory, free_memory)
            
            if min_free_memory == float('inf'):
                return cls.DEFAULT_BATCH_SIZE
            
            # Estimate batch size based on available memory
            # Assuming each sample needs approximately model_size_gb * 1.5 GB
            estimated_batch_size = int(min_free_memory / (model_size_gb * 1.5))
            
            return max(1, min(estimated_batch_size, 16))  # Clamp between 1 and 16
            
        except Exception:
            return cls.DEFAULT_BATCH_SIZE
    
    @classmethod
    def get_environment_vars(cls, gpu_devices: str, use_flash_attention: bool = True, use_cache: bool = True) -> Dict[str, str]:
        """Get environment variables for xDiT"""
        env_vars = {
            "CUDA_VISIBLE_DEVICES": gpu_devices,
        }
        
        if use_flash_attention:
            env_vars["XFUSER_ENABLE_FLASH_ATTENTION"] = "1"
        
        if use_cache:
            env_vars["XFUSER_ENABLE_CACHE"] = "1"
        
        return env_vars
    
    @classmethod
    def print_system_info(cls):
        """Print system information for debugging"""
        print("=== ComfyUI xDiT Multi-GPU Plugin System Info ===")
        
        # GPU information
        available_gpus = cls.get_available_gpus()
        print(f"Available GPUs: {available_gpus}")
        
        if available_gpus:
            gpu_info = cls.get_gpu_memory_info()
            for gpu_id, info in gpu_info.items():
                print(f"GPU {gpu_id}: {info['name']}")
                print(f"  Memory: {info['memory_total_gb']:.1f}GB total, {info['memory_free_gb']:.1f}GB free")
                print(f"  Compute Capability: {info['compute_capability']}")
        
        # PyTorch information
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
        
        print("=" * 50)

# Global configuration instance
config = PluginConfig() 