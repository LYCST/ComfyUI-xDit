"""
xDiT Dispatcher
==============

Worker scheduling strategies for multi-GPU inference.
"""

import time
import logging
import threading
import os
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import torch

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .worker import XDiTWorker, XDiTWorkerFallback, find_free_port
from .ray_manager import initialize_ray, is_ray_available

logger = logging.getLogger(__name__)

class SchedulingStrategy(Enum):
    """Scheduling strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    ADAPTIVE = "adaptive"

class XDiTDispatcher:
    """
    Dispatcher for managing multiple xDiT workers.
    Supports different scheduling strategies.
    """
    
    def __init__(self, gpu_devices: List[int], model_path: str, strategy: str = "Hybrid", 
                 scheduling_strategy: SchedulingStrategy = SchedulingStrategy.ROUND_ROBIN):
        self.gpu_devices = gpu_devices
        self.model_path = model_path
        self.strategy = strategy
        self.scheduling_strategy = scheduling_strategy
        self.workers = {}
        self.current_worker_index = 0
        self.worker_loads = {}
        self.lock = threading.Lock()
        self.is_initialized = False
        
        # 分布式配置
        self.master_addr = "127.0.0.1"
        self.master_port = find_free_port()
        self.world_size = len(gpu_devices)
        
        logger.info(f"Initializing XDiT Dispatcher with {len(gpu_devices)} GPUs")
        logger.info(f"Scheduling strategy: {scheduling_strategy.value}")
        logger.info(f"Distributed config: {self.master_addr}:{self.master_port}, world_size={self.world_size}")
    
    def initialize(self) -> bool:
        """Initialize all workers with coordinated distributed setup"""
        try:
            if RAY_AVAILABLE:
                # Initialize Ray if not already done
                if not is_ray_available():
                    success = initialize_ray()
                    if not success:
                        logger.error("Failed to initialize Ray")
                        return False
                    logger.info("Ray initialized successfully")
                
                # Create Ray workers with proper distributed parameters
                worker_init_futures = []
                
                for i, gpu_id in enumerate(self.gpu_devices):
                    worker = XDiTWorker.remote(
                        gpu_id=gpu_id, 
                        model_path=self.model_path, 
                        strategy=self.strategy,
                        master_addr=self.master_addr,
                        master_port=self.master_port,
                        world_size=self.world_size,
                        rank=i  # 使用索引作为rank
                    )
                    self.workers[gpu_id] = worker
                    self.worker_loads[gpu_id] = 0
                    
                    # Initialize worker (basic setup)
                    future = worker.initialize.remote()
                    worker_init_futures.append((gpu_id, future))
                
                # Wait for all workers to complete basic initialization
                logger.info("Waiting for all workers to complete basic initialization...")
                for gpu_id, future in worker_init_futures:
                    try:
                        success = ray.get(future, timeout=60)
                        if not success:
                            logger.error(f"Failed to initialize Ray worker on GPU {gpu_id}")
                            return False
                        logger.info(f"✅ Ray worker initialized on GPU {gpu_id}")
                    except Exception as e:
                        logger.error(f"Worker initialization failed on GPU {gpu_id}: {e}")
                        return False
                
                # Now initialize distributed environment for multi-GPU
                if self.world_size > 1:
                    logger.info("Initializing distributed environment for multi-GPU...")
                    distributed_futures = []
                    
                    for gpu_id, worker in self.workers.items():
                        future = worker.initialize_distributed.remote()
                        distributed_futures.append((gpu_id, future))
                    
                    # Wait for distributed initialization
                    distributed_success = True
                    for gpu_id, future in distributed_futures:
                        try:
                            success = ray.get(future, timeout=300)  # 5分钟超时
                            if not success:
                                logger.error(f"Failed to initialize distributed on GPU {gpu_id}")
                                distributed_success = False
                            else:
                                logger.info(f"✅ Distributed initialized on GPU {gpu_id}")
                        except Exception as e:
                            logger.error(f"Distributed initialization failed on GPU {gpu_id}: {e}")
                            distributed_success = False
                    
                    if not distributed_success:
                        logger.warning("Some workers failed distributed initialization, falling back to single-GPU mode")
                        # 不返回False，而是继续，让系统回退到单GPU
                else:
                    logger.info("Single GPU mode, skipping distributed initialization")
                    
            else:
                # Use fallback workers
                logger.warning("Ray not available, using fallback workers")
                for gpu_id in self.gpu_devices:
                    worker = XDiTWorkerFallback(gpu_id, self.model_path, self.strategy)
                    self.workers[gpu_id] = worker
                    self.worker_loads[gpu_id] = 0
                    
                    # Initialize worker
                    success = worker.initialize()
                    if not success:
                        logger.error(f"Failed to initialize fallback worker on GPU {gpu_id}")
                        return False
                    
                    logger.info(f"✅ Fallback worker initialized on GPU {gpu_id}")
            
            self.is_initialized = True
            logger.info(f"✅ XDiT Dispatcher initialized with {len(self.workers)} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize XDiT Dispatcher: {e}")
            logger.exception("Dispatcher initialization traceback:")
            return False
    
    def get_next_worker(self) -> Optional[Any]:
        """Get next worker based on scheduling strategy"""
        if not self.is_initialized or not self.workers:
            return None
        
        with self.lock:
            if self.scheduling_strategy == SchedulingStrategy.ROUND_ROBIN:
                return self._round_robin_schedule()
            elif self.scheduling_strategy == SchedulingStrategy.LEAST_LOADED:
                return self._least_loaded_schedule()
            elif self.scheduling_strategy == SchedulingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_schedule()
            elif self.scheduling_strategy == SchedulingStrategy.ADAPTIVE:
                return self._adaptive_schedule()
            else:
                # Default to round robin
                return self._round_robin_schedule()
    
    def _round_robin_schedule(self) -> Optional[Any]:
        """Round robin scheduling"""
        worker_ids = list(self.workers.keys())
        if not worker_ids:
            return None
        
        worker_id = worker_ids[self.current_worker_index % len(worker_ids)]
        self.current_worker_index += 1
        
        worker = self.workers[worker_id]
        self.worker_loads[worker_id] += 1
        
        logger.debug(f"Round robin: assigned to GPU {worker_id}")
        return worker
    
    def _least_loaded_schedule(self) -> Optional[Any]:
        """Least loaded scheduling"""
        if not self.worker_loads:
            return None
        
        # Find worker with minimum load
        min_load = min(self.worker_loads.values())
        candidates = [gpu_id for gpu_id, load in self.worker_loads.items() if load == min_load]
        
        # If multiple candidates, choose the first one
        worker_id = candidates[0]
        worker = self.workers[worker_id]
        self.worker_loads[worker_id] += 1
        
        logger.debug(f"Least loaded: assigned to GPU {worker_id} (load: {min_load})")
        return worker
    
    def _weighted_round_robin_schedule(self) -> Optional[Any]:
        """Weighted round robin scheduling based on GPU memory"""
        worker_ids = list(self.workers.keys())
        if not worker_ids:
            return None
        
        # Get GPU memory info to determine weights
        weights = []
        for gpu_id in worker_ids:
            try:
                if RAY_AVAILABLE:
                    gpu_info = ray.get(self.workers[gpu_id].get_gpu_info.remote())
                else:
                    gpu_info = self.workers[gpu_id].get_gpu_info()
                
                # Weight based on available memory
                memory_weight = gpu_info.get('memory_free_gb', 0) / gpu_info.get('memory_total_gb', 1)
                weights.append(memory_weight)
            except Exception as e:
                logger.warning(f"Error getting GPU info for {gpu_id}: {e}")
                weights.append(1.0)  # Default weight
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(worker_ids)] * len(worker_ids)
        
        # Choose worker based on weights
        import random
        worker_id = random.choices(worker_ids, weights=weights)[0]
        worker = self.workers[worker_id]
        self.worker_loads[worker_id] += 1
        
        logger.debug(f"Weighted round robin: assigned to GPU {worker_id}")
        return worker
    
    def _adaptive_schedule(self) -> Optional[Any]:
        """Adaptive scheduling based on current load and performance"""
        if not self.worker_loads:
            return None
        
        # Get current GPU info for all workers
        gpu_infos = {}
        for gpu_id, worker in self.workers.items():
            try:
                if RAY_AVAILABLE:
                    gpu_info = ray.get(worker.get_gpu_info.remote())
                else:
                    gpu_info = worker.get_gpu_info()
                gpu_infos[gpu_id] = gpu_info
            except Exception as e:
                logger.warning(f"Error getting GPU info for {gpu_id}: {e}")
                continue
        
        # Calculate adaptive score for each worker
        scores = {}
        for gpu_id, gpu_info in gpu_infos.items():
            if 'error' in gpu_info:
                continue
            
            # Score based on multiple factors
            memory_score = gpu_info.get('memory_free_gb', 0) / gpu_info.get('memory_total_gb', 1)
            load_score = 1.0 / (self.worker_loads.get(gpu_id, 0) + 1)  # Lower load = higher score
            utilization_score = 1.0 - (gpu_info.get('memory_allocated_gb', 0) / gpu_info.get('memory_total_gb', 1))
            
            # Combined score
            total_score = memory_score * 0.4 + load_score * 0.3 + utilization_score * 0.3
            scores[gpu_id] = total_score
        
        if not scores:
            return self._round_robin_schedule()
        
        # Choose worker with highest score
        best_gpu_id = max(scores.keys(), key=lambda x: scores[x])
        worker = self.workers[best_gpu_id]
        self.worker_loads[best_gpu_id] += 1
        
        logger.debug(f"Adaptive: assigned to GPU {best_gpu_id} (score: {scores[best_gpu_id]:.3f})")
        return worker
    
    def run_inference(self, 
                     model_state_dict: Dict,
                     conditioning_positive: Any,
                     conditioning_negative: Any,
                     latent_samples: torch.Tensor,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[torch.Tensor]:
        """Run inference using the dispatcher with ComfyUI model integration"""
        try:
            if not self.is_initialized:
                logger.error("Dispatcher not initialized")
                return None
            
            # Get next available worker
            worker = self.get_next_worker()
            if worker is None:
                logger.error("No available workers")
                return None
                
            # 🔧 修复模型路径处理：直接使用safetensors文件进行xDiT推理
            effective_model_path = self.model_path
            
            # 检查是否是safetensors文件
            if self.model_path.endswith('.safetensors'):
                logger.info(f"Using safetensors file for xDiT: {self.model_path}")
                # 对于xDiT，我们可以直接使用safetensors文件路径
                # xFuserFluxPipeline应该能够处理safetensors文件
                effective_model_path = self.model_path
                
                # 验证文件是否存在
                if not os.path.exists(effective_model_path):
                    logger.error(f"Model file not found: {effective_model_path}")
                    return None
                    
                logger.info(f"✅ Safetensors file verified: {effective_model_path}")
            else:
                # 如果是目录路径，验证diffusers格式
                if os.path.isdir(self.model_path):
                    model_index_path = os.path.join(self.model_path, "model_index.json")
                    if os.path.exists(model_index_path):
                        logger.info(f"✅ Diffusers directory verified: {self.model_path}")
                        effective_model_path = self.model_path
                    else:
                        logger.error(f"Invalid diffusers directory (no model_index.json): {self.model_path}")
                        return None
                else:
                    logger.error(f"Model path is neither safetensors file nor diffusers directory: {self.model_path}")
                    return None
            
            # 🔧 优化：传递轻量级模型信息而非完整state_dict
            # 提取模型基本信息
            model_info = {
                'type': 'flux' if 'flux' in effective_model_path.lower() else 'sd',  # 识别Flux模型
                'path': effective_model_path,  # 使用有效的模型路径
                'original_path': self.model_path,  # 保留原始路径
                'format': 'safetensors' if effective_model_path.endswith('.safetensors') else 'diffusers',
                'in_channels': 16 if 'flux' in effective_model_path.lower() else 4,  # Flux使用16通道
                'device': 'cuda',
                'dtype': 'torch.float16',
            }
            
            logger.info(f"Running inference with model: {model_info['type']}, format: {model_info['format']}")
            logger.info(f"Model path: {effective_model_path}")
            
            # Run inference with ComfyUI model data
            if RAY_AVAILABLE:
                # Use Ray remote call  
                result_ref = worker.run_inference.remote(
                    model_info=model_info,  # 传递轻量级模型信息
                    conditioning_positive=conditioning_positive,
                    conditioning_negative=conditioning_negative,
                    latent_samples=latent_samples,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                result = ray.get(result_ref)
            else:
                # Use direct call
                result = worker.run_inference(
                    model_info=model_info,  # 传递轻量级模型信息
                    conditioning_positive=conditioning_positive,
                    conditioning_negative=conditioning_negative,
                    latent_samples=latent_samples,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            
            # Decrease load count
            with self.lock:
                for gpu_id, w in self.workers.items():
                    if w == worker:
                        self.worker_loads[gpu_id] = max(0, self.worker_loads[gpu_id] - 1)
                        break
            
            return result
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get dispatcher status"""
        status = {
            "is_initialized": self.is_initialized,
            "num_workers": len(self.workers),
            "scheduling_strategy": self.scheduling_strategy.value,
            "worker_loads": self.worker_loads.copy(),
            "gpu_devices": self.gpu_devices
        }
        
        # Get detailed GPU info
        gpu_infos = {}
        for gpu_id, worker in self.workers.items():
            try:
                if RAY_AVAILABLE:
                    gpu_info = ray.get(worker.get_gpu_info.remote())
                else:
                    gpu_info = worker.get_gpu_info()
                gpu_infos[gpu_id] = gpu_info
            except Exception as e:
                gpu_infos[gpu_id] = {"error": str(e)}
        
        status["gpu_infos"] = gpu_infos
        return status
    
    def cleanup(self):
        """Clean up all workers"""
        try:
            logger.info("Cleaning up XDiT Dispatcher")
            
            for gpu_id, worker in self.workers.items():
                try:
                    if RAY_AVAILABLE:
                        ray.get(worker.cleanup.remote())
                    else:
                        worker.cleanup()
                    logger.info(f"✅ Worker on GPU {gpu_id} cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up worker on GPU {gpu_id}: {e}")
            
            self.workers.clear()
            self.worker_loads.clear()
            self.is_initialized = False
            
            # Shutdown Ray if we initialized it
            if RAY_AVAILABLE and ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown")
            
            logger.info("✅ XDiT Dispatcher cleaned up")
            
        except Exception as e:
            logger.error(f"Error during dispatcher cleanup: {e}") 