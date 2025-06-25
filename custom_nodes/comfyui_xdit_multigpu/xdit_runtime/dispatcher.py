"""
xDiT Dispatcher
==============

Worker scheduling strategies for multi-GPU inference.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .worker import XDiTWorker, XDiTWorkerFallback
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
        
        logger.info(f"Initializing XDiT Dispatcher with {len(gpu_devices)} GPUs")
        logger.info(f"Scheduling strategy: {scheduling_strategy.value}")
    
    def initialize(self) -> bool:
        """Initialize all workers"""
        try:
            if RAY_AVAILABLE:
                # Initialize Ray if not already done
                if not is_ray_available():
                    success = initialize_ray()
                    if not success:
                        logger.error("Failed to initialize Ray")
                        return False
                    logger.info("Ray initialized successfully")
                
                # Create Ray workers
                for gpu_id in self.gpu_devices:
                    worker = XDiTWorker.remote(gpu_id, self.model_path, self.strategy)
                    self.workers[gpu_id] = worker
                    self.worker_loads[gpu_id] = 0
                    
                    # Initialize worker
                    success = ray.get(worker.initialize.remote())
                    if not success:
                        logger.error(f"Failed to initialize Ray worker on GPU {gpu_id}")
                        return False
                    
                    logger.info(f"✅ Ray worker initialized on GPU {gpu_id}")
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
                     prompt: str,
                     negative_prompt: str = "",
                     height: int = 512,
                     width: int = 512,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[Any]:
        """Run inference using the dispatcher"""
        try:
            if not self.is_initialized:
                logger.error("Dispatcher not initialized")
                return None
            
            # Get next available worker
            worker = self.get_next_worker()
            if worker is None:
                logger.error("No available workers")
                return None
            
            # Run inference
            if RAY_AVAILABLE:
                # Use Ray remote call
                result_ref = worker.run_inference.remote(
                    prompt, negative_prompt, height, width,
                    num_inference_steps, guidance_scale, seed
                )
                result = ray.get(result_ref)
            else:
                # Use direct call
                result = worker.run_inference(
                    prompt, negative_prompt, height, width,
                    num_inference_steps, guidance_scale, seed
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