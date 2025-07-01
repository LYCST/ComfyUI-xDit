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
import numpy as np

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .worker import XDiTWorker, XDiTWorkerFallback, find_free_port
from .ray_manager import initialize_ray, is_ray_available
from .comfyui_model_wrapper import ComfyUIModelWrapper

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
        self.workers = []
        self.current_worker_index = 0
        self.worker_loads = {}
        self.lock = threading.Lock()
        self.is_initialized = False
        self.model_wrapper = ComfyUIModelWrapper(model_path)  # 使用自定义模型加载器
        self.pipeline = None
        
        # 分布式配置
        self.master_addr = "127.0.0.1"
        self.master_port = find_free_port()
        self.world_size = len(gpu_devices)
        
        logger.info(f"Initializing XDiT Dispatcher with {len(gpu_devices)} GPUs")
        logger.info(f"Scheduling strategy: {scheduling_strategy.value}")
        logger.info(f"Distributed config: {self.master_addr}:{self.master_port}, world_size={self.world_size}")

        # 初始化 ComfyUI 模型包装器
        if not self.model_wrapper.load_components():
            logger.error(f"Failed to initialize model wrapper for {model_path}")
            return

        self.pipeline = self.model_wrapper.get_pipeline()  # 获取组合后的 pipeline

        logger.info(f"Pipeline ready: {self.pipeline}")
    
    # def initialize(self) -> bool:
    #     """Initialize all workers with coordinated distributed setup"""
    #     try:
    #         if RAY_AVAILABLE:
    #             # Initialize Ray if not already done
    #             if not is_ray_available():
    #                 success = initialize_ray()
    #                 if not success:
    #                     logger.error("Failed to initialize Ray")
    #                     return False
    #                 logger.info("Ray initialized successfully")
                
    #             # Create Ray workers with proper distributed parameters
    #             worker_init_futures = []
                
    #             for i, gpu_id in enumerate(self.gpu_devices):
    #                 worker = XDiTWorker.remote(
    #                     gpu_id=gpu_id, 
    #                     model_path=self.model_path, 
    #                     strategy=self.strategy,
    #                     master_addr=self.master_addr,
    #                     master_port=self.master_port,
    #                     world_size=self.world_size,
    #                     rank=i  # 使用索引作为rank
    #                 )
    #                 self.workers[gpu_id] = worker
    #                 self.worker_loads[gpu_id] = 0
                    
    #                 # Initialize worker (basic setup)
    #                 future = worker.initialize.remote()
    #                 worker_init_futures.append((gpu_id, future))
                
    #             # Wait for all workers to complete basic initialization
    #             logger.info("Waiting for all workers to complete basic initialization...")
    #             for gpu_id, future in worker_init_futures:
    #                 try:
    #                     success = ray.get(future, timeout=60)
    #                     if not success:
    #                         logger.error(f"Failed to initialize Ray worker on GPU {gpu_id}")
    #                         return False
    #                     logger.info(f"✅ Ray worker initialized on GPU {gpu_id}")
    #                 except Exception as e:
    #                     logger.error(f"Worker initialization failed on GPU {gpu_id}: {e}")
    #                     return False
                
    #             # Now initialize distributed environment for multi-GPU
    #             if self.world_size > 1:
    #                 logger.info("Initializing distributed environment for multi-GPU...")
    #                 distributed_futures = []
                    
    #                 for gpu_id, worker in self.workers.items():
    #                     future = worker.initialize_distributed.remote()
    #                     distributed_futures.append((gpu_id, future))
                    
    #                 # Wait for distributed initialization
    #                 distributed_success = True
    #                 for gpu_id, future in distributed_futures:
    #                     try:
    #                         success = ray.get(future, timeout=300)  # 5分钟超时
    #                         if not success:
    #                             logger.error(f"Failed to initialize distributed on GPU {gpu_id}")
    #                             distributed_success = False
    #                         else:
    #                             logger.info(f"✅ Distributed initialized on GPU {gpu_id}")
    #                     except Exception as e:
    #                         logger.error(f"Distributed initialization failed on GPU {gpu_id}: {e}")
    #                         distributed_success = False
                    
    #                 if not distributed_success:
    #                     logger.warning("Some workers failed distributed initialization, falling back to single-GPU mode")
    #                     # 不返回False，而是继续，让系统回退到单GPU
    #             else:
    #                 logger.info("Single GPU mode, skipping distributed initialization")
                    
    #         else:
    #             # Use fallback workers
    #             logger.warning("Ray not available, using fallback workers")
    #             for gpu_id in self.gpu_devices:
    #                 worker = XDiTWorkerFallback(gpu_id, self.model_path, self.strategy)
    #                 self.workers[gpu_id] = worker
    #                 self.worker_loads[gpu_id] = 0
                    
    #                 # Initialize worker
    #                 success = worker.initialize()
    #                 if not success:
    #                     logger.error(f"Failed to initialize fallback worker on GPU {gpu_id}")
    #                     return False
                    
    #                 logger.info(f"✅ Fallback worker initialized on GPU {gpu_id}")
            
    #         self.is_initialized = True
    #         logger.info(f"✅ XDiT Dispatcher initialized with {len(self.workers)} workers")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Failed to initialize XDiT Dispatcher: {e}")
    #         logger.exception("Dispatcher initialization traceback:")
    #         return False
    
    def initialize(self):
        """Initialize workers"""
        try:
            logger.info(f"🚀 Initializing {len(self.gpu_devices)} workers...")
            
            if RAY_AVAILABLE:
                # 使用Ray Actor
                worker_futures = []

                for i, gpu_id in enumerate(self.gpu_devices):
                    try:
                         # 创建Ray worker
                        worker = XDiTWorker.remote(
                            gpu_id=gpu_id, 
                            model_path=self.model_path, 
                            strategy=self.strategy,
                            master_addr=self.master_addr,
                            master_port=self.master_port,
                            world_size=self.world_size,
                            rank=i
                        )
                        self.workers.append(worker)
                        self.worker_loads[gpu_id] = 0
                        
                        # 🔧 关键修复：调用worker的initialize方法
                        future = worker.initialize.remote()
                        worker_futures.append((gpu_id, worker, future))
                        
                        logger.info(f"📦 Created Ray worker for GPU {gpu_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to create worker for GPU {gpu_id}: {e}")
                        return False

                # 等待所有worker初始化完成
                logger.info("⏳ Waiting for all workers to initialize...")
                
                for gpu_id, worker, future in worker_futures:
                    try:
                        success = ray.get(future, timeout=60)
                        if success:
                            logger.info(f"✅ Worker {gpu_id} initialized successfully")
                        else:
                            logger.error(f"❌ Worker {gpu_id} initialization failed")
                            return False
                    except Exception as e:
                        logger.error(f"❌ Worker {gpu_id} initialization error: {e}")
                        return False
                
                logger.info(f"✅ All {len(self.workers)} Ray workers initialized")
            else:
                 # 使用fallback worker
                for gpu_id in self.gpu_devices:
                    try:
                        worker = XDiTWorkerFallback(gpu_id, self.model_path, self.strategy)
                        success = worker.initialize()
                        
                        if success:
                            self.workers.append(worker)
                            self.worker_loads[gpu_id] = 0
                            logger.info(f"✅ Fallback worker {gpu_id} initialized")
                        else:
                            logger.error(f"❌ Fallback worker {gpu_id} initialization failed")
                            return False
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to create fallback worker for GPU {gpu_id}: {e}")
                        return False
            
            self.is_initialized = True
            logger.info(f"✅ Initialized {len(self.workers)} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize workers: {e}")
            return False
    
    def _initialize_distributed(self, worker_actors) -> bool:
        """分布式环境初始化，带重试机制"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"分布式初始化尝试 {attempt + 1}/{max_retries}")
                
                # 确保所有workers使用相同的master配置
                logger.info(f"使用master配置: {self.master_addr}:{self.master_port}")
                
                # 启动分布式初始化
                dist_futures = []
                for gpu_id, worker in worker_actors.items():
                    future = worker.initialize_distributed.remote()
                    dist_futures.append((gpu_id, future))
                
                # 等待分布式初始化，延长超时时间
                timeout = 300 + attempt * 60  # 逐次增加超时时间
                success_count = 0
                
                for gpu_id, future in dist_futures:
                    try:
                        success = ray.get(future, timeout=timeout)
                        if success:
                            success_count += 1
                            logger.info(f"✅ GPU {gpu_id} 分布式初始化成功")
                        else:
                            logger.error(f"❌ GPU {gpu_id} 分布式初始化失败")
                    except Exception as e:
                        logger.error(f"❌ GPU {gpu_id} 分布式初始化异常: {e}")
                
                # 检查成功率
                if success_count == len(worker_actors):
                    logger.info("✅ 所有workers分布式初始化成功")
                    return True
                elif success_count >= len(worker_actors) // 2:
                    logger.warning(f"⚠️ 部分workers初始化成功 ({success_count}/{len(worker_actors)})")
                    return True
                else:
                    logger.error(f"❌ 大部分workers初始化失败 ({success_count}/{len(worker_actors)})")
                    if attempt < max_retries - 1:
                        logger.info("等待重试...")
                        time.sleep(5)
                        continue
                    else:
                        return False
                        
            except Exception as e:
                logger.error(f"分布式初始化尝试{attempt + 1}失败: {e}")
                if attempt == max_retries - 1:
                    return False
                time.sleep(5)
        
        return False
    
    def _initialize_fallback(self) -> bool:
        """Fallback初始化模式"""
        try:
            logger.info("使用fallback模式初始化")
            
            # 只使用第一个GPU
            primary_gpu = self.gpu_devices[0] if self.gpu_devices else 0
            
            worker = XDiTWorkerFallback(primary_gpu, self.model_path, self.strategy)
            success = worker.initialize()
            
            if success:
                self.workers = {primary_gpu: worker}
                self.worker_loads = {primary_gpu: 0}
                self.is_initialized = True
                self.world_size = 1  # 强制单GPU模式
                
                logger.info(f"✅ Fallback模式初始化成功，使用GPU {primary_gpu}")
                return True
            else:
                logger.error("❌ Fallback模式初始化也失败")
                return False
                
        except Exception as e:
            logger.error(f"Fallback初始化失败: {e}")
            return False
    
    def run_inference(self, model_info, conditioning_positive, conditioning_negative, 
                     latent_samples, num_inference_steps=20, guidance_scale=8.0, seed=42, 
                     comfyui_vae=None, comfyui_clip=None) -> Optional[torch.Tensor]:
        """改进的推理方法"""
        try:
            if not self.is_initialized or not self.workers:
                logger.error("Dispatcher未初始化")
                return None

            # 更新model_info以包含ComfyUI组件
            enhanced_model_info = model_info.copy()

            # 从方法参数获取VAE和CLIP
            actual_vae = comfyui_vae
            actual_clip = comfyui_clip
            
            # 如果参数中没有，尝试从model_info中获取
            if actual_vae is None:
                actual_vae = model_info.get('vae')
            if actual_clip is None:
                actual_clip = model_info.get('clip')


            enhanced_model_info.update({
                'vae': actual_vae,
                'clip': actual_clip,
                'comfyui_mode': True,
                'vae_available': actual_vae is not None,
                'clip_available': actual_clip is not None
            })

            
            # 🔧 处理从KSampler传来的序列化数据
            logger.info(f"🎯 运行推理: {num_inference_steps}步, CFG={guidance_scale}")
            logger.info(f"  • Workers: {len(self.workers)}")
            logger.info(f"  • VAE: {'✅' if actual_vae is not None else '❌'}")
            logger.info(f"  • CLIP: {'✅' if actual_clip is not None else '❌'}")

                        # 如果VAE和CLIP仍然为空，尝试调试
            if actual_vae is None or actual_clip is None:
                logger.warning("🔍 VAE/CLIP debugging:")
                logger.warning(f"  • comfyui_vae parameter: {type(comfyui_vae) if comfyui_vae else 'None'}")
                logger.warning(f"  • comfyui_clip parameter: {type(comfyui_clip) if comfyui_clip else 'None'}")
                logger.warning(f"  • model_info keys: {list(model_info.keys())}")
                logger.warning("  • Check your ComfyUI workflow connections!")
            
            # 🔧 将numpy数组转换回tensor（如果需要）
            try:
                if hasattr(latent_samples, 'numpy'):  # 检查是否是tensor
                    # 已经是tensor，不需要转换
                    pass
                elif hasattr(latent_samples, 'shape'):  # 可能是numpy数组
                    if isinstance(latent_samples, np.ndarray):
                        latent_samples = torch.from_numpy(latent_samples)
                        logger.info("🔧 Converted latent_samples back to tensor")
            except Exception as e:
                logger.warning(f"Warning: Could not process latent_samples: {e}")
            
            # 🔧 处理conditioning数据
            try:
                if conditioning_positive is not None and isinstance(conditioning_positive, list):
                    if len(conditioning_positive) > 0 and isinstance(conditioning_positive[0], np.ndarray):
                        conditioning_positive = [torch.from_numpy(p) for p in conditioning_positive]
                        logger.info("🔧 Converted positive conditioning back to tensors")
                
                if conditioning_negative is not None and isinstance(conditioning_negative, list):
                    if len(conditioning_negative) > 0 and isinstance(conditioning_negative[0], np.ndarray):
                        conditioning_negative = [torch.from_numpy(n) for n in conditioning_negative]
                        logger.info("🔧 Converted negative conditioning back to tensors")
            except Exception as e:
                logger.warning(f"Warning: Could not process conditioning: {e}")
            
            # 选择worker
            worker = self.get_next_worker()
            if worker is None:
                logger.error("没有可用的worker")
                return None
            
            # 执行推理
            if RAY_AVAILABLE and not isinstance(worker, XDiTWorkerFallback):
                # Ray模式
                future = worker.run_inference.remote(
                    model_info=enhanced_model_info,
                    conditioning_positive=conditioning_positive,
                    conditioning_negative=conditioning_negative,
                    latent_samples=latent_samples,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                
                # 等待结果，带超时
                timeout = min(600, num_inference_steps * 15)  # 最多10分钟
                try:
                    result = ray.get(future, timeout=timeout)
                    if result is not None:
                        logger.info("✅ Ray推理完成")
                        return result
                    else:
                        logger.warning("⚠️ Ray推理返回None")
                        return None
                except ray.exceptions.GetTimeoutError:
                    logger.error(f"⏰ Ray推理超时 ({timeout}s)")
                    return None
            else:
                # Fallback模式
                result = worker.run_inference(
                    model_info=enhanced_model_info,
                    conditioning_positive=conditioning_positive,
                    conditioning_negative=conditioning_negative,
                    latent_samples=latent_samples,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                
                if result is not None:
                    logger.info("✅ Fallback推理完成")
                    return result
                else:
                    logger.warning("⚠️ Fallback推理返回None")
                    return None
                    
        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            logger.exception("推理错误:")
            return None

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
        if not self.workers:
            return None
        
        worker = self.workers[self.current_worker_index % len(self.workers)]
        self.current_worker_index += 1
        
        # Update load for the worker's GPU
        if hasattr(worker, 'gpu_id'):
            gpu_id = worker.gpu_id
        else:
            # 对于Ray Actor，我们需要从索引推断GPU ID
            gpu_id = self.gpu_devices[self.current_worker_index - 1]
        
        self.worker_loads[gpu_id] = self.worker_loads.get(gpu_id, 0) + 1
        
        logger.debug(f"Round robin: assigned to GPU {gpu_id}")
        return worker
    
    def _least_loaded_schedule(self) -> Optional[Any]:
        """Least loaded scheduling"""
        if not self.workers:
            return None
        
        # Find worker with minimum load
        min_load = min(self.worker_loads.values()) if self.worker_loads else 0
        candidates = []
        
        for i, worker in enumerate(self.workers):
            gpu_id = self.gpu_devices[i] if i < len(self.gpu_devices) else 0
            load = self.worker_loads.get(gpu_id, 0)
            if load == min_load:
                candidates.append((worker, gpu_id))
        
        # If multiple candidates, choose the first one
        if candidates:
            worker, gpu_id = candidates[0]
            self.worker_loads[gpu_id] = self.worker_loads.get(gpu_id, 0) + 1
            
            logger.debug(f"Least loaded: assigned to GPU {gpu_id} (load: {min_load})")
            return worker
        
        return None
    
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
    
    def _validate_model_path(self, model_path: str) -> bool:
        """验证模型路径是否有效"""
        try:
            if model_path.endswith('.safetensors'):
                return os.path.exists(model_path)
            elif os.path.isdir(model_path):
                model_index_path = os.path.join(model_path, "model_index.json")
                return os.path.exists(model_index_path)
            else:
                return False
        except Exception as e:
            logger.error(f"Error validating model path {model_path}: {e}")
            return False

    def load_model_distributed(self, model_path: str, model_type: str = "flux"):
        """分布式加载模型"""
        try:
            logger.info(f"🚀 Starting distributed model loading: {model_path}")
            
            # 验证模型路径
            if not self._validate_model_path(model_path):
                raise ValueError(f"Invalid model path: {model_path}")
            
            # 检查模型格式
            if model_path.endswith('.safetensors'):
                logger.info("💡 Safetensors format detected - using ComfyUI component reuse strategy")
                logger.info("⚡ No downloads needed! Will use ComfyUI loaded VAE/CLIP components")
                logger.info("🎯 This should complete in seconds, not minutes")
            else:
                logger.info("📦 Diffusers format detected - loading complete pipeline")
            
            # 使用新的load_model方法
            futures = []
            for worker in self.workers.values():
                if RAY_AVAILABLE:
                    future = worker.load_model.remote(model_path, model_type)
                    futures.append(future)
                else:
                    # Fallback模式直接调用
                    result = worker.load_model(model_path, model_type)
                    futures.append(('direct', result))
            
            logger.info("⏳ Initializing workers with intelligent component reuse...")
            
            # 等待所有worker完成加载 - 对于safetensors应该很快
            timeout = 300 if model_path.endswith('.safetensors') else 1800  # safetensors: 5分钟, diffusers: 30分钟
            if RAY_AVAILABLE:
                results = ray.get(futures, timeout=timeout)
            else:
                results = [f[1] for f in futures]  # 提取结果
            
            # 分析结果
            success_count = sum(1 for r in results if r == "success")
            deferred_count = sum(1 for r in results if r == "deferred_loading")
            
            logger.info(f"📊 Loading results: {success_count} success, {deferred_count} deferred")
            
            if success_count > 0:
                logger.info("✅ Multi-GPU acceleration enabled!")
                self.model_loaded = True
                return "multi_gpu_success"
            elif deferred_count > 0:
                logger.info("✅ Workers ready for ComfyUI component integration")
                self.model_loaded = True
                return "fallback_to_comfyui"
            else:
                raise Exception("All workers failed to load model")
                
        except Exception as e:
            logger.error(f"❌ Distributed model loading failed: {e}")
            logger.exception("Full traceback:")
            return "failed"

    def get_status(self) -> Dict[str, Any]:
        """Get dispatcher status"""
        status = {
            "is_initialized": self.is_initialized,
            "num_workers": len(self.workers),
            "scheduling_strategy": self.scheduling_strategy.value,
            "worker_loads": self.worker_loads.copy(),
            "gpu_devices": self.gpu_devices,
            "model_path": self.model_path  # 添加这一行！
        }
        
        # Get detailed GPU info
        gpu_infos = {}
        for i, worker in enumerate(self.workers):
            gpu_id = self.gpu_devices[i] if i < len(self.gpu_devices) else 0
            try:
                if RAY_AVAILABLE and hasattr(worker, 'get_gpu_info'):
                    # Ray Actor
                    gpu_info = ray.get(worker.get_gpu_info.remote())
                else:
                    # Fallback worker
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
            
            for i, worker in enumerate(self.workers):
                try:
                    gpu_id = self.gpu_devices[i] if i < len(self.gpu_devices) else 0
                    
                    if RAY_AVAILABLE and hasattr(worker, 'cleanup'):
                        # Ray Actor
                        ray.get(worker.cleanup.remote())
                    else:
                        # Fallback worker
                        worker.cleanup()
                        
                    logger.info(f"✅ Worker on GPU {gpu_id} cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up worker on GPU {gpu_id}: {e}")
            
            self.workers.clear()
            self.worker_loads.clear()
            self.is_initialized = False
            
            logger.info("✅ XDiT Dispatcher cleaned up")
            
        except Exception as e:
            logger.error(f"Error during dispatcher cleanup: {e}") 