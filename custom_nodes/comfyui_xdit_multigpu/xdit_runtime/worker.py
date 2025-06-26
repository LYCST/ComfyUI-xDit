"""
xDiT Ray Worker
==============

Ray Actor for GPU-specific inference operations.
"""

import os
import torch
import logging
import time
import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import threading
import socket

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Try to import xDiT with correct API
try:
    import xfuser
    from xfuser import xFuserArgs
    from xfuser.core.distributed import get_world_group
    # Import FLUX specific pipeline and config
    from xfuser import xFuserFluxPipeline
    from xfuser.config.config import EngineConfig
    XDIT_AVAILABLE = True
except ImportError:
    XDIT_AVAILABLE = False

logger = logging.getLogger(__name__)

def find_free_port():
    """Find a free port for distributed communication"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@ray.remote(num_gpus=1) if RAY_AVAILABLE else None
class XDiTWorker:
    """
    Ray Actor for GPU-specific xDiT operations.
    Each worker is assigned to one GPU.
    """
    
    def __init__(self, gpu_id: int, model_path: str, strategy: str = "Hybrid", 
                 master_addr: str = "127.0.0.1", master_port: int = 29500, 
                 world_size: int = 8, rank: int = 0):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.strategy = strategy
        self.model_wrapper = None
        self.is_initialized = False
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.distributed_initialized = False
        
        # 设置详细的日志级别
        logging.getLogger("xfuser").setLevel(logging.DEBUG)
        logging.getLogger("torch.distributed").setLevel(logging.DEBUG)
        
        # 设置GPU设备
        try:
            # 在Ray环境中设置CUDA设备
            if RAY_AVAILABLE:
                # Ray会自动设置CUDA_VISIBLE_DEVICES
                self.device = f"cuda:{0}"  # 在Ray中总是0
                torch.cuda.set_device(0)
            else:
                # 非Ray环境下手动设置
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                self.device = f"cuda:{0}"
                torch.cuda.set_device(0)
                
            logger.info(f"Worker initialized on GPU {gpu_id} (device={self.device}, rank={rank})")
            
        except Exception as e:
            logger.error(f"Failed to set GPU device for worker {gpu_id}: {e}")
            raise
    
    def initialize(self) -> bool:
        """Initialize the worker"""
        try:
            if not XDIT_AVAILABLE:
                logger.error(f"xDiT not available on GPU {self.gpu_id}")
                return False
            
            logger.info(f"Initializing worker on GPU {self.gpu_id} with strategy: {self.strategy}")
            
            # 验证GPU可用性
            if not torch.cuda.is_available():
                logger.error(f"CUDA not available for worker on GPU {self.gpu_id}")
                return False
            
            # 验证当前设备
            current_device = torch.cuda.current_device()
            logger.info(f"Current CUDA device for GPU {self.gpu_id}: {current_device}")
            
            # 标记初始化完成
            self.is_initialized = True
            logger.info(f"✅ Worker on GPU {self.gpu_id} initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return False
    
    def initialize_distributed(self) -> bool:
        """Initialize distributed environment - called after all workers are ready"""
        if self.distributed_initialized:
            return True
            
        try:
            logger.info(f"[GPU {self.gpu_id}] Initializing distributed environment...")
            
            # 清理可能存在的旧进程组
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                    logger.info(f"[GPU {self.gpu_id}] Destroyed existing process group")
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")
            
            # 设置分布式环境变量
            os.environ['MASTER_ADDR'] = self.master_addr
            os.environ['MASTER_PORT'] = str(self.master_port)
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['RANK'] = str(self.rank)
            os.environ['LOCAL_RANK'] = '0'  # Ray中每个worker都是local rank 0
            os.environ['NCCL_DEBUG'] = 'INFO'  # 启用NCCL调试
            
            logger.info(f"[GPU {self.gpu_id}] Distributed env: MASTER={self.master_addr}:{self.master_port}, "
                       f"WORLD_SIZE={self.world_size}, RANK={self.rank}")
            
            # 初始化分布式环境
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=f'tcp://{self.master_addr}:{self.master_port}',
                world_size=self.world_size,
                rank=self.rank,
                timeout=datetime.timedelta(minutes=10)  # 增加超时时间
            )
            
            # 验证分布式初始化
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                logger.info(f"[GPU {self.gpu_id}] ✅ Distributed initialized: world_size={world_size}, rank={rank}")
                self.distributed_initialized = True
                return True
            else:
                logger.error(f"[GPU {self.gpu_id}] Distributed initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Failed to initialize distributed: {e}")
            logger.exception("Distributed init traceback:")
            self._cleanup_distributed()
            return False
    
    def _create_xfuser_pipeline_if_needed(self, model_path: str = None) -> bool:
        """延迟创建xfuser pipeline（仅在第一次推理时调用）"""
        if self.model_wrapper is not None:
            return True
            
        try:
            # 使用传递的模型路径，如果没有则使用默认路径
            effective_model_path = model_path or self.model_path
            logger.info(f"[GPU {self.gpu_id}] Creating xfuser pipeline with path: {effective_model_path}")
            
            # 确保分布式环境已初始化
            if not self.distributed_initialized:
                logger.error(f"[GPU {self.gpu_id}] Distributed not initialized")
                return False
            
            # 创建pipeline
            try:
                # 确保在正确的设备上
                torch.cuda.set_device(0)  # Ray中总是0
                
                # 🔧 检查模型路径类型并相应处理
                if effective_model_path.endswith('.safetensors'):
                    logger.info(f"[GPU {self.gpu_id}] Processing safetensors file: {effective_model_path}")
                    
                    # 对于safetensors文件，我们需要使用ComfyUI的方式加载，然后转换为xDiT可用的格式
                    # 这里我们先尝试直接使用safetensors路径，如果失败则返回False触发fallback
                    try:
                        # 尝试直接使用safetensors路径创建pipeline
                        # 注意：这可能需要xDiT/xFuser支持直接加载safetensors
                        logger.info(f"[GPU {self.gpu_id}] Attempting to load safetensors directly with xFuser")
                        
                        # 使用xFuserArgs创建配置，但先检查是否支持safetensors
                        xfuser_args = xFuserArgs(
                            model=effective_model_path,
                            height=1024,
                            width=1024,
                            num_inference_steps=20,
                            guidance_scale=3.5,
                            output_type="latent",
                            tensor_parallel_degree=self.world_size,
                            use_ray=True,
                            ray_world_size=self.world_size
                        )
                        
                        engine_config = xfuser_args.create_config()
                        
                        # 尝试创建pipeline
                        self.model_wrapper = xFuserFluxPipeline.from_pretrained(
                            effective_model_path,
                            engine_config=engine_config,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True
                        )
                        
                        logger.info(f"[GPU {self.gpu_id}] ✅ Successfully loaded safetensors with xFuser")
                        
                    except Exception as safetensors_error:
                        logger.warning(f"[GPU {self.gpu_id}] Direct safetensors loading failed: {safetensors_error}")
                        logger.info(f"[GPU {self.gpu_id}] xFuser may not support direct safetensors loading")
                        # 对于safetensors，我们返回False让系统fallback到ComfyUI原生方式
                        return False
                        
                elif os.path.isdir(effective_model_path):
                    logger.info(f"[GPU {self.gpu_id}] Processing diffusers directory: {effective_model_path}")
                    
                    # 验证diffusers格式
                    model_index_path = os.path.join(effective_model_path, "model_index.json")
                    if not os.path.exists(model_index_path):
                        logger.error(f"[GPU {self.gpu_id}] Invalid diffusers directory: no model_index.json")
                        return False
                    
                    # 使用xFuserArgs创建完整配置
                    xfuser_args = xFuserArgs(
                        model=effective_model_path,
                        height=1024,
                        width=1024,
                        num_inference_steps=20,
                        guidance_scale=3.5,
                        output_type="latent",
                        tensor_parallel_degree=self.world_size,
                        use_ray=True,
                        ray_world_size=self.world_size
                    )
                    
                    engine_config = xfuser_args.create_config()
                    
                    # 创建pipeline
                    self.model_wrapper = xFuserFluxPipeline.from_pretrained(
                        effective_model_path,
                        engine_config=engine_config,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True
                    )
                    
                    logger.info(f"[GPU {self.gpu_id}] ✅ Successfully loaded diffusers directory")
                    
                else:
                    logger.error(f"[GPU {self.gpu_id}] Unsupported model path format: {effective_model_path}")
                    return False
                
                # 确保模型在正确的设备上
                self.model_wrapper.to(self.device)
                
                logger.info(f"[GPU {self.gpu_id}] ✅ Pipeline created on {self.device}")
                return True
                
            except Exception as e:
                logger.error(f"[GPU {self.gpu_id}] Failed to create pipeline: {e}")
                logger.exception("Pipeline creation traceback:")
                self._cleanup_distributed()
                return False
                
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Pipeline creation failed: {e}")
            logger.exception("Full traceback:")
            self._cleanup_distributed()
            return False
    
    def _cleanup_distributed(self):
        """清理分布式环境"""
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info(f"[GPU {self.gpu_id}] Process group destroyed")
            
            # 清理环境变量
            for var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'NCCL_DEBUG']:
                if var in os.environ:
                    del os.environ[var]
            
            self.distributed_initialized = False
                    
        except Exception as e:
            logger.warning(f"Error during distributed cleanup: {e}")
    
    def run_inference(self, 
                     model_info: Dict,
                     conditioning_positive: Any,
                     conditioning_negative: Any,
                     latent_samples: torch.Tensor,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[torch.Tensor]:
        """运行推理"""
        try:
            if not self.is_initialized:
                logger.error(f"Worker on GPU {self.gpu_id} not initialized")
                return None
            
            # 检查是否需要分布式推理
            if self.world_size > 1:
                # 多GPU模式：确保分布式已初始化
                if not self.distributed_initialized:
                    logger.error(f"[GPU {self.gpu_id}] Distributed not initialized for multi-GPU inference")
                    return None
                
                # 创建pipeline，传递正确的模型路径
                model_path = model_info.get('path', self.model_path)
                if not self._create_xfuser_pipeline_if_needed(model_path):
                    logger.warning(f"⚠️ Pipeline creation failed on GPU {self.gpu_id}")
                    return None
            else:
                # 单GPU模式：直接运行
                logger.info(f"[GPU {self.gpu_id}] Running single-GPU inference")
            
            logger.info(f"Running inference on GPU {self.gpu_id}: {num_inference_steps} steps")
            start_time = time.time()
            
            # 设置随机种子
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            # 确保数据在正确的设备上
            latent_samples = latent_samples.to(self.device)
            
            # 处理conditioning
            prompt_embeds = None
            pooled_prompt_embeds = None
            if conditioning_positive and len(conditioning_positive) > 0:
                if isinstance(conditioning_positive[0], torch.Tensor):
                    prompt_embeds = conditioning_positive[0].to(self.device)
                    if len(conditioning_positive) > 1:
                        pooled_prompt_embeds = conditioning_positive[1].to(self.device)
            
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            if conditioning_negative and len(conditioning_negative) > 0:
                if isinstance(conditioning_negative[0], torch.Tensor):
                    negative_prompt_embeds = conditioning_negative[0].to(self.device)
                    if len(conditioning_negative) > 1:
                        negative_pooled_prompt_embeds = conditioning_negative[1].to(self.device)
            
            # 如果是单GPU或分布式失败，返回None触发fallback
            if self.world_size == 1 or not self.distributed_initialized:
                logger.info(f"[GPU {self.gpu_id}] Returning None to trigger ComfyUI fallback")
                return None
            
            # 创建推理配置
            model_path = model_info.get('path', self.model_path)
            xfuser_args = xFuserArgs(
                model=model_path,
                height=latent_samples.shape[2] * 8,
                width=latent_samples.shape[3] * 8,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="latent",
                strategy=self.strategy,
                device=self.device,
                dtype="bfloat16"
            )
            
            engine_config = xfuser_args.create_config()
            logger.info(f"[GPU {self.gpu_id}] Config created: steps={num_inference_steps}, CFG={guidance_scale}")
            
            # 运行推理
            try:
                with torch.inference_mode():
                    result = self.model_wrapper(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        height=latent_samples.shape[2] * 8,
                        width=latent_samples.shape[3] * 8,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(seed),
                        output_type="latent",
                        latents=latent_samples,
                        engine_config=engine_config
                    )
                
                # 提取结果
                if hasattr(result, 'images'):
                    result_latents = result.images
                elif hasattr(result, 'latents'):
                    result_latents = result.latents
                else:
                    result_latents = result
                
                # 确保结果在正确的设备上
                result_latents = result_latents.to(self.device)
                
                end_time = time.time()
                logger.info(f"✅ Inference completed on GPU {self.gpu_id} in {end_time - start_time:.2f}s")
                
                return result_latents
                
            except Exception as e:
                logger.error(f"Inference failed on GPU {self.gpu_id}: {e}")
                logger.exception("Inference traceback:")
                return None
                
        except Exception as e:
            logger.error(f"Error during inference on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return None
    
    def cleanup(self):
        """清理资源"""
        try:
            # 清理模型
            if self.model_wrapper is not None:
                del self.model_wrapper
                self.model_wrapper = None
            
            # 清理分布式环境
            self._cleanup_distributed()
            
            # 清理CUDA缓存
            torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info(f"✅ Worker on GPU {self.gpu_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup on GPU {self.gpu_id}: {e}")
            logger.exception("Cleanup traceback:")

# Non-Ray fallback worker for when Ray is not available
class XDiTWorkerFallback:
    """Fallback worker when Ray is not available"""
    
    def __init__(self, gpu_id: int, model_path: str, strategy: str = "Hybrid"):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.strategy = strategy
        self.model_wrapper = None
        self.is_initialized = False
        
        logger.info(f"Initializing fallback worker on GPU {gpu_id}")
    
    def initialize(self) -> bool:
        """Initialize the fallback worker"""
        try:
            # 设置GPU设备
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
            
            logger.info(f"Fallback worker on GPU {self.gpu_id} ready")
            
            self.is_initialized = True
            logger.info(f"✅ Fallback worker on GPU {self.gpu_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback worker on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return False
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            if not torch.cuda.is_available():
                return {"gpu_id": self.gpu_id, "error": "CUDA not available"}
                
            props = torch.cuda.get_device_properties(self.gpu_id)
            memory_total = props.total_memory / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(self.gpu_id) / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
            
            return {
                "gpu_id": self.gpu_id,
                "name": props.name,
                "memory_total_gb": memory_total,
                "memory_reserved_gb": memory_reserved,
                "memory_allocated_gb": memory_allocated,
                "memory_free_gb": memory_total - memory_reserved,
                "compute_capability": f"{props.major}.{props.minor}",
                "is_initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Error getting GPU info for GPU {self.gpu_id}: {e}")
            return {"gpu_id": self.gpu_id, "error": str(e)}
    
    def wrap_model_for_xdit(self, model_state_dict: Dict, model_config: Dict) -> bool:
        """Wrap model with xDiT (fallback version)"""
        try:
            logger.info(f"Wrapping model with fallback method on GPU {self.gpu_id}")
            
            # 在没有xDiT的情况下，简单地准备好接受模型
            self.model_wrapper = "fallback_placeholder"
            
            logger.info(f"✅ Model ready on GPU {self.gpu_id} (fallback mode)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to wrap model on GPU {self.gpu_id}: {e}")
            return False
    
    def run_inference(self, 
                     model_info: Dict,  # 改为轻量级的模型信息
                     conditioning_positive: Any,
                     conditioning_negative: Any,
                     latent_samples: torch.Tensor,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[torch.Tensor]:
        """Run inference (fallback version)"""
        try:
            if not self.is_initialized:
                logger.error(f"Fallback worker on GPU {self.gpu_id} not initialized")
                return None
            
            logger.info(f"Running fallback inference on GPU {self.gpu_id}")
            logger.info(f"Model info: {model_info.get('type', 'unknown')}, Latent shape: {latent_samples.shape}")
            start_time = time.time()
            
            # 设置随机种子
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # 🔧 优化：确保数据在正确的GPU上
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            if latent_samples.device != torch.device(device):
                latent_samples = latent_samples.to(device)
            
            # 🔧 修复Flux模型通道数问题
            # Flux模型使用16通道latent space，如果输入是4通道，需要进行转换
            if latent_samples.shape[1] == 4 and model_info.get('type', '').lower() == 'flux':
                logger.info(f"Converting 4-channel to 16-channel latent for Flux model")
                # 简单的通道扩展策略（实际的xDiT会有更复杂的处理）
                # 方法1：重复通道 [1,4,H,W] -> [1,16,H,W]
                expanded_latents = latent_samples.repeat(1, 4, 1, 1)
                # 方法2：也可以用零填充
                # expanded_latents = torch.cat([latent_samples, torch.zeros_like(latent_samples).repeat(1, 3, 1, 1)], dim=1)
                latent_samples = expanded_latents
                logger.info(f"Latent shape converted to: {latent_samples.shape}")
            
            # 在fallback模式下，只是返回原始latents
            # 实际使用中会回退到ComfyUI的标准采样
            # result_latents = latent_samples.clone()
            
            # TODO: fallback模式也返回None，让系统使用标准ComfyUI采样
            logger.info(f"⚠️ Fallback worker returning None to trigger standard ComfyUI sampling")
            return None
            
            end_time = time.time()
            logger.info(f"✅ Fallback inference completed on GPU {self.gpu_id} in {end_time - start_time:.2f}s")
            
            # return result_latents
                
        except Exception as e:
            logger.error(f"Error during fallback inference on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.model_wrapper is not None:
                del self.model_wrapper
                self.model_wrapper = None
            
            self.is_initialized = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"✅ Fallback worker on GPU {self.gpu_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup on GPU {self.gpu_id}: {e}") 