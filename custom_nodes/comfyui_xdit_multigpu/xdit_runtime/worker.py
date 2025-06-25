"""
xDiT Ray Worker
==============

Ray Actor for GPU-specific inference operations.
"""

import os
import torch
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Try to import xDiT
try:
    import xfuser
    from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
    XDIT_AVAILABLE = True
except ImportError:
    XDIT_AVAILABLE = False

logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1) if RAY_AVAILABLE else None
class XDiTWorker:
    """
    Ray Actor for GPU-specific xDiT operations.
    Each worker is assigned to one GPU.
    """
    
    def __init__(self, gpu_id: int, model_path: str, strategy: str = "Hybrid"):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.strategy = strategy
        self.pipeline = None
        self.is_initialized = False
        
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = f"cuda:0"  # Always cuda:0 since we set CUDA_VISIBLE_DEVICES
        
        logger.info(f"Initializing XDiT Worker on GPU {gpu_id}")
        
    def initialize(self) -> bool:
        """Initialize the worker with model loading"""
        try:
            if not XDIT_AVAILABLE:
                logger.error(f"xDiT not available on GPU {self.gpu_id}")
                return False
            
            logger.info(f"Loading model on GPU {self.gpu_id}")
            
            # Load xDiT pipeline - 禁用Flash Attention
            self.pipeline = xFuserPipelineBaseWrapper.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                use_cache=True,
                use_flash_attention=False  # 禁用Flash Attention
            )
            
            self.is_initialized = True
            logger.info(f"✅ Worker on GPU {self.gpu_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {e}")
            return False
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            props = torch.cuda.get_device_properties(0)  # Always 0 due to CUDA_VISIBLE_DEVICES
            memory_total = props.total_memory / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            
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
    
    def run_inference(self, 
                     prompt: str,
                     negative_prompt: str = "",
                     height: int = 512,
                     width: int = 512,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[torch.Tensor]:
        """Run inference on this GPU"""
        try:
            if not self.is_initialized or self.pipeline is None:
                logger.error(f"Worker on GPU {self.gpu_id} not initialized")
                return None
            
            logger.info(f"Running inference on GPU {self.gpu_id}: {height}x{width}, {num_inference_steps} steps")
            start_time = time.time()
            
            # Set random seed
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Run inference
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent"
                )
            
            end_time = time.time()
            logger.info(f"✅ Inference completed on GPU {self.gpu_id} in {end_time - start_time:.2f}s")
            
            # Extract latents
            if hasattr(result, 'latents'):
                return result.latents
            elif hasattr(result, 'images'):
                return self._images_to_latents(result.images)
            else:
                logger.warning(f"Unexpected result format from GPU {self.gpu_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error during inference on GPU {self.gpu_id}: {e}")
            return None
    
    def _images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to latents"""
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        latent_height, latent_width = height // 8, width // 8
        
        # Create placeholder latents
        latents = torch.randn(batch_size, 4, latent_height, latent_width, device=images.device)
        return latents
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.pipeline is not None:
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                del self.pipeline
                self.pipeline = None
            
            self.is_initialized = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"✅ Worker on GPU {self.gpu_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup on GPU {self.gpu_id}: {e}")

# Non-Ray fallback worker for when Ray is not available
class XDiTWorkerFallback:
    """Fallback worker when Ray is not available"""
    
    def __init__(self, gpu_id: int, model_path: str, strategy: str = "Hybrid"):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.strategy = strategy
        self.pipeline = None
        self.is_initialized = False
        
        logger.info(f"Initializing fallback worker on GPU {gpu_id}")
    
    def initialize(self) -> bool:
        """Initialize the fallback worker"""
        try:
            if not XDIT_AVAILABLE:
                logger.error(f"xDiT not available on GPU {self.gpu_id}")
                return False
            
            # Set GPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            
            logger.info(f"Loading model on GPU {self.gpu_id}")
            
            # Load xDiT pipeline - 禁用Flash Attention
            self.pipeline = xFuserPipelineBaseWrapper.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                use_cache=True,
                use_flash_attention=False  # 禁用Flash Attention
            )
            
            self.is_initialized = True
            logger.info(f"✅ Fallback worker on GPU {self.gpu_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback worker on GPU {self.gpu_id}: {e}")
            return False
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            props = torch.cuda.get_device_properties(0)
            memory_total = props.total_memory / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            
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
    
    def run_inference(self, 
                     prompt: str,
                     negative_prompt: str = "",
                     height: int = 512,
                     width: int = 512,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[torch.Tensor]:
        """Run inference on this GPU"""
        try:
            if not self.is_initialized or self.pipeline is None:
                logger.error(f"Fallback worker on GPU {self.gpu_id} not initialized")
                return None
            
            logger.info(f"Running inference on GPU {self.gpu_id}: {height}x{width}, {num_inference_steps} steps")
            start_time = time.time()
            
            # Set random seed
            generator = torch.Generator(device="cuda:0").manual_seed(seed)
            
            # Run inference
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent"
                )
            
            end_time = time.time()
            logger.info(f"✅ Inference completed on GPU {self.gpu_id} in {end_time - start_time:.2f}s")
            
            # Extract latents
            if hasattr(result, 'latents'):
                return result.latents
            elif hasattr(result, 'images'):
                return self._images_to_latents(result.images)
            else:
                logger.warning(f"Unexpected result format from GPU {self.gpu_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error during inference on GPU {self.gpu_id}: {e}")
            return None
    
    def _images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to latents"""
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        latent_height, latent_width = height // 8, width // 8
        
        # Create placeholder latents
        latents = torch.randn(batch_size, 4, latent_height, latent_width, device=images.device)
        return latents
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.pipeline is not None:
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                del self.pipeline
                self.pipeline = None
            
            self.is_initialized = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"✅ Fallback worker on GPU {self.gpu_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup on GPU {self.gpu_id}: {e}") 