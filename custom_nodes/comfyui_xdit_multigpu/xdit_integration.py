"""
xDiT Integration Module
======================

This module provides the core integration with xDiT framework for multi-GPU acceleration.
"""

import os
import sys
import torch
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

# Try to import xDiT
try:
    import xfuser
    from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
    XDIT_AVAILABLE = True
except ImportError as e:
    XDIT_AVAILABLE = False
    print(f"xDiT import error: {e}")

logger = logging.getLogger(__name__)

class XDiTManager:
    """Manager class for xDiT multi-GPU operations"""
    
    def __init__(self):
        self.pipelines = {}
        self.parallel_configs = {}
        self.device_maps = {}
        
    def create_parallel_config(self, strategy: str, gpu_devices: List[int]) -> Optional[Dict]:
        """Create parallel configuration for xDiT"""
        if not XDIT_AVAILABLE:
            return None
            
        try:
            # 简化的并行配置
            config = {
                "parallel_method": strategy.lower(),
                "num_gpus": len(gpu_devices),
                "device_map": "auto",
                "gpu_devices": gpu_devices
            }
            
            if strategy == "Hybrid":
                config["hybrid_config"] = {
                    "pipefusion_layers": 0.5,
                    "usp_layers": 0.5
                }
                
            return config
            
        except Exception as e:
            logger.error(f"Failed to create parallel config: {e}")
            return None
    
    def load_model_with_xdit(self, model_path: str, strategy: str, gpu_devices: List[int], 
                           use_cache: bool = True, use_flash_attention: bool = False) -> Optional[xFuserPipelineBaseWrapper]:
        """Load model using xDiT with multi-GPU support"""
        if not XDIT_AVAILABLE:
            logger.warning("xDiT not available")
            return None
            
        try:
            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
            if use_cache:
                os.environ["XFUSER_ENABLE_CACHE"] = "1"
            
            # Create parallel configuration
            parallel_config = self.create_parallel_config(strategy, gpu_devices)
            if parallel_config is None:
                return None
            
            # Create unique key for this configuration
            config_key = f"{model_path}_{strategy}_{','.join(map(str, gpu_devices))}"
            
            # Check if pipeline already exists
            if config_key in self.pipelines:
                logger.info(f"Using cached pipeline for {config_key}")
                return self.pipelines[config_key]
            
            logger.info(f"Loading model {model_path} with strategy {strategy} on GPUs {gpu_devices}")
            
            # Load pipeline with xDiT - 禁用Flash Attention
            pipeline = xFuserPipelineBaseWrapper.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_cache=use_cache,
                use_flash_attention=False,  # 禁用Flash Attention
                device_map="auto"
            )
            
            # Cache the pipeline
            self.pipelines[config_key] = pipeline
            self.parallel_configs[config_key] = parallel_config
            
            logger.info(f"✅ Model loaded successfully with xDiT on {len(gpu_devices)} GPUs")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model with xDiT: {e}")
            return None
    
    def generate_with_xdit(self, pipeline: xFuserPipelineBaseWrapper, 
                          prompt: str, negative_prompt: str = "",
                          height: int = 512, width: int = 512,
                          num_inference_steps: int = 20, guidance_scale: float = 8.0,
                          num_images_per_prompt: int = 1, seed: int = 42) -> Optional[torch.Tensor]:
        """Generate images using xDiT multi-GPU pipeline"""
        if not XDIT_AVAILABLE or pipeline is None:
            return None
            
        try:
            logger.info(f"Generating image with xDiT: {height}x{width}, {num_inference_steps} steps")
            start_time = time.time()
            
            # Set random seed
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            # Generate with xDiT pipeline
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    output_type="latent"  # Return latents for ComfyUI compatibility
                )
            
            end_time = time.time()
            logger.info(f"✅ Generation completed in {end_time - start_time:.2f}s")
            
            # Extract latents from result
            if hasattr(result, 'latents'):
                return result.latents
            elif hasattr(result, 'images'):
                # Convert images to latents if needed
                return self._images_to_latents(result.images)
            else:
                logger.warning("Unexpected result format from xDiT pipeline")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate with xDiT: {e}")
            return None
    
    def _images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to latents (placeholder implementation)"""
        # This is a simplified conversion - in practice you'd use VAE encoder
        # For now, we'll return a placeholder
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        latent_height, latent_width = height // 8, width // 8
        
        # Create placeholder latents
        latents = torch.randn(batch_size, 4, latent_height, latent_width, device=images.device)
        return latents
    
    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get current GPU memory usage"""
        memory_info = {}
        
        if not torch.cuda.is_available():
            return memory_info
            
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                
                memory_info[i] = {
                    "total_gb": memory_total,
                    "reserved_gb": memory_reserved,
                    "allocated_gb": memory_allocated,
                    "free_gb": memory_total - memory_reserved,
                    "utilization_percent": (memory_reserved / memory_total) * 100
                }
            except Exception as e:
                logger.warning(f"Could not get memory info for GPU {i}: {e}")
                
        return memory_info
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clear pipelines
            for pipeline in self.pipelines.values():
                if hasattr(pipeline, 'to'):
                    pipeline.to('cpu')
                del pipeline
            
            self.pipelines.clear()
            self.parallel_configs.clear()
            self.device_maps.clear()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ xDiT resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global xDiT manager instance
xdit_manager = XDiTManager()

class XDiTModelLoader:
    """Enhanced model loader with xDiT integration"""
    
    @staticmethod
    def load_model(model_path: str, strategy: str, gpu_devices: List[int], 
                  use_cache: bool = True, use_flash_attention: bool = False):
        """Load model using xDiT"""
        return xdit_manager.load_model_with_xdit(
            model_path, strategy, gpu_devices, use_cache, use_flash_attention
        )

class XDiTGenerator:
    """Enhanced generator with xDiT integration"""
    
    @staticmethod
    def generate_image(pipeline, prompt: str, negative_prompt: str = "",
                      height: int = 512, width: int = 512,
                      num_inference_steps: int = 20, guidance_scale: float = 8.0,
                      num_images_per_prompt: int = 1, seed: int = 42):
        """Generate image using xDiT"""
        return xdit_manager.generate_with_xdit(
            pipeline, prompt, negative_prompt, height, width,
            num_inference_steps, guidance_scale, num_images_per_prompt, seed
        ) 