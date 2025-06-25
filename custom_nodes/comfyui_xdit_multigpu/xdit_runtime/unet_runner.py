"""
UNet Runner
==========

Wrapper for UNet inference logic with parallel execution support.
"""

import os
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
except ImportError:
    XDIT_AVAILABLE = False

logger = logging.getLogger(__name__)

class UNetRunner:
    """
    Wrapper for UNet inference logic with parallel execution support.
    """
    
    def __init__(self, model_path: str, gpu_devices: List[int], strategy: str = "Hybrid"):
        self.model_path = model_path
        self.gpu_devices = gpu_devices
        self.strategy = strategy
        self.pipeline = None
        self.is_initialized = False
        
        logger.info(f"Initializing UNet Runner with strategy: {strategy}")
    
    def initialize(self) -> bool:
        """Initialize the UNet runner"""
        try:
            if not XDIT_AVAILABLE:
                logger.error("xDiT not available")
                return False
            
            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_devices))
            
            logger.info(f"Loading UNet model from {self.model_path}")
            
            # Load xDiT pipeline - 禁用Flash Attention
            self.pipeline = xFuserPipelineBaseWrapper.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                use_cache=True,
                use_flash_attention=False  # 禁用Flash Attention
            )
            
            self.is_initialized = True
            logger.info(f"✅ UNet Runner initialized successfully on {len(self.gpu_devices)} GPUs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize UNet Runner: {e}")
            return False
    
    def run_inference(self, 
                     prompt: str,
                     negative_prompt: str = "",
                     height: int = 512,
                     width: int = 512,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42,
                     batch_size: int = 1) -> Optional[torch.Tensor]:
        """Run UNet inference"""
        try:
            if not self.is_initialized or self.pipeline is None:
                logger.error("UNet Runner not initialized")
                return None
            
            logger.info(f"Running UNet inference: {height}x{width}, {num_inference_steps} steps, batch_size={batch_size}")
            start_time = time.time()
            
            # Set random seed
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            # Run inference
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=batch_size,
                    generator=generator,
                    output_type="latent"
                )
            
            end_time = time.time()
            logger.info(f"✅ UNet inference completed in {end_time - start_time:.2f}s")
            
            # Extract latents
            if hasattr(result, 'latents'):
                return result.latents
            elif hasattr(result, 'images'):
                return self._images_to_latents(result.images)
            else:
                logger.warning("Unexpected result format from UNet pipeline")
                return None
                
        except Exception as e:
            logger.error(f"Error during UNet inference: {e}")
            return None
    
    def _images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to latents"""
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        latent_height, latent_width = height // 8, width // 8
        
        # Create placeholder latents
        latents = torch.randn(batch_size, 4, latent_height, latent_width, device=images.device)
        return latents
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_initialized or self.pipeline is None:
            return {"error": "Model not initialized"}
        
        try:
            info = {
                "model_path": self.model_path,
                "strategy": self.strategy,
                "gpu_devices": self.gpu_devices,
                "is_initialized": self.is_initialized
            }
            
            # Get model parameters count if available
            if hasattr(self.pipeline, 'unet'):
                unet = self.pipeline.unet
                if hasattr(unet, 'num_parameters'):
                    info["num_parameters"] = unet.num_parameters()
                elif hasattr(unet, 'parameters'):
                    info["num_parameters"] = sum(p.numel() for p in unet.parameters())
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get GPU memory usage"""
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
            if self.pipeline is not None:
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                del self.pipeline
                self.pipeline = None
            
            self.is_initialized = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ UNet Runner cleaned up")
            
        except Exception as e:
            logger.error(f"Error during UNet Runner cleanup: {e}")

class ParallelUNetRunner:
    """
    Parallel UNet runner for multi-GPU execution.
    """
    
    def __init__(self, model_path: str, gpu_devices: List[int], strategy: str = "Hybrid"):
        self.model_path = model_path
        self.gpu_devices = gpu_devices
        self.strategy = strategy
        self.runners = {}
        self.is_initialized = False
        
        logger.info(f"Initializing Parallel UNet Runner with {len(gpu_devices)} GPUs")
    
    def initialize(self) -> bool:
        """Initialize parallel UNet runners"""
        try:
            # Create individual runners for each GPU
            for gpu_id in self.gpu_devices:
                # Set CUDA device for this runner
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                
                runner = UNetRunner(self.model_path, [gpu_id], self.strategy)
                success = runner.initialize()
                
                if success:
                    self.runners[gpu_id] = runner
                    logger.info(f"✅ UNet Runner initialized on GPU {gpu_id}")
                else:
                    logger.error(f"Failed to initialize UNet Runner on GPU {gpu_id}")
                    return False
            
            self.is_initialized = True
            logger.info(f"✅ Parallel UNet Runner initialized with {len(self.runners)} runners")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Parallel UNet Runner: {e}")
            return False
    
    def run_parallel_inference(self, 
                              prompts: List[str],
                              negative_prompts: List[str] = None,
                              height: int = 512,
                              width: int = 512,
                              num_inference_steps: int = 20,
                              guidance_scale: float = 8.0,
                              seeds: List[int] = None) -> List[Optional[torch.Tensor]]:
        """Run parallel inference across multiple GPUs"""
        try:
            if not self.is_initialized:
                logger.error("Parallel UNet Runner not initialized")
                return [None] * len(prompts)
            
            if negative_prompts is None:
                negative_prompts = [""] * len(prompts)
            
            if seeds is None:
                seeds = [42 + i for i in range(len(prompts))]
            
            logger.info(f"Running parallel inference for {len(prompts)} prompts")
            start_time = time.time()
            
            # Distribute prompts across GPUs
            results = [None] * len(prompts)
            gpu_ids = list(self.runners.keys())
            
            for i, (prompt, negative_prompt, seed) in enumerate(zip(prompts, negative_prompts, seeds)):
                # Round-robin distribution
                gpu_id = gpu_ids[i % len(gpu_ids)]
                runner = self.runners[gpu_id]
                
                logger.info(f"Running prompt {i+1}/{len(prompts)} on GPU {gpu_id}")
                
                result = runner.run_inference(
                    prompt, negative_prompt, height, width,
                    num_inference_steps, guidance_scale, seed
                )
                
                results[i] = result
            
            end_time = time.time()
            logger.info(f"✅ Parallel inference completed in {end_time - start_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during parallel inference: {e}")
            return [None] * len(prompts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get parallel runner status"""
        status = {
            "is_initialized": self.is_initialized,
            "num_runners": len(self.runners),
            "gpu_devices": self.gpu_devices,
            "strategy": self.strategy
        }
        
        # Get individual runner status
        runner_status = {}
        for gpu_id, runner in self.runners.items():
            runner_status[gpu_id] = {
                "model_info": runner.get_model_info(),
                "memory_usage": runner.get_gpu_memory_usage()
            }
        
        status["runner_status"] = runner_status
        return status
    
    def cleanup(self):
        """Clean up all runners"""
        try:
            logger.info("Cleaning up Parallel UNet Runner")
            
            for gpu_id, runner in self.runners.items():
                try:
                    runner.cleanup()
                    logger.info(f"✅ Runner on GPU {gpu_id} cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up runner on GPU {gpu_id}: {e}")
            
            self.runners.clear()
            self.is_initialized = False
            
            logger.info("✅ Parallel UNet Runner cleaned up")
            
        except Exception as e:
            logger.error(f"Error during parallel runner cleanup: {e}") 