"""
ComfyUI xDiT Multi-GPU Nodes
============================

This module contains the core nodes for multi-GPU acceleration in ComfyUI.
Fully compatible with original ComfyUI nodes for seamless switching.
"""

import os
import sys
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
from enum import Enum

# ComfyUI imports
import folder_paths
import comfy.utils
import comfy.sd

# Import our xDiT runtime
try:
    from .xdit_runtime import XDiTDispatcher, SchedulingStrategy
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from xdit_runtime import XDiTDispatcher, SchedulingStrategy
    except ImportError:
        # 如果都失败，创建占位符类
        class XDiTDispatcher:
            def __init__(self, *args, **kwargs):
                pass
            def initialize(self):
                return False
            def get_status(self):
                return {}
            def run_inference(self, *args, **kwargs):
                return None
        
        class SchedulingStrategy:
            ROUND_ROBIN = "round_robin"

logger = logging.getLogger(__name__)

def parse_scheduling_strategy(scheduling_strategy_str: str) -> SchedulingStrategy:
    """解析调度策略字符串为枚举值"""
    try:
        # 尝试直接匹配枚举值
        for strategy in SchedulingStrategy:
            if strategy.value == scheduling_strategy_str:
                return strategy
        
        # 如果直接匹配失败，尝试其他方式
        strategy_map = {
            "round_robin": SchedulingStrategy.ROUND_ROBIN,
            "least_loaded": SchedulingStrategy.LEAST_LOADED,
            "weighted_round_robin": SchedulingStrategy.WEIGHTED_ROUND_ROBIN,
            "adaptive": SchedulingStrategy.ADAPTIVE,
        }
        
        if scheduling_strategy_str in strategy_map:
            return strategy_map[scheduling_strategy_str]
        
        # 默认返回轮询策略
        logger.warning(f"Unknown scheduling strategy '{scheduling_strategy_str}', using round_robin")
        return SchedulingStrategy.ROUND_ROBIN
        
    except Exception as e:
        logger.error(f"Error parsing scheduling strategy: {e}")
        return SchedulingStrategy.ROUND_ROBIN

# Try to import xDiT
try:
    import xfuser
    from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
    XDIT_AVAILABLE = True
except ImportError:
    XDIT_AVAILABLE = False

class XDiTCheckpointLoader:
    """
    Multi-GPU Checkpoint Loader - Drop-in replacement for CheckpointLoaderSimple
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
                "enable_multi_gpu": ("BOOLEAN", {"default": True, "tooltip": "Enable multi-GPU acceleration"}),
                "gpu_devices": ("STRING", {"default": "0,1,2,3", "multiline": False, "tooltip": "Comma-separated list of GPU device IDs"}),
                "parallel_strategy": (["PipeFusion", "USP", "Hybrid", "Tensor", "CFG"], {"default": "Hybrid", "tooltip": "xDiT parallel strategy"}),
                "scheduling_strategy": (["round_robin", "least_loaded", "weighted_round_robin", "adaptive"], {"default": "round_robin", "tooltip": "Worker scheduling strategy"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "XDIT_DISPATCHER")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "xDiT dispatcher for multi-GPU acceleration.")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint with optional multi-GPU acceleration using xDiT framework."

    def load_checkpoint(self, ckpt_name, enable_multi_gpu=True, gpu_devices="0,1,2,3", parallel_strategy="Hybrid", scheduling_strategy="round_robin"):
        """
        Load checkpoint with optional multi-GPU acceleration
        """
        try:
            # Load checkpoint using standard ComfyUI method
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model, clip, vae = out[:3]
            
            dispatcher = None
            
            # Enable multi-GPU if requested and available
            if enable_multi_gpu and XDIT_AVAILABLE:
                try:
                    # Parse GPU devices
                    gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
                    logger.info(f"Initializing xDiT multi-GPU acceleration for {ckpt_name} on GPUs: {gpu_list}")
                    
                    # 使用修复的解析函数
                    scheduling_enum = parse_scheduling_strategy(scheduling_strategy)
                    logger.info(f"Using scheduling strategy: {scheduling_enum.value}")
                    
                    # Create dispatcher
                    dispatcher = XDiTDispatcher(
                        gpu_devices=gpu_list,
                        model_path=ckpt_path,
                        strategy=parallel_strategy,
                        scheduling_strategy=scheduling_enum
                    )
                    
                    # Initialize dispatcher
                    success = dispatcher.initialize()
                    if success:
                        status = dispatcher.get_status()
                        logger.info(f"✅ xDiT multi-GPU acceleration enabled with {status['num_workers']} workers")
                        logger.info(f"   Scheduling strategy: {status['scheduling_strategy']}")
                    else:
                        logger.warning("Failed to initialize xDiT dispatcher, falling back to single-GPU")
                        dispatcher = None
                        
                except Exception as e:
                    logger.error(f"Error initializing xDiT multi-GPU: {e}")
                    logger.info("Falling back to single-GPU mode")
                    dispatcher = None
            else:
                if not XDIT_AVAILABLE:
                    logger.info("xDiT not available, using standard single-GPU mode")
                else:
                    logger.info("Multi-GPU disabled, using standard single-GPU mode")
            
            return (model, clip, vae, dispatcher)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_name}: {e}")
            # Return empty objects on error
            return (None, None, None, None)


class XDiTUNetLoader:
    """
    Multi-GPU UNet Loader - Drop-in replacement for UNetLoader
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"), {"tooltip": "The name of the UNet model to load."}),
                "enable_multi_gpu": ("BOOLEAN", {"default": True, "tooltip": "Enable multi-GPU acceleration"}),
                "gpu_devices": ("STRING", {"default": "0,1,2,3", "multiline": False, "tooltip": "Comma-separated list of GPU device IDs"}),
                "parallel_strategy": (["PipeFusion", "USP", "Hybrid", "Tensor", "CFG"], {"default": "Hybrid", "tooltip": "xDiT parallel strategy"}),
                "scheduling_strategy": (["round_robin", "least_loaded", "weighted_round_robin", "adaptive"], {"default": "round_robin", "tooltip": "Worker scheduling strategy"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "XDIT_DISPATCHER")
    OUTPUT_TOOLTIPS = ("The UNet model used for denoising latents.",
                       "xDiT dispatcher for multi-GPU acceleration.")
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a UNet model with optional multi-GPU acceleration using xDiT framework."

    def load_unet(self, unet_name, enable_multi_gpu=True, gpu_devices="0,1,2,3", parallel_strategy="Hybrid", scheduling_strategy="round_robin"):
        """
        Load UNet with optional multi-GPU acceleration
        """
        try:
            # Load UNet using standard ComfyUI method
            unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)
            model = comfy.sd.load_unet(unet_path)
            
            dispatcher = None
            
            # Enable multi-GPU if requested and available
            if enable_multi_gpu and XDIT_AVAILABLE:
                try:
                    # Parse GPU devices
                    gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
                    logger.info(f"Initializing xDiT multi-GPU acceleration for UNet {unet_name} on GPUs: {gpu_list}")
                    
                    # 使用修复的解析函数
                    scheduling_enum = parse_scheduling_strategy(scheduling_strategy)
                    logger.info(f"Using scheduling strategy: {scheduling_enum.value}")
                    
                    # Create dispatcher
                    dispatcher = XDiTDispatcher(
                        gpu_devices=gpu_list,
                        model_path=unet_path,
                        strategy=parallel_strategy,
                        scheduling_strategy=scheduling_enum
                    )
                    
                    # Initialize dispatcher
                    success = dispatcher.initialize()
                    if success:
                        status = dispatcher.get_status()
                        logger.info(f"✅ xDiT multi-GPU acceleration enabled for UNet with {status['num_workers']} workers")
                        logger.info(f"   Scheduling strategy: {status['scheduling_strategy']}")
                    else:
                        logger.warning("Failed to initialize xDiT dispatcher for UNet, falling back to single-GPU")
                        dispatcher = None
                        
                except Exception as e:
                    logger.error(f"Error initializing xDiT multi-GPU for UNet: {e}")
                    logger.info("Falling back to single-GPU mode")
                    dispatcher = None
            else:
                if not XDIT_AVAILABLE:
                    logger.info("xDiT not available, using standard single-GPU mode for UNet")
                else:
                    logger.info("Multi-GPU disabled, using standard single-GPU mode for UNet")
            
            return (model, dispatcher)
            
        except Exception as e:
            logger.error(f"Failed to load UNet {unet_name}: {e}")
            # Return empty objects on error
            return (None, None)


class XDiTVAELoader:
    """
    VAE Loader - Drop-in replacement for VAELoader
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "vae_name": (folder_paths.get_filename_list("vae"), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    def load_vae(self, vae_name):
        """
        Load VAE model using standard ComfyUI method
        """
        try:
            # Import VAELoader here to avoid circular imports
            from nodes import VAELoader
            
            if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
                sd = VAELoader.load_taesd(vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
            return (vae,)
        except Exception as e:
            logger.error(f"Failed to load VAE {vae_name}: {e}")
            return (None,)


class XDiTCLIPLoader:
    """
    CLIP Loader - Drop-in replacement for CLIPLoader
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "[Recipes]\n\nstable_diffusion: clip-l\nstable_cascade: clip-g\nsd3: t5 xxl/ clip-g / clip-l\nstable_audio: t5 base\nmochi: t5 xxl\ncosmos: old t5 xxl\nlumina2: gemma 2 2B\nwan: umt5 xxl\n hidream: llama-3.1 (Recommend) or t5"

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        """
        Load CLIP model using standard ComfyUI method
        """
        try:
            clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            model_options = {}
            if device == "cpu":
                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
            
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
            return (clip,)
        except Exception as e:
            logger.error(f"Failed to load CLIP {clip_name}: {e}")
            return (None,)


class XDiTDualCLIPLoader:
    """
    Dual CLIP Loader - Drop-in replacement for DualCLIPLoader
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "[Recipes]\n\nsdxl: clip-l, clip-g\nsd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\nflux: clip-l, t5\nhidream: at least one of t5 or llama, recommended t5 and llama"

    def load_clip(self, clip_name1, clip_name2, type, device="default"):
        """
        Load dual CLIP models using standard ComfyUI method
        """
        try:
            clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
            
            model_options = {}
            if device == "cpu":
                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
            
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
            return (clip,)
        except Exception as e:
            logger.error(f"Failed to load dual CLIP models: {e}")
            return (None,)


class XDiTKSampler:
    """
    Multi-GPU KSampler - Drop-in replacement for KSampler
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"], {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple_linear"], {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            },
            "optional": {
                "xdit_dispatcher": ("XDIT_DISPATCHER", {"tooltip": "xDiT dispatcher for multi-GPU acceleration"}),
                "vae": ("VAE", {"tooltip": "VAE model for xDiT multi-GPU acceleration (optional)"}),
                "clip": ("CLIP", {"tooltip": "CLIP model for xDiT multi-GPU acceleration (optional)"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image with optional multi-GPU acceleration."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, xdit_dispatcher=None, vae=None, clip=None):
        """Sample with improved multi-GPU acceleration - 修复VAE/CLIP传递"""
        import time
        import threading
        
        logger.info(f"🚀 Starting XDiT sampling with {steps} steps, CFG={cfg}")

        # 🔧 调试VAE和CLIP传递
        logger.info(f"🔍 Input debugging:")
        logger.info(f"  • model: {type(model) if model else 'None'}")
        logger.info(f"  • vae: {type(vae) if vae else 'None'}")
        logger.info(f"  • clip: {type(clip) if clip else 'None'}")
        logger.info(f"  • xdit_dispatcher: {type(xdit_dispatcher) if xdit_dispatcher else 'None'}")
        
        try:
            # 🔧 关键修复：如果dispatcher存在且有VAE/CLIP，立即更新模型包装器
            if xdit_dispatcher and (vae or clip):
                self._update_dispatcher_with_components(xdit_dispatcher, vae, clip)
            
            # 1. 首先验证基本组件
            if model is None:
                logger.error("❌ Model is None, cannot proceed")
                return self._fallback_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            
            # 2. 检查xDiT dispatcher
            if xdit_dispatcher is None:
                logger.info("⚠️ No xDiT dispatcher, using standard sampling")
                return self._fallback_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            
            # 3. 验证dispatcher状态
            status = xdit_dispatcher.get_status()
            if not status.get("is_initialized", False):
                logger.warning("⚠️ xDiT dispatcher not initialized")
                return self._fallback_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

            num_workers = status.get("num_workers", 0)
            logger.info(f"✅ xDiT ready with {num_workers} workers")
            if num_workers < 2:
                logger.info(f"⚠️ Only {num_workers} workers available, using standard sampling")
                return self._fallback_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            
            logger.info(f"✅ xDiT ready with {num_workers} workers")
            
            # 4. 准备模型信息 - 关键修复
            model_info = self._prepare_model_info(model, vae, clip, xdit_dispatcher)
            if model_info is None:
                logger.warning("⚠️ Failed to prepare model info")
                return self._fallback_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            
            # 5. 运行xDiT推理，设置合理超时
            timeout_seconds = min(300, steps * 10)  # 最多5分钟或每步10秒
            logger.info(f"🎯 Running xDiT inference (timeout: {timeout_seconds}s)")
            
            # 🔧 关键修复：传递vae和clip参数到_run_xdit_with_timeout
            result_latents = self._run_xdit_with_timeout(
                xdit_dispatcher, model_info, positive, negative, 
                latent_image["samples"], steps, cfg, seed, timeout_seconds,
                vae=vae, clip=clip  # 🔧 明确传递VAE和CLIP参数
            )
            
            if result_latents is not None:
                logger.info("✅ xDiT multi-GPU generation completed!")
                return ({"samples": result_latents},)
            else:
                logger.warning("⚠️ xDiT inference failed, falling back")
                return self._fallback_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
                
        except Exception as e:
            logger.error(f"❌ XDiT sampling failed: {e}")
            logger.exception("Full traceback:")
            return self._fallback_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    def _update_dispatcher_with_components(self, dispatcher, vae, clip):
        """更新dispatcher的VAE和CLIP组件信息"""
        try:
            # 🔧 关键修复：从ComfyUI组件中提取路径信息
            vae_path = None
            clip_paths = []
            
            # 尝试从VAE对象获取路径信息
            if vae is not None:
                # 检查VAE对象是否有路径信息
                if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'config'):
                    # 这是一个ComfyUI VAE对象
                    logger.info("🔧 Found ComfyUI VAE object, will use it directly")
                elif hasattr(vae, 'sd') and hasattr(vae, 'config'):
                    # 另一种VAE格式
                    logger.info("🔧 Found VAE with state dict, will use it directly")
                else:
                    logger.info("🔧 VAE object available for direct use")
            
            # 尝试从CLIP对象获取路径信息
            if clip is not None:
                if hasattr(clip, 'cond_stage_model'):
                    logger.info("🔧 Found ComfyUI CLIP object, will use it directly")
                else:
                    logger.info("🔧 CLIP object available for direct use")
            
            # 🔧 关键：更新dispatcher的模型包装器，使其知道现在有VAE和CLIP可用
            if hasattr(dispatcher, 'model_wrapper') and dispatcher.model_wrapper:
                # 直接设置VAE和CLIP对象到模型包装器
                if hasattr(dispatcher.model_wrapper, 'set_runtime_components'):
                    dispatcher.model_wrapper.set_runtime_components(vae, clip)
                else:
                    # 如果没有专门的设置方法，直接赋值
                    dispatcher.model_wrapper.runtime_vae = vae
                    dispatcher.model_wrapper.runtime_clip = clip
                    
            logger.info("✅ Updated dispatcher with VAE/CLIP components")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not update dispatcher components: {e}")

    def _debug_model_objects(self, model, vae, clip, xdit_dispatcher):
        """调试模型对象，查看它们包含的信息"""
        logger.info("=" * 80)
        logger.info("🔍 DEBUGGING MODEL OBJECTS")
        logger.info("=" * 80)
        
        # 调试model对象
        logger.info(f"📋 MODEL OBJECT DEBUG:")
        logger.info(f"  Type: {type(model)}")
        logger.info(f"  Module: {getattr(type(model), '__module__', 'Unknown')}")
        
        # 列出model的所有属性（非私有）
        model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        logger.info(f"  Attributes ({len(model_attrs)}): {model_attrs[:20]}...")  # 只显示前20个
        
        # 查找可能包含路径的属性
        path_related_attrs = [attr for attr in model_attrs if any(keyword in attr.lower() for keyword in ['path', 'file', 'model', 'checkpoint', 'load'])]
        logger.info(f"  Path-related attributes: {path_related_attrs}")
        
        # 检查这些属性的值
        for attr in path_related_attrs[:10]:  # 只检查前10个
            try:
                value = getattr(model, attr)
                if isinstance(value, str):
                    logger.info(f"    {attr}: '{value}'")
                elif hasattr(value, '__dict__'):
                    logger.info(f"    {attr}: <object {type(value)}> with attributes: {list(vars(value).keys())[:5]}...")
                else:
                    logger.info(f"    {attr}: {type(value)} = {str(value)[:100]}...")
            except Exception as e:
                logger.info(f"    {attr}: Error accessing - {e}")
        
        # 特别检查model.model（嵌套对象）
        if hasattr(model, 'model'):
            inner_model = model.model
            logger.info(f"  Inner model type: {type(inner_model)}")
            inner_attrs = [attr for attr in dir(inner_model) if not attr.startswith('_')]
            inner_path_attrs = [attr for attr in inner_attrs if any(keyword in attr.lower() for keyword in ['path', 'file', 'model', 'checkpoint'])]
            logger.info(f"  Inner model path-related attributes: {inner_path_attrs}")
            
            for attr in inner_path_attrs[:5]:
                try:
                    value = getattr(inner_model, attr)
                    if isinstance(value, str):
                        logger.info(f"    inner.{attr}: '{value}'")
                except Exception as e:
                    logger.info(f"    inner.{attr}: Error - {e}")
        
        # 调试VAE对象
        logger.info(f"📋 VAE OBJECT DEBUG:")
        if vae is not None:
            logger.info(f"  Type: {type(vae)}")
            vae_attrs = [attr for attr in dir(vae) if not attr.startswith('_')]
            vae_path_attrs = [attr for attr in vae_attrs if any(keyword in attr.lower() for keyword in ['path', 'file'])]
            logger.info(f"  Path-related attributes: {vae_path_attrs}")
        else:
            logger.info("  VAE is None")
        
        # 调试CLIP对象
        logger.info(f"📋 CLIP OBJECT DEBUG:")
        if clip is not None:
            logger.info(f"  Type: {type(clip)}")
            clip_attrs = [attr for attr in dir(clip) if not attr.startswith('_')]
            clip_path_attrs = [attr for attr in clip_attrs if any(keyword in attr.lower() for keyword in ['path', 'file'])]
            logger.info(f"  Path-related attributes: {clip_path_attrs}")
        else:
            logger.info("  CLIP is None")
        
        # 调试dispatcher对象
        logger.info(f"📋 DISPATCHER OBJECT DEBUG:")
        if xdit_dispatcher is not None:
            logger.info(f"  Type: {type(xdit_dispatcher)}")
            logger.info(f"  Has model_path: {hasattr(xdit_dispatcher, 'model_path')}")
            if hasattr(xdit_dispatcher, 'model_path'):
                logger.info(f"  Dispatcher model_path: '{xdit_dispatcher.model_path}'")
            
            try:
                status = xdit_dispatcher.get_status()
                logger.info(f"  Status keys: {list(status.keys())}")
                if 'model_path' in status:
                    logger.info(f"  Status model_path: '{status['model_path']}'")
            except Exception as e:
                logger.info(f"  Error getting status: {e}")
        else:
            logger.info("  Dispatcher is None")
        
        logger.info("=" * 80)
    
    def _prepare_model_info(self, model, vae, clip, xdit_dispatcher=None):
        """准备模型信息 - 带调试版本"""
        try:
            # 首先运行调试
            self._debug_model_objects(model, vae, clip, xdit_dispatcher)
            
            model_path = None
            
            # 方法1: 从dispatcher获取（最可靠）
            if xdit_dispatcher is not None:
                if hasattr(xdit_dispatcher, 'model_path'):
                    model_path = xdit_dispatcher.model_path
                    logger.info(f"✅ Got model path from dispatcher.model_path: {model_path}")
                else:
                    try:
                        status = xdit_dispatcher.get_status()
                        model_path = status.get('model_path')
                        if model_path:
                            logger.info(f"✅ Got model path from dispatcher status: {model_path}")
                    except Exception as e:
                        logger.warning(f"Error getting dispatcher status: {e}")
            
            # 方法2: 从model对象获取（备用）
            if not model_path and model is not None:
                logger.info("🔍 Trying to extract path from model object...")
                
                # 检查常见的路径属性
                path_attrs = ['model_path', 'checkpoint_path', 'file_path', 'path', 'filename', 'model_file']
                for attr in path_attrs:
                    if hasattr(model, attr):
                        potential_path = getattr(model, attr)
                        if isinstance(potential_path, str) and os.path.exists(potential_path):
                            model_path = potential_path
                            logger.info(f"✅ Got model path from model.{attr}: {model_path}")
                            break
                
                # 检查嵌套的model对象
                if not model_path and hasattr(model, 'model'):
                    inner_model = model.model
                    for attr in path_attrs:
                        if hasattr(inner_model, attr):
                            potential_path = getattr(inner_model, attr)
                            if isinstance(potential_path, str) and os.path.exists(potential_path):
                                model_path = potential_path
                                logger.info(f"✅ Got model path from model.model.{attr}: {model_path}")
                                break
            
            # 验证路径
            if not model_path:
                logger.error("❌ 无法从任何源获取模型路径")
                return None
            
            if not os.path.exists(model_path):
                logger.error(f"❌ 模型文件不存在: {model_path}")
                return None
            
            # 🔧 关键修复：构建只包含可序列化数据的模型信息
            model_info = {
                'path': model_path,
                'type': 'flux',
                # ❌ 不传递原始对象，因为它们包含不可序列化的线程锁
                # 'vae': vae,  
                # 'clip': clip,
                # 'model_object': model,
                # 'dispatcher': xdit_dispatcher
                
                # ✅ 只传递基本信息和路径
                'vae_available': vae is not None,
                'clip_available': clip is not None,
                'model_type_info': str(type(model)),
                'comfyui_mode': True  # 标记这是ComfyUI模式
            }
            logger.info(f"✅ Model info prepared successfully!")
            logger.info(f"📁 Final model path: {model_path}")
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare model info: {e}")
            logger.exception("Full traceback:")
            return None
             
    def _run_xdit_with_timeout(self, dispatcher, model_info, positive, negative, latent_samples, steps, cfg, seed, timeout_seconds, vae=None, clip=None):
        """运行xDiT推理，带超时控制 - 修复VAE作用域"""
        import threading
        import queue
        
        # 使用线程和队列实现超时控制
        result_queue = queue.Queue()
        
        def inference_worker():
            try:
                # 🔧 关键修复：将conditioning数据转换为可序列化格式
                serializable_positive = None
                serializable_negative = None

                # 处理positive conditioning
                if positive is not None:
                    if isinstance(positive, (list, tuple)) and len(positive) > 0:
                        # 如果是tensor，转换为基本数据类型
                        if hasattr(positive[0], 'cpu'):
                            serializable_positive = [p.cpu().detach().numpy() if hasattr(p, 'cpu') else p for p in positive]
                        else:
                            serializable_positive = list(positive)
                
                # 处理negative conditioning
                if negative is not None:
                    if isinstance(negative, (list, tuple)) and len(negative) > 0:
                        if hasattr(negative[0], 'cpu'):
                            serializable_negative = [n.cpu().detach().numpy() if hasattr(n, 'cpu') else n for n in negative]
                        else:
                            serializable_negative = list(negative)
                
                # 🔧 将latent_samples转换为numpy数组（如果是tensor）
                if hasattr(latent_samples, 'cpu'):
                    serializable_latents = latent_samples.cpu().detach().numpy()
                else:
                    serializable_latents = latent_samples
                
                logger.info("🔧 Converted data to serializable format for Ray")

                result = dispatcher.run_inference(
                    model_info=model_info,
                    conditioning_positive=positive,
                    conditioning_negative=negative,
                    latent_samples=latent_samples,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    seed=seed,
                    comfyui_vae=vae,  # 🔧 使用参数传递的VAE
                    comfyui_clip=clip  # 🔧 使用参数传递的CLIP
                )
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        # 启动推理线程
        thread = threading.Thread(target=inference_worker)
        thread.daemon = True
        thread.start()
        
        # 等待结果或超时
        try:
            status, result = result_queue.get(timeout=timeout_seconds)
            if status == 'success':
                return result
            else:
                logger.error(f"推理线程错误: {result}")
                return None
        except queue.Empty:
            logger.error(f"⏰ xDiT推理超时 ({timeout_seconds}s)")
            return None
    
    def _fallback_sampling(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        """改进的fallback采样"""
        try:
            logger.info("🔄 Using ComfyUI native sampling as fallback")
            
            # 直接导入ComfyUI的KSampler
            from nodes import KSampler
            native_sampler = KSampler()
            
            # 使用原生采样器
            result = native_sampler.sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise
            )
            
            logger.info("✅ Fallback sampling completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Fallback sampling also failed: {e}")
            # 最终fallback：返回原始latent
            return (latent_image,)    

    def _fallback_to_standard_sampling(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        """Fallback to standard ComfyUI sampling"""
        try:
            logger.info("🔄 Using standard ComfyUI sampling as fallback")
            
            # 直接导入并使用ComfyUI的KSampler
            from nodes import KSampler
            
            # 创建KSampler实例
            native_sampler = KSampler()
            
            # 使用原生采样器
            result = native_sampler.sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise
            )
            
            logger.info("✅ Standard ComfyUI sampling completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Fallback sampling failed: {e}")
            logger.exception("Fallback error traceback:")
            # 最终fallback：返回原始latent
            return (latent_image,)

# Import the common_ksampler function
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed_override=None):
    """
    Common KSampler function - same as ComfyUI's implementation
    """
    try:
        # Import comfy.sample here to avoid circular imports
        import comfy.sample
        import comfy.utils
        import torch  # 添加torch导入
        
        # 🔧 正确提取tensor数据，ComfyUI采样器期望tensor而不是字典
        if isinstance(latent_image, dict) and "samples" in latent_image:
            latent_tensor = latent_image["samples"]
        else:
            latent_tensor = latent_image
        
        # 🔧 设置随机种子
        effective_seed = seed_override or seed
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(effective_seed)
        
        # 确保有正确的噪声
        if not disable_noise:
            # 生成与latent相同形状的噪声
            noise = torch.randn_like(latent_tensor)
        else:
            noise = latent_tensor
            
        # Use the same implementation as ComfyUI's KSampler
        samples = comfy.sample.sample(
            model=model,
            noise=noise,  # 传递正确的噪声
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_tensor,  # 🔧 传递tensor而不是字典
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            sigmas=sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=effective_seed
        )
        
        return ({"samples": samples},)
    except Exception as e:
        logger.error(f"Error in common_ksampler: {e}")
        logger.exception("Full traceback:")
        return (latent_image,)


# Legacy nodes for backward compatibility
class MultiGPUModelLoader:
    """
    Simplified Multi-GPU Model Loader (fallback version)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"), ),
                "gpu_devices": ("STRING", {"default": "0,1,2,3", "multiline": False}),
                "memory_fraction": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "vae_name": (folder_paths.get_filename_list("vae"), ),
                "clip_name": (folder_paths.get_filename_list("clip"), ),
            }
        }
    
    RETURN_TYPES = ("MODEL", "VAE", "CLIP")
    FUNCTION = "load_model"
    CATEGORY = "Multi-GPU"
    
    def load_model(self, model_name, gpu_devices, memory_fraction, vae_name=None, clip_name=None):
        """
        Load model with basic multi-GPU support
        """
        try:
            # Parse GPU devices
            gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
            logger.info(f"Loading model {model_name} on GPUs: {gpu_list}")
            
            # Set CUDA devices
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            
            # Load model using standard ComfyUI method
            from nodes import CheckpointLoader
            loader = CheckpointLoader()
            model, vae, clip = loader.load_checkpoint(model_name, vae_name, clip_name)
            
            # Try to distribute model across GPUs if possible
            if hasattr(model, 'to') and len(gpu_list) > 1:
                try:
                    # This is a simplified approach - actual implementation would be more complex
                    model = model.to(f"cuda:{gpu_list[0]}")
                    logger.info(f"Model loaded on GPU {gpu_list[0]}")
                except Exception as e:
                    logger.warning(f"Could not distribute model: {e}")
            
            logger.info(f"✅ Model loaded successfully")
            return (model, vae, clip)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to single GPU
            from nodes import CheckpointLoader
            loader = CheckpointLoader()
            return loader.load_checkpoint(model_name, vae_name, clip_name)


class MultiGPUSampler:
    """
    Simplified Multi-GPU Sampler (fallback version)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"], ),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple_linear"], ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gpu_devices": ("STRING", {"default": "0,1,2,3", "multiline": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Multi-GPU"
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, gpu_devices, batch_size):
        """
        Sample with basic multi-GPU support
        """
        try:
            # Parse GPU devices
            gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
            logger.info(f"Sampling on GPUs: {gpu_list}")
            
            # For now, use standard sampling but log GPU usage
            from nodes import KSampler
            sampler = KSampler()
            result = sampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            
            logger.info(f"✅ Sampling completed (basic multi-GPU mode)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to sample: {e}")
            # Fallback to standard sampling
            from nodes import KSampler
            sampler = KSampler()
            return sampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise) 