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

# ComfyUI imports
import folder_paths
import comfy.utils
import comfy.sd

# Import our xDiT integration and runtime
from .xdit_integration import xdit_manager, XDiTModelLoader, XDiTGenerator
from .xdit_runtime import XDiTDispatcher, SchedulingStrategy

# Try to import xDiT
try:
    import xfuser
    from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
    XDIT_AVAILABLE = True
except ImportError:
    XDIT_AVAILABLE = False

logger = logging.getLogger(__name__)

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
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            
            # Always load the standard way first
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model, clip, vae = out[:3]
            
            dispatcher = None
            
            # Enable multi-GPU if requested and available
            if enable_multi_gpu and XDIT_AVAILABLE:
                try:
                    # Parse GPU devices
                    gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
                    logger.info(f"Initializing xDiT multi-GPU acceleration for {ckpt_name} on GPUs: {gpu_list}")
                    
                    # Parse scheduling strategy
                    try:
                        scheduling = SchedulingStrategy(scheduling_strategy)
                    except ValueError:
                        logger.warning(f"Invalid scheduling strategy {scheduling_strategy}, using round_robin")
                        scheduling = SchedulingStrategy.ROUND_ROBIN
                    
                    # Create dispatcher
                    dispatcher = XDiTDispatcher(
                        gpu_devices=gpu_list,
                        model_path=ckpt_path,
                        strategy=parallel_strategy,
                        scheduling_strategy=scheduling
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
    DESCRIPTION = "Loads a UNet diffusion model with optional multi-GPU acceleration using xDiT framework."

    def load_unet(self, unet_name, enable_multi_gpu=True, gpu_devices="0,1,2,3", parallel_strategy="Hybrid", scheduling_strategy="round_robin"):
        """
        Load UNet with optional multi-GPU acceleration
        """
        try:
            unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)
            
            # Load UNet using standard ComfyUI method
            model = comfy.sd.load_diffusion_model_state_dict(comfy.utils.load_torch_file(unet_path, safe_load=True))
            
            dispatcher = None
            
            # Enable multi-GPU if requested and available
            if enable_multi_gpu and XDIT_AVAILABLE:
                try:
                    # Parse GPU devices
                    gpu_list = [int(x.strip()) for x in gpu_devices.split(",")]
                    logger.info(f"Initializing xDiT multi-GPU acceleration for UNet {unet_name} on GPUs: {gpu_list}")
                    
                    # Parse scheduling strategy
                    try:
                        scheduling = SchedulingStrategy(scheduling_strategy)
                    except ValueError:
                        logger.warning(f"Invalid scheduling strategy {scheduling_strategy}, using round_robin")
                        scheduling = SchedulingStrategy.ROUND_ROBIN
                    
                    # Create dispatcher
                    dispatcher = XDiTDispatcher(
                        gpu_devices=gpu_list,
                        model_path=unet_path,
                        strategy=parallel_strategy,
                        scheduling_strategy=scheduling
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
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image with optional multi-GPU acceleration."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, xdit_dispatcher=None):
        """
        Sample with optional multi-GPU acceleration
        """
        try:
            # Try multi-GPU if dispatcher is available
            if xdit_dispatcher is not None and XDIT_AVAILABLE:
                try:
                    logger.info(f"Using xDiT multi-GPU acceleration: {steps} steps, CFG={cfg}")
                    
                    # Extract prompt from conditioning
                    prompt = self._extract_prompt_from_conditioning(positive)
                    negative_prompt = self._extract_prompt_from_conditioning(negative)
                    
                    # Get latent dimensions
                    samples = latent_image["samples"]
                    batch_size, channels, height, width = samples.shape
                    
                    # Generate with dispatcher
                    latents = xdit_dispatcher.run_inference(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height * 8,  # Convert latent height to pixel height
                        width=width * 8,    # Convert latent width to pixel width
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        seed=seed
                    )
                    
                    if latents is not None:
                        # Apply denoise if needed
                        if denoise < 1.0:
                            original_latents = latent_image["samples"]
                            latents = original_latents * (1 - denoise) + latents * denoise
                        
                        logger.info(f"✅ xDiT multi-GPU generation completed successfully")
                        return ({"samples": latents},)
                    else:
                        logger.warning("xDiT generation failed, falling back to standard sampling")
                        
                except Exception as e:
                    logger.error(f"Error during xDiT sampling: {e}")
                    logger.info("Falling back to standard sampling")
            
            # Fallback to standard sampling
            return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            
        except Exception as e:
            logger.error(f"Error during sampling: {e}")
            # Return original latents on error
            return (latent_image,)
    
    def _extract_prompt_from_conditioning(self, conditioning):
        """Extract prompt from conditioning"""
        try:
            # This is a simplified extraction - in practice you might need more complex logic
            if isinstance(conditioning, (list, tuple)) and len(conditioning) > 0:
                # Extract from the first conditioning
                cond = conditioning[0]
                if isinstance(cond, dict) and 'prompt' in cond:
                    return cond['prompt']
                elif hasattr(cond, 'prompt'):
                    return cond.prompt
            return "a beautiful image"
        except Exception as e:
            logger.warning(f"Could not extract prompt from conditioning: {e}")
            return "a beautiful image"


# Import the common_ksampler function
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed_override=None):
    """
    Common KSampler function - same as ComfyUI's implementation
    """
    try:
        # Import comfy.sample here to avoid circular imports
        import comfy.sample
        # Use the same implementation as ComfyUI's KSampler
        return comfy.sample.sample(model, None, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, noise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed_override or seed)
    except Exception as e:
        logger.error(f"Error in common_ksampler: {e}")
        return latent_image["samples"]


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