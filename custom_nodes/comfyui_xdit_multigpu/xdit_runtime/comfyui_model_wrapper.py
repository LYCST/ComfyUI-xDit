"""
ComfyUI Model Wrapper for xDiT
==============================

包装ComfyUI模型以兼容xDiT的分布式推理
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

class ComfyUIModelWrapper:
    """
    包装ComfyUI的模型组件，使其可以被xDiT使用
    """
    
    def __init__(self, model_path: str, vae_path: str = None, clip_paths: list = None):
        self.model_path = model_path
        self.vae_path = vae_path
        self.clip_paths = clip_paths or []
        
        # 模型组件
        self.unet = None
        self.vae = None
        self.clip = None
        self.model_config = None
        
        logger.info(f"Initializing ComfyUI model wrapper")
        logger.info(f"  UNet: {model_path}")
        logger.info(f"  VAE: {vae_path}")
        logger.info(f"  CLIP: {clip_paths}")
    
    def load_components(self):
        """加载ComfyUI模型组件"""
        try:
            # 1. 加载UNet/Transformer
            logger.info(f"Loading UNet from: {self.model_path}")
            if self.model_path.endswith('.safetensors'):
                # 使用ComfyUI的加载方法
                import safetensors.torch
                sd = safetensors.torch.load_file(self.model_path)
                
                # 检测模型类型
                if any(key.startswith('transformer_blocks') for key in sd.keys()):
                    # FLUX transformer
                    from diffusers import FluxTransformer2DModel
                    self.unet = FluxTransformer2DModel(
                        patch_size=1,
                        in_channels=64,
                        num_layers=19,
                        num_single_layers=38,
                        attention_head_dim=128,
                        num_attention_heads=24,
                        joint_attention_dim=4096,
                        pooled_projection_dim=768,
                        guidance_embeds=False,
                        axes_dims_rope=[16, 56, 56]
                    )
                    self.unet.load_state_dict(sd, strict=False)
                    logger.info("✅ Loaded FLUX transformer")
                else:
                    logger.warning("Unknown model format - no transformer_blocks found")
                    return False
            
            # 2. 加载VAE（如果提供且存在）
            if self.vae_path and os.path.exists(self.vae_path):
                try:
                    logger.info(f"Loading VAE from: {self.vae_path}")
                    vae_sd = comfy.utils.load_torch_file(self.vae_path)
                    self.vae = comfy.sd.VAE(sd=vae_sd)
                    logger.info("✅ Loaded VAE")
                except Exception as e:
                    logger.warning(f"Failed to load VAE: {e}")
            else:
                logger.info("VAE not provided or not found - will use ComfyUI VAE")
            
            # 3. 加载CLIP（如果提供且存在）
            if self.clip_paths:
                existing_clip_paths = [p for p in self.clip_paths if os.path.exists(p)]
                if existing_clip_paths:
                    try:
                        logger.info(f"Loading CLIP from: {existing_clip_paths}")
                        # 使用ComfyUI的CLIP加载方法
                        self.clip = comfy.sd.load_clip(
                            ckpt_paths=existing_clip_paths,
                            embedding_directory=folder_paths.get_folder_paths("embeddings"),
                            clip_type=comfy.sd.CLIPType.FLUX
                        )
                        logger.info("✅ Loaded CLIP")
                    except Exception as e:
                        logger.warning(f"Failed to load CLIP: {e}")
                else:
                    logger.info("CLIP files not found - will use ComfyUI CLIP")
            else:
                logger.info("CLIP not provided - will use ComfyUI CLIP")
            
            # 只要UNet加载成功就返回True
            if self.unet is not None:
                logger.info("✅ ComfyUI model wrapper ready (UNet loaded)")
                return True
            else:
                logger.error("Failed to load UNet/Transformer")
                return False
            
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            logger.exception("Load error:")
            return False
    
    def to(self, device):
        """移动到指定设备"""
        if self.unet is not None:
            self.unet = self.unet.to(device)
        return self
    
    def get_model_config(self):
        """获取模型配置"""
        return {
            "model_type": "flux",
            "in_channels": 64,
            "out_channels": 64,
            "latent_channels": 16,
            "sample_size": 1024
        }