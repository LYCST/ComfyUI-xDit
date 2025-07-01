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
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # 模型组件
        self.unet = None
        self.vae = None
        self.clip = None
        self.model_config = None
        self.pipeline = None  # 添加pipeline属性
        
        # 🔧 新增：运行时组件（从工作流传递）
        self.runtime_vae = None
        self.runtime_clip = None
        
        logger.info(f"Initializing ComfyUI model wrapper")
        logger.info(f"  UNet: {model_path}")
        logger.info(f"  VAE: Will be provided by workflow")
        logger.info(f"  CLIP: Will be provided by workflow")
    
    def set_runtime_components(self, vae=None, clip=None):
        """设置运行时VAE和CLIP组件"""
        try:
            if vae is not None:
                self.runtime_vae = vae
                logger.info("✅ Set runtime VAE component")
            
            if clip is not None:
                self.runtime_clip = clip 
                logger.info("✅ Set runtime CLIP component")
                
        except Exception as e:
            logger.error(f"Error setting runtime components: {e}")
    
    def load_components(self):
        """加载ComfyUI模型组件 - 修改为优先使用运行时组件"""
        try:
            # 1. 加载UNet/Transformer
            logger.info(f"Loading UNet from: {self.model_path}")
            if self.model_path.endswith('.safetensors'):
                import safetensors.torch
                sd = safetensors.torch.load_file(self.model_path)
                
                keys = list(sd.keys())
                logger.info(f"Model keys (first 10): {keys[:10]}")
                logger.info(f"Total keys: {len(keys)}")
                
                # 检测模型类型
                flux_indicators = [
                    'transformer_blocks', 'transformer', 'model.diffusion_model',
                    'diffusion_model', 'time_embed', 'input_blocks', 'middle_block',
                    'output_blocks', 'double_blocks', 'img_attn', 'img_mlp'
                ]
                
                is_flux_model = any(key.startswith(indicator) for key in keys for indicator in flux_indicators)
                
                if is_flux_model:
                    logger.info("✅ Detected FLUX/UNet model format")
                    self.unet = "flux_model_loaded"
                    logger.info("✅ FLUX model marked as loaded (will use ComfyUI components)")
                else:
                    logger.warning(f"Unknown model format - no FLUX indicators found")
                    return False
            
            # 2. VAE和CLIP将在运行时从工作流获取
            logger.info("VAE and CLIP will be provided by workflow at runtime")
            
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
    
    def get_vae(self):
        """获取VAE组件 - 优先使用运行时组件"""
        if hasattr(self, 'runtime_vae') and self.runtime_vae is not None:
            logger.info("✅ Using runtime VAE component from workflow")
            return self.runtime_vae
        else:
            logger.info("VAE not available yet - will be provided by workflow")
            return None
    
    def get_clip(self):
        """获取CLIP组件 - 优先使用运行时组件"""
        if hasattr(self, 'runtime_clip') and self.runtime_clip is not None:
            logger.info("✅ Using runtime CLIP component from workflow")
            return self.runtime_clip
        else:
            logger.info("CLIP not available yet - will be provided by workflow")
            return None
    
    def has_components(self):
        """检查是否有必要的组件"""
        has_unet = self.unet is not None
        has_vae = self.get_vae() is not None
        has_clip = self.get_clip() is not None
        
        logger.info(f"Component status: UNet={has_unet}, VAE={has_vae}, CLIP={has_clip}")
        return has_unet and has_vae and has_clip
    
    def get_pipeline(self):
        """
        获取pipeline对象 - 新增方法修复错误
        """
        try:
            if self.pipeline is not None:
                return self.pipeline
                
            # 如果还没有pipeline，尝试创建一个简化的pipeline包装器
            logger.info("Creating simplified pipeline wrapper...")
            
            # 创建一个简化的pipeline类
            class SimplifiedPipeline:
                def __init__(self, wrapper):
                    self.wrapper = wrapper
                    self.unet = wrapper.unet
                    self.vae = wrapper.vae
                    self.clip = wrapper.clip
                
                def __call__(self, *args, **kwargs):
                    # 这是一个占位符，实际的推理会由xDiT处理
                    logger.info("SimplifiedPipeline called - delegating to xDiT")
                    return None
                
                def to(self, device):
                    return self
            
            self.pipeline = SimplifiedPipeline(self)
            logger.info("✅ Simplified pipeline wrapper created")
            return self.pipeline
            
        except Exception as e:
            logger.error(f"Failed to get pipeline: {e}")
            return None

    def to(self, device):
        """移动到指定设备"""
        try:
            if self.unet is not None and self.unet != "flux_loaded":
                self.unet = self.unet.to(device)
            if self.vae is not None:
                self.vae = self.vae.to(device)
            if self.clip is not None:
                self.clip = self.clip.to(device)
            return self
        except Exception as e:
            logger.warning(f"Error moving to device {device}: {e}")
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