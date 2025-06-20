"""
TACO-DiT Model Wrapper

Wraps ComfyUI models with xDit parallel inference capabilities.
"""

import logging
import torch
from typing import Optional, Dict, Any, Union
from comfy.model_patcher import ModelPatcher

logger = logging.getLogger(__name__)

class TACODiTModelPatcher(ModelPatcher):
    """TACO-DiT模型包装器，继承自ComfyUI的ModelPatcher"""
    
    def __init__(self, model, load_device, offload_device, size=0, 
                 parallel_config=None, **kwargs):
        # 确保在调用父类构造函数之前设置parallel_config
        self.parallel_config = parallel_config
        self.xdit_wrapper = None
        self.model_type = None
        
        # 调用父类构造函数
        super().__init__(model, load_device, offload_device, size, **kwargs)
        
        # 设置xDit包装器
        if self.parallel_config and self.parallel_config.enabled:
            self._setup_xdit_wrapper()
    
    def state_dict(self):
        """添加state_dict方法，委托给model"""
        if hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        else:
            return self.model_state_dict()
    
    def _setup_xdit_wrapper(self):
        """设置xDit包装器"""
        try:
            # 检测模型类型
            self.model_type = self._detect_model_type()
            logger.info(f"Detected model type: {self.model_type}")
            
            # 根据模型类型选择合适的xDit包装器
            wrapper_class = self._get_wrapper_class()
            if wrapper_class:
                self.xdit_wrapper = wrapper_class(self.model)
                logger.info(f"xDit wrapper initialized: {wrapper_class.__name__}")
            else:
                logger.warning(f"No xDit wrapper available for model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to setup xDit wrapper: {e}")
            self.xdit_wrapper = None
    
    def _detect_model_type(self) -> str:
        """检测模型类型"""
        model_class_name = type(self.model).__name__.lower()
        
        # 基于模型类名检测
        if 'flux' in model_class_name:
            return 'flux'
        elif 'pixart' in model_class_name:
            return 'pixart'
        elif 'sd3' in model_class_name or 'stable_diffusion_3' in model_class_name:
            return 'sd3'
        elif 'hunyuan' in model_class_name:
            return 'hunyuan'
        elif 'cogvideo' in model_class_name:
            return 'cogvideo'
        elif 'stepvideo' in model_class_name:
            return 'stepvideo'
        elif 'latte' in model_class_name:
            return 'latte'
        elif 'consisid' in model_class_name:
            return 'consisid'
        else:
            # 尝试基于模型结构检测
            return self._detect_by_structure()
    
    def _detect_by_structure(self) -> str:
        """基于模型结构检测类型"""
        try:
            # 检查模型属性
            if hasattr(self.model, 'unet'):
                unet_class = type(self.model.unet).__name__.lower()
                if 'flux' in unet_class:
                    return 'flux'
                elif 'pixart' in unet_class:
                    return 'pixart'
                elif 'sd3' in unet_class:
                    return 'sd3'
            
            # 检查模型配置
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'model_type'):
                    return config.model_type.lower()
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Failed to detect model structure: {e}")
            return 'unknown'
    
    def _get_wrapper_class(self):
        """根据模型类型获取对应的xDit包装器"""
        try:
            # 尝试导入xDit包装器
            from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
            
            # 模型类型到包装器的映射
            wrapper_map = {
                'flux': self._get_flux_wrapper,
                'pixart': self._get_pixart_wrapper,
                'sd3': self._get_sd3_wrapper,
                'hunyuan': self._get_hunyuan_wrapper,
                'cogvideo': self._get_cogvideo_wrapper,
                'stepvideo': self._get_stepvideo_wrapper,
                'latte': self._get_latte_wrapper,
                'consisid': self._get_consisid_wrapper,
            }
            
            wrapper_getter = wrapper_map.get(self.model_type)
            if wrapper_getter:
                return wrapper_getter()
            else:
                logger.warning(f"No wrapper available for model type: {self.model_type}")
                return xFuserBaseWrapper
                
        except ImportError:
            logger.warning("xDit not available, using base wrapper")
            return None
        except Exception as e:
            logger.error(f"Failed to get wrapper class: {e}")
            return None
    
    def _get_flux_wrapper(self):
        """获取Flux包装器"""
        try:
            from xfuser import xFuserFluxPipeline
            return xFuserFluxPipeline
        except ImportError:
            logger.warning("xFuserFluxPipeline not available")
            return None
    
    def _get_pixart_wrapper(self):
        """获取PixArt包装器"""
        try:
            from xfuser import xFuserPixArtPipeline
            return xFuserPixArtPipeline
        except ImportError:
            logger.warning("xFuserPixArtPipeline not available")
            return None
    
    def _get_sd3_wrapper(self):
        """获取SD3包装器"""
        try:
            from xfuser import xFuserSD3Pipeline
            return xFuserSD3Pipeline
        except ImportError:
            logger.warning("xFuserSD3Pipeline not available")
            return None
    
    def _get_hunyuan_wrapper(self):
        """获取Hunyuan包装器"""
        try:
            from xfuser import xFuserHunyuanPipeline
            return xFuserHunyuanPipeline
        except ImportError:
            logger.warning("xFuserHunyuanPipeline not available")
            return None
    
    def _get_cogvideo_wrapper(self):
        """获取CogVideo包装器"""
        try:
            from xfuser import xFuserCogVideoPipeline
            return xFuserCogVideoPipeline
        except ImportError:
            logger.warning("xFuserCogVideoPipeline not available")
            return None
    
    def _get_stepvideo_wrapper(self):
        """获取StepVideo包装器"""
        try:
            from xfuser import xFuserStepVideoPipeline
            return xFuserStepVideoPipeline
        except ImportError:
            logger.warning("xFuserStepVideoPipeline not available")
            return None
    
    def _get_latte_wrapper(self):
        """获取Latte包装器"""
        try:
            from xfuser import xFuserLattePipeline
            return xFuserLattePipeline
        except ImportError:
            logger.warning("xFuserLattePipeline not available")
            return None
    
    def _get_consisid_wrapper(self):
        """获取ConsisID包装器"""
        try:
            from xfuser import xFuserConsisIDPipeline
            return xFuserConsisIDPipeline
        except ImportError:
            logger.warning("xFuserConsisIDPipeline not available")
            return None
    
    def forward(self, *args, **kwargs):
        """重写forward方法以支持并行推理"""
        if self.xdit_wrapper and self.parallel_config and self.parallel_config.enabled:
            try:
                # 使用xDit包装器进行并行推理
                return self.xdit_wrapper.forward(*args, **kwargs)
            except Exception as e:
                logger.error(f"xDit forward failed, falling back to original: {e}")
                return super().forward(*args, **kwargs)
        else:
            # 使用原始forward方法
            return super().forward(*args, **kwargs)
    
    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """重写load方法以支持并行加载"""
        if self.xdit_wrapper and self.parallel_config and self.parallel_config.enabled:
            try:
                # 使用xDit的并行加载
                return self._load_with_xdit(device_to, lowvram_model_memory, force_patch_weights, full_load)
            except Exception as e:
                logger.error(f"xDit load failed, falling back to original: {e}")
                return super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)
        else:
            return super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)
    
    def _load_with_xdit(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """使用xDit进行并行加载"""
        # 这里需要根据具体的xDit API实现并行加载逻辑
        # 暂时使用原始加载方法
        return super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': self.model_type,
            'parallel_enabled': self.parallel_config.enabled if self.parallel_config else False,
            'xdit_wrapper': type(self.xdit_wrapper).__name__ if self.xdit_wrapper else None,
            'model_class': type(self.model).__name__,
        }
        
        if self.parallel_config:
            info.update({
                'sequence_parallel': self.parallel_config.sequence_parallel,
                'pipeline_parallel': self.parallel_config.pipeline_parallel,
                'tensor_parallel': self.parallel_config.tensor_parallel,
                'cfg_parallel': self.parallel_config.cfg_parallel,
            })
        
        return info
    
    def __str__(self) -> str:
        return f"TACODiTModelPatcher(type={self.model_type}, parallel={self.parallel_config.enabled if self.parallel_config else False})" 