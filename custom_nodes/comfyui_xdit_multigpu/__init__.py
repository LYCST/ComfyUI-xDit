"""
ComfyUI xDiT Multi-GPU Plugin
=============================

A ComfyUI plugin that provides multi-GPU acceleration for diffusion models
using the xDiT framework with drop-in replacement nodes.
"""

import os
import sys
import logging
from pathlib import Path


# 设置环境变量来解决tokenizer警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置日志
logger = logging.getLogger(__name__)

# 添加下面这些行：
# 调试控制
DEBUG_LEVEL = os.environ.get('XDIT_DEBUG', '0')
if DEBUG_LEVEL == '1':
    logging.getLogger("xfuser").setLevel(logging.DEBUG)
    logging.getLogger("ray").setLevel(logging.DEBUG)
    logging.getLogger("torch.distributed").setLevel(logging.DEBUG)
    logger.info("🔍 xDiT Debug mode enabled")

# 检查xDiT可用性
XDIT_AVAILABLE = False
try:
    import xfuser
    XDIT_AVAILABLE = True
    logger.info("✅ xDiT framework loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  xDiT framework not available: {e}")
    logger.info("   Plugin will fall back to single-GPU mode")

# 检查Ray可用性
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
    logger.info("✅ Ray framework loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  Ray framework not available: {e}")
    logger.info("   Plugin will fall back to single-GPU mode")

# 导入节点
try:
    from .nodes import (
        XDiTCheckpointLoader,
        XDiTUNetLoader,
        XDiTVAELoader,
        XDiTCLIPLoader,
        XDiTDualCLIPLoader,
        XDiTKSampler,
        MultiGPUModelLoader,
        MultiGPUSampler
    )
    
    # 注册节点
    NODE_CLASS_MAPPINGS = {
        "XDiTCheckpointLoader": XDiTCheckpointLoader,
        "XDiTUNetLoader": XDiTUNetLoader,
        "XDiTVAELoader": XDiTVAELoader,
        "XDiTCLIPLoader": XDiTCLIPLoader,
        "XDiTDualCLIPLoader": XDiTDualCLIPLoader,
        "XDiTKSampler": XDiTKSampler,
        "MultiGPUModelLoader": MultiGPUModelLoader,
        "MultiGPUSampler": MultiGPUSampler
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "XDiTCheckpointLoader": "Load Checkpoint (xDiT)",
        "XDiTUNetLoader": "Load UNet (xDiT)",
        "XDiTVAELoader": "Load VAE (xDiT)",
        "XDiTCLIPLoader": "Load CLIP (xDiT)",
        "XDiTDualCLIPLoader": "Load Dual CLIP (xDiT)",
        "XDiTKSampler": "KSampler (xDiT)",
        "MultiGPUModelLoader": "Multi-GPU Model Loader",
        "MultiGPUSampler": "Multi-GPU Sampler"
    }
    
    logger.info("🚀 ComfyUI xDiT Multi-GPU Plugin v1.0.0 loaded successfully!")
    logger.info("   📋 Drop-in replacements for standard ComfyUI nodes:")
    logger.info("      • XDiTCheckpointLoader -> CheckpointLoaderSimple")
    logger.info("      • XDiTUNetLoader -> UNetLoader")
    logger.info("      • XDiTVAELoader -> VAELoader")
    logger.info("      • XDiTCLIPLoader -> CLIPLoader")
    logger.info("      • XDiTDualCLIPLoader -> DualCLIPLoader")
    logger.info("      • XDiTKSampler -> KSampler")
    if XDIT_AVAILABLE:
        logger.info("   ✅ Multi-GPU acceleration enabled")
        logger.info("   📊 Available parallel strategies: PipeFusion, USP, Hybrid, Tensor, CFG")
    if RAY_AVAILABLE:
        logger.info("   🎯 Ray-based distributed computing enabled")
        logger.info("   📈 Available scheduling strategies: round_robin, least_loaded, weighted_round_robin, adaptive")
    logger.info("   💡 Usage: Simply replace standard nodes with xDiT versions for automatic multi-GPU acceleration")
    
except Exception as e:
    logger.error(f"❌ Failed to load xDiT Multi-GPU Plugin: {e}")
    logger.exception("Full traceback:")
    
    # 提供空的映射作为fallback
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"
__author__ = "ComfyUI xDiT Team"
__description__ = "Multi-GPU acceleration for ComfyUI using xDiT framework"

# Custom types for ComfyUI
CUSTOM_NODES = {
    "XDIT_DISPATCHER": {
        "name": "XDIT_DISPATCHER",
        "description": "xDiT Dispatcher for Ray-based multi-GPU acceleration",
        "category": "xDiT"
    }
}

# Web extensions (if any)
WEB_DIRECTORY = "./web" 