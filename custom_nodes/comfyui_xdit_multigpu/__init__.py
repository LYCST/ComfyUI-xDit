"""
ComfyUI xDiT Multi-GPU Acceleration Plugin
==========================================

This plugin provides multi-GPU acceleration for ComfyUI using xDiT framework.
It includes drop-in replacements for standard ComfyUI nodes with optional multi-GPU acceleration.

Author: Your Name
Version: 1.0.0
"""

import os
import sys
import torch
import logging
from typing import Dict, Any, Optional, List

# Add xDiT path to Python path
xdit_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "xDiT")
if os.path.exists(xdit_path):
    sys.path.insert(0, xdit_path)

try:
    import xfuser
    XDIT_AVAILABLE = True
    print("✅ xDiT framework loaded successfully")
except ImportError as e:
    XDIT_AVAILABLE = False
    print(f"⚠️  xDiT framework not available: {e}")
    print("   Plugin will fall back to single-GPU mode")

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
    print("✅ Ray framework loaded successfully")
except ImportError as e:
    RAY_AVAILABLE = False
    print(f"⚠️  Ray framework not available: {e}")
    print("   Plugin will use fallback workers")

# Import our custom nodes
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

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Drop-in replacements for standard nodes
    "XDiTCheckpointLoader": XDiTCheckpointLoader,
    "XDiTUNetLoader": XDiTUNetLoader,
    "XDiTVAELoader": XDiTVAELoader,
    "XDiTCLIPLoader": XDiTCLIPLoader,
    "XDiTDualCLIPLoader": XDiTDualCLIPLoader,
    "XDiTKSampler": XDiTKSampler,
    
    # Legacy nodes for backward compatibility
    "MultiGPUModelLoader": MultiGPUModelLoader,
    "MultiGPUSampler": MultiGPUSampler,
}

# Node display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Drop-in replacements with clear naming
    "XDiTCheckpointLoader": "Load Checkpoint (xDiT Multi-GPU)",
    "XDiTUNetLoader": "Load UNet (xDiT Multi-GPU)",
    "XDiTVAELoader": "Load VAE (xDiT)",
    "XDiTCLIPLoader": "Load CLIP (xDiT)",
    "XDiTDualCLIPLoader": "Load Dual CLIP (xDiT)",
    "XDiTKSampler": "KSampler (xDiT Multi-GPU)",
    
    # Legacy nodes
    "MultiGPUModelLoader": "Multi-GPU Model Loader",
    "MultiGPUSampler": "Multi-GPU Sampler",
}

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

# Version info
__version__ = "1.0.0"
__author__ = "Your Name"

print(f"🚀 ComfyUI xDiT Multi-GPU Plugin v{__version__} loaded successfully!")
print("   📋 Drop-in replacements for standard ComfyUI nodes:")
print("      • XDiTCheckpointLoader -> CheckpointLoaderSimple")
print("      • XDiTUNetLoader -> UNetLoader")
print("      • XDiTVAELoader -> VAELoader")
print("      • XDiTCLIPLoader -> CLIPLoader")
print("      • XDiTDualCLIPLoader -> DualCLIPLoader")
print("      • XDiTKSampler -> KSampler")

if XDIT_AVAILABLE:
    print("   ✅ Multi-GPU acceleration enabled")
    print("   📊 Available parallel strategies: PipeFusion, USP, Hybrid, Tensor, CFG")
    if RAY_AVAILABLE:
        print("   🎯 Ray-based distributed computing enabled")
        print("   📈 Available scheduling strategies: round_robin, least_loaded, weighted_round_robin, adaptive")
    else:
        print("   ⚠️  Ray not available, using fallback workers")
else:
    print("   ⚠️  Running in single-GPU mode (xDiT not available)")

print("   💡 Usage: Simply replace standard nodes with xDiT versions for automatic multi-GPU acceleration")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 