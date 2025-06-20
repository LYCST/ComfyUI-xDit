#!/bin/bash

# ComfyUI with TACO-DiT Integration Startup Script
# 集成TACO-DiT的ComfyUI启动脚本

echo "============================================================"
echo "🎯 ComfyUI with TACO-DiT Integration"
echo "============================================================"

# 设置环境变量 - 只使用3、4、5号显卡
export CUDA_VISIBLE_DEVICES=3,4,5
export XFUSER_ENABLE_FLASH_ATTENTION=1
export XFUSER_ENABLE_CACHE=1
export COMFYUI_ENABLE_TACO_DIT=1

echo "Environment variables set:"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (Using GPUs 3,4,5)"
echo "  - XFUSER_ENABLE_FLASH_ATTENTION: $XFUSER_ENABLE_FLASH_ATTENTION"
echo "  - XFUSER_ENABLE_CACHE: $XFUSER_ENABLE_CACHE"
echo "  - COMFYUI_ENABLE_TACO_DIT: $COMFYUI_ENABLE_TACO_DIT"

# 检查Python环境
echo ""
echo "Checking Python environment..."
python --version

# 检查GPU
echo ""
echo "Checking GPU availability..."
python -c "
import torch
gpu_count = torch.cuda.device_count()
print(f'GPU count: {gpu_count}')
if gpu_count > 0:
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f'GPU {i}: {props.name}, {memory_gb:.1f} GB')
"

# 检查TACO-DiT可用性
echo ""
echo "Checking TACO-DiT availability..."
python -c "
try:
    import xfuser
    print('✅ xDit available')
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
    
    from comfy.taco_dit import TACODiTConfigManager
    config_manager = TACODiTConfigManager()
    config = config_manager.config
    print(f'✅ TACO-DiT available: {config}')
    
except ImportError as e:
    print(f'❌ TACO-DiT not available: {e}')
except Exception as e:
    print(f'❌ TACO-DiT check failed: {e}')
"

# 创建自定义节点目录
echo ""
echo "Setting up custom nodes..."
mkdir -p custom_nodes

# 检查TACO-DiT节点文件
if [ ! -f "custom_nodes/TACO_DiT_Nodes.py" ]; then
    echo "Creating basic TACO-DiT nodes file..."
    cat > custom_nodes/TACO_DiT_Nodes.py << 'EOF'
"""
TACO-DiT ComfyUI Nodes (Basic Version)

Basic ComfyUI nodes for TACO-DiT integration
"""

import logging
logger = logging.getLogger(__name__)

class TACODiTLoader:
    """TACO-DiT模型加载器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_taco_dit": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_taco_dit"
    CATEGORY = "TACO-DiT"
    
    def load_taco_dit(self, model, enable_taco_dit):
        logger.info(f"TACO-DiT loader called, enabled: {enable_taco_dit}")
        return (model,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "TACODiTLoader": TACODiTLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TACODiTLoader": "TACO-DiT Loader",
}
EOF
    echo "✅ Created basic TACO-DiT nodes file"
else
    echo "✅ TACO-DiT nodes file already exists"
fi

# 启动ComfyUI
echo ""
echo "🚀 Starting ComfyUI with TACO-DiT integration..."
echo "URL: http://0.0.0.0:12215"
echo "Using GPUs: 3, 4, 5"
echo "Press Ctrl+C to stop"
echo ""

# 启动ComfyUI
python main.py --listen 0.0.0.0 --port 12215 