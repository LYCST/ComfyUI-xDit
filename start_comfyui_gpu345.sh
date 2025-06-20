#!/bin/bash

# 简单的ComfyUI启动脚本 - 只使用3、4、5号显卡
echo "🚀 Starting ComfyUI with TACO-DiT (GPUs 3,4,5)..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=3,4,5
export XFUSER_ENABLE_FLASH_ATTENTION=1
export XFUSER_ENABLE_CACHE=1
export COMFYUI_ENABLE_TACO_DIT=1

echo "Using GPUs: 3, 4, 5"
echo "URL: http://0.0.0.0:12215"
echo "Press Ctrl+C to stop"

# 启动ComfyUI
python main.py --listen 0.0.0.0 --port 12215 