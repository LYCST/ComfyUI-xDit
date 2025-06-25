#!/bin/bash

# 简单的ComfyUI启动脚本 - 只使用3、4、5号显卡
echo "🚀 Starting ComfyUI with TACO-DiT "

# 设置环境变量
export XFUSER_ENABLE_FLASH_ATTENTION=1
export XFUSER_ENABLE_CACHE=1
export COMFYUI_ENABLE_TACO_DIT=1

echo "URL: http://0.0.0.0:12411"
echo "Press Ctrl+C to stop"

# 启动ComfyUI
python main.py --listen 0.0.0.0 --port 12411