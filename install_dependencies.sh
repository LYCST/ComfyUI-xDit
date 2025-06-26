#!/bin/bash
"""
xDiT依赖安装脚本
===============
确保在comfyui-xdit环境中安装所有必要依赖
"""

set -e  # 遇到错误立即退出

echo "🚀 Installing xDiT dependencies for 8x RTX 4090 setup..."

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "comfyui-xdit" ]]; then
    echo "⚠️ Warning: Not in comfyui-xdit environment!"
    echo "Please run: conda activate comfyui-xdit"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 安装基础依赖
echo "📦 Installing basic dependencies..."
pip install -r requirements.txt

# 专门安装xDiT相关包
echo "🔧 Installing xDiT packages..."
pip install einops>=0.6.0
pip install xfuser>=0.4.0
pip install ray>=2.0.0

# 安装flash-attn（可选，提升性能）
echo "⚡ Installing flash-attn..."
pip install flash-attn>=2.6.0 || echo "⚠️ flash-attn installation failed, will use pytorch attention"

# 验证安装
echo "🧪 Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import einops; print(f'Einops: {einops.__version__}')" || echo "❌ Einops not found"
python -c "import ray; print(f'Ray: {ray.__version__}')" || echo "❌ Ray not found"
python -c "import xfuser; print(f'xFuser: {xfuser.__version__}')" || echo "❌ xFuser not found"

# 检查GPU
echo "🖥️ GPU Status:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f} GB)')
else:
    print('❌ CUDA not available')
"

echo "✅ Installation completed!"
echo ""
echo "Next steps:"
echo "1. Run the test: python test_memory_fixes.py"
echo "2. Start ComfyUI: python main.py --listen 0.0.0.0 --port 12411" 