#!/bin/bash
# ComfyUI xDiT多GPU加速安装脚本
# 作者: AI助手
# 用途: 自动安装依赖并测试xDiT集成

set -e  # 遇到错误立即退出

echo "🚀 ComfyUI xDiT多GPU加速安装脚本"
echo "=================================="

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "comfyui-xdit" ]]; then
    echo "❌ 错误：请先激活conda环境"
    echo "   运行: conda activate comfyui-xdit"
    exit 1
fi

echo "✅ 检测到正确的conda环境: $CONDA_DEFAULT_ENV"

# 创建日志目录
LOG_FILE="/home/shuzuan/prj/ComfyUI-xDit/error.log"
echo "📝 日志文件: $LOG_FILE"

# 函数：记录日志
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 开始安装
log_with_timestamp "开始安装xDiT依赖包..."

echo "🔧 步骤1: 更新pip"
pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"

echo "🔧 步骤2: 安装基础依赖"
pip install -r requirements.txt 2>&1 | tee -a "$LOG_FILE"

echo "🔧 步骤3: 安装xDiT相关包"
log_with_timestamp "安装xfuser..."
pip install xfuser 2>&1 | tee -a "$LOG_FILE"

log_with_timestamp "安装ray..."
pip install ray 2>&1 | tee -a "$LOG_FILE"

log_with_timestamp "安装flash-attn (可能需要较长时间)..."
pip install flash-attn 2>&1 | tee -a "$LOG_FILE" || {
    log_with_timestamp "警告: flash-attn安装失败，这是可选的依赖"
}

echo "🧪 步骤4: 运行集成测试"
log_with_timestamp "开始运行集成测试..."
python test_xdit_integration.py 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "🎉 安装完成！"
echo "📄 完整日志已保存到: $LOG_FILE"
echo ""
echo "💡 下一步:"
echo "   1. 启动ComfyUI: python main.py"
echo "   2. 在ComfyUI中查找'xDiT'相关节点"
echo "   3. 替换标准节点使用多GPU加速"
echo ""
echo "🔍 如果遇到问题，请查看日志文件: $LOG_FILE" 