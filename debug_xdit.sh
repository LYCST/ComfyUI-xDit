#!/bin/bash
# ComfyUI xDiT调试脚本
# 用途: 查看日志、诊断问题、运行测试

LOG_FILE="/home/shuzuan/prj/ComfyUI-xDit/error.log"

# 显示帮助信息
show_help() {
    echo "🔧 ComfyUI xDiT调试工具"
    echo "======================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示帮助信息"
    echo "  -l, --log      查看最新日志"
    echo "  -t, --tail     实时监控日志"
    echo "  -c, --clear    清空日志文件"
    echo "  -e, --env      检查环境状态"
    echo "  -r, --test     运行集成测试"
    echo "  -g, --gpu      显示GPU信息"
    echo "  -s, --status   显示系统状态"
    echo ""
    echo "示例:"
    echo "  $0 -l          # 查看最新的50行日志"
    echo "  $0 -t          # 实时监控日志更新"
    echo "  $0 -e          # 检查conda环境和依赖"
    echo "  $0 -r          # 运行完整的集成测试"
}

# 检查conda环境
check_env() {
    echo "🔍 检查环境状态..."
    echo "==================="
    
    echo "📦 Conda环境: $CONDA_DEFAULT_ENV"
    if [[ "$CONDA_DEFAULT_ENV" != "comfyui-xdit" ]]; then
        echo "❌ 警告: 当前不在comfyui-xdit环境中"
        echo "   请运行: conda activate comfyui-xdit"
    else
        echo "✅ Conda环境正确"
    fi
    
    echo ""
    echo "🐍 Python版本:"
    python --version
    
    echo ""
    echo "📚 关键依赖检查:"
    
    # 检查PyTorch
    python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch未安装"
    
    # 检查xfuser
    python -c "import xfuser; print('✅ xfuser: 已安装')" 2>/dev/null || echo "❌ xfuser未安装"
    
    # 检查ray
    python -c "import ray; print(f'✅ Ray: {ray.__version__}')" 2>/dev/null || echo "❌ Ray未安装"
    
    # 检查flash_attn
    python -c "import flash_attn; print('✅ Flash Attention: 已安装')" 2>/dev/null || echo "⚠️  Flash Attention未安装(可选)"
    
    echo ""
    echo "🖥️  CUDA环境:"
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
}

# 显示GPU信息
show_gpu_info() {
    echo "🖥️  GPU信息"
    echo "============"
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        echo "❌ nvidia-smi不可用"
    fi
    
    echo ""
    echo "🔧 PyTorch GPU信息:"
    python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f'GPU {i}: {props.name} ({memory_gb:.1f} GB)')
else:
    print('❌ CUDA不可用')
"
}

# 显示系统状态
show_status() {
    echo "📊 系统状态"
    echo "==========="
    
    echo "🕐 当前时间: $(date)"
    echo "📁 工作目录: $(pwd)"
    echo "💾 磁盘使用:"
    df -h . | tail -1
    
    echo ""
    echo "🧠 内存使用:"
    free -h
    
    echo ""
    echo "📋 最近的进程:"
    ps aux | grep -E "(python|comfy|xdit)" | grep -v grep | head -5
}

# 运行测试
run_test() {
    echo "🧪 运行xDiT集成测试..."
    echo "===================="
    
    if [[ "$CONDA_DEFAULT_ENV" != "comfyui-xdit" ]]; then
        echo "❌ 错误：请先激活conda环境"
        echo "   运行: conda activate comfyui-xdit"
        exit 1
    fi
    
    echo "开始测试..." | tee -a "$LOG_FILE"
    python test_xdit_integration.py 2>&1 | tee -a "$LOG_FILE"
    echo "测试完成，结果已保存到日志文件"
}

# 主逻辑
case "$1" in
    -h|--help)
        show_help
        ;;
    -l|--log)
        echo "📄 最新日志 (最后50行):"
        echo "====================="
        if [[ -f "$LOG_FILE" ]]; then
            tail -n 50 "$LOG_FILE"
        else
            echo "❌ 日志文件不存在: $LOG_FILE"
        fi
        ;;
    -t|--tail)
        echo "📄 实时监控日志 (按Ctrl+C退出):"
        echo "=========================="
        if [[ -f "$LOG_FILE" ]]; then
            tail -f "$LOG_FILE"
        else
            echo "❌ 日志文件不存在: $LOG_FILE"
            echo "等待日志文件创建..."
            touch "$LOG_FILE"
            tail -f "$LOG_FILE"
        fi
        ;;
    -c|--clear)
        echo "🗑️  清空日志文件..."
        > "$LOG_FILE"
        echo "✅ 日志文件已清空"
        ;;
    -e|--env)
        check_env
        ;;
    -r|--test)
        run_test
        ;;
    -g|--gpu)
        show_gpu_info
        ;;
    -s|--status)
        show_status
        ;;
    "")
        echo "❌ 错误: 请指定选项"
        echo ""
        show_help
        exit 1
        ;;
    *)
        echo "❌ 错误: 未知选项 '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac 