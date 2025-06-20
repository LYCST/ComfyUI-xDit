#!/usr/bin/env python3
"""
启动ComfyUI并使用修复版TACO-DiT节点
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境配置"""
    logger.info("检查环境配置...")
    
    # 检查conda环境
    try:
        result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
        if 'comfyui-xdit' in result.stdout:
            logger.info("✓ 找到comfyui-xdit conda环境")
        else:
            logger.warning("⚠ 未找到comfyui-xdit环境，请先创建环境")
            return False
    except Exception as e:
        logger.error(f"✗ 检查conda环境失败: {e}")
        return False
    
    # 检查TACO-DiT文件
    fixed_nodes_file = Path("custom_nodes/TACO_DiT_Fixed_Nodes.py")
    if fixed_nodes_file.exists():
        logger.info("✓ 找到修复版TACO-DiT节点文件")
    else:
        logger.error("✗ 未找到修复版TACO-DiT节点文件")
        return False
    
    # 检查TACO-DiT模块
    try:
        sys.path.insert(0, str(Path.cwd()))
        from comfy.taco_dit import TACODiTConfig
        logger.info("✓ TACO-DiT模块可用")
    except ImportError as e:
        logger.error(f"✗ TACO-DiT模块导入失败: {e}")
        return False
    
    return True

def setup_gpu_config():
    """设置GPU配置"""
    logger.info("设置GPU配置...")
    
    # 设置环境变量，只使用GPU 3、4、5
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
    logger.info("✓ 设置CUDA_VISIBLE_DEVICES=3,4,5")
    
    # 设置TACO-DiT配置
    os.environ['TACO_DIT_AUTO_DETECT'] = 'true'
    os.environ['TACO_DIT_SEQUENCE_PARALLEL_DEGREE'] = '3'
    os.environ['TACO_DIT_CFG_PARALLEL'] = 'true'
    logger.info("✓ 设置TACO-DiT环境变量")

def start_comfyui():
    """启动ComfyUI"""
    logger.info("启动ComfyUI...")
    
    try:
        # 激活conda环境并启动ComfyUI
        cmd = [
            'conda', 'run', '-n', 'comfyui-xdit',
            'python', 'main.py',
            '--listen', '0.0.0.0',
            '--port', '12215',
            '--enable-cors-header'
        ]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        logger.info("ComfyUI将在 http://localhost:12215 启动")
        logger.info("请使用 'TACO-DiT KSampler (Fixed)' 节点避免VAE通道数问题")
        
        # 启动ComfyUI
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在停止ComfyUI...")
    except Exception as e:
        logger.error(f"启动ComfyUI失败: {e}")

def show_usage_tips():
    """显示使用提示"""
    logger.info("="*60)
    logger.info("TACO-DiT 修复版使用提示")
    logger.info("="*60)
    logger.info("")
    logger.info("1. 在ComfyUI中，请使用以下节点：")
    logger.info("   - TACO-DiT Model Loader")
    logger.info("   - TACO-DiT KSampler (Fixed) ← 重要！使用修复版")
    logger.info("   - TACO-DiT Info Display")
    logger.info("   - TACO-DiT Stats Display")
    logger.info("")
    logger.info("2. 避免使用以下节点（可能有VAE问题）：")
    logger.info("   - TACO-DiT KSampler (原始版本)")
    logger.info("")
    logger.info("3. 如果遇到VAE通道数错误，请：")
    logger.info("   - 确保使用 'TACO-DiT KSampler (Fixed)' 节点")
    logger.info("   - 检查模型是否为DiT类型")
    logger.info("   - 查看控制台日志获取详细信息")
    logger.info("")
    logger.info("4. 性能优化建议：")
    logger.info("   - 使用GPU 3、4、5进行并行推理")
    logger.info("   - 启用CFG并行获得更好性能")
    logger.info("   - 使用Flash Attention减少内存使用")
    logger.info("")
    logger.info("5. 故障排除：")
    logger.info("   - 运行: python test_taco_dit_fixed.py")
    logger.info("   - 查看: TACO_DiT_VAE_Fix_Guide.md")
    logger.info("")

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("TACO-DiT 修复版启动器")
    logger.info("="*60)
    
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，请检查配置")
        return 1
    
    # 设置GPU配置
    setup_gpu_config()
    
    # 显示使用提示
    show_usage_tips()
    
    # 启动ComfyUI
    start_comfyui()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 