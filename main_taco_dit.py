#!/usr/bin/env python3
"""
ComfyUI with TACO-DiT Integration

集成TACO-DiT多GPU并行推理的ComfyUI启动脚本
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_taco_dit_availability():
    """检查TACO-DiT是否可用"""
    try:
        # 检查xDit
        import xfuser
        logger.info("✅ xDit available")
        
        # 检查TACO-DiT模块
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from comfy.taco_dit import TACODiTConfigManager
        
        config_manager = TACODiTConfigManager()
        config = config_manager.config
        
        logger.info("✅ TACO-DiT modules available")
        logger.info(f"   - Auto-detected config: {config}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ TACO-DiT not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TACO-DiT check failed: {e}")
        return False

def setup_environment():
    """设置环境变量"""
    # 设置CUDA相关环境变量 - 只使用3、4、5号显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
    
    # 设置xDit相关环境变量
    os.environ['XFUSER_ENABLE_FLASH_ATTENTION'] = '1'
    os.environ['XFUSER_ENABLE_CACHE'] = '1'
    
    # 设置ComfyUI相关环境变量
    os.environ['COMFYUI_ENABLE_TACO_DIT'] = '1'
    
    logger.info("Environment variables set for TACO-DiT (GPUs 3,4,5)")

def create_custom_nodes_dir():
    """创建自定义节点目录"""
    custom_nodes_dir = Path("custom_nodes")
    custom_nodes_dir.mkdir(exist_ok=True)
    
    # 检查TACO-DiT节点文件是否存在
    taco_dit_nodes_file = custom_nodes_dir / "TACO_DiT_Nodes.py"
    if not taco_dit_nodes_file.exists():
        logger.warning("TACO-DiT nodes file not found, creating basic version")
        
        # 创建基础的TACO-DiT节点文件
        basic_nodes_content = '''
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
'''
        
        with open(taco_dit_nodes_file, 'w', encoding='utf-8') as f:
            f.write(basic_nodes_content)
        
        logger.info("Created basic TACO-DiT nodes file")
    
    return custom_nodes_dir

def start_comfyui(args):
    """启动ComfyUI"""
    logger.info("🚀 Starting ComfyUI with TACO-DiT integration...")
    
    # 检查TACO-DiT可用性
    if not check_taco_dit_availability():
        logger.warning("TACO-DiT not available, starting ComfyUI without TACO-DiT")
    
    # 设置环境
    setup_environment()
    
    # 创建自定义节点目录
    create_custom_nodes_dir()
    
    # 构建ComfyUI启动命令
    cmd = [
        sys.executable, "main.py",
        "--listen", args.listen,
        "--port", str(args.port)
    ]
    
    # 添加其他参数
    if args.extra_args:
        cmd.extend(args.extra_args)
    
    logger.info(f"Starting ComfyUI with command: {' '.join(cmd)}")
    
    try:
        # 启动ComfyUI
        process = subprocess.Popen(cmd)
        
        logger.info(f"✅ ComfyUI started successfully!")
        logger.info(f"   - URL: http://{args.listen}:{args.port}")
        logger.info(f"   - TACO-DiT: {'Enabled' if check_taco_dit_availability() else 'Disabled'}")
        logger.info(f"   - Process ID: {process.pid}")
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        if process:
            process.terminate()
            process.wait()
    except Exception as e:
        logger.error(f"Failed to start ComfyUI: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ComfyUI with TACO-DiT Integration")
    parser.add_argument("--listen", default="0.0.0.0", help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=12215, help="Port number (default: 12215)")
    parser.add_argument("--extra-args", nargs="*", help="Extra arguments to pass to ComfyUI")
    
    args = parser.parse_args()
    
    # 显示启动信息
    logger.info("="*60)
    logger.info("🎯 ComfyUI with TACO-DiT Integration")
    logger.info("="*60)
    logger.info(f"Listen: {args.listen}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Extra args: {args.extra_args}")
    
    # 检查系统信息
    import torch
    gpu_count = torch.cuda.device_count()
    logger.info(f"GPU count: {gpu_count}")
    
    if gpu_count > 0:
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU {i}: {props.name}, {memory_gb:.1f} GB")
    
    # 启动ComfyUI
    success = start_comfyui(args)
    
    if success:
        logger.info("✅ ComfyUI with TACO-DiT completed successfully")
    else:
        logger.error("❌ ComfyUI with TACO-DiT failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 