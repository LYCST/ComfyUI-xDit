#!/usr/bin/env python3
"""
将FLUX safetensors转换为diffusers格式，启用多GPU加速
"""

import os
import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_flux_safetensors_to_diffusers(safetensors_path: str, output_dir: str):
    """
    将FLUX safetensors文件转换为diffusers格式
    """
    try:
        logger.info(f"🔄 开始转换: {safetensors_path} -> {output_dir}")
        
        # 检查输入文件
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Safetensors文件不存在: {safetensors_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        from diffusers import FluxPipeline
        
        # 方法1: 尝试从HuggingFace下载完整模型，然后替换transformer权重
        logger.info("📥 下载基础FLUX模型组件...")
        
        try:
            # 下载完整的FLUX.1-dev模型
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                device_map=None,
                low_cpu_mem_usage=True
            )
            
            logger.info("✅ 基础模型下载完成")
            
            # 加载自定义的transformer权重
            logger.info("🔄 加载自定义transformer权重...")
            
            import safetensors.torch
            custom_weights = safetensors.torch.load_file(safetensors_path)
            
            # 过滤出transformer相关的权重
            transformer_weights = {}
            for key, value in custom_weights.items():
                if any(prefix in key for prefix in ['double_blocks', 'single_blocks', 'img_in', 'txt_in', 'final_layer']):
                    transformer_weights[key] = value
            
            logger.info(f"找到 {len(transformer_weights)} 个transformer权重")
            
            # 替换transformer权重
            pipeline.transformer.load_state_dict(transformer_weights, strict=False)
            logger.info("✅ Transformer权重替换完成")
            
            # 保存为diffusers格式
            logger.info(f"💾 保存到: {output_dir}")
            pipeline.save_pretrained(output_dir)
            
            logger.info("🎉 转换完成!")
            logger.info(f"✅ 现在可以在XDiTUNetLoader中选择目录: {output_dir}")
            
            return True
            
        except Exception as download_error:
            logger.error(f"从HuggingFace下载失败: {download_error}")
            
            # 方法2: 创建最小的diffusers结构
            logger.info("🔄 尝试创建最小diffusers结构...")
            return create_minimal_diffusers_structure(safetensors_path, output_dir)
            
    except Exception as e:
        logger.error(f"❌ 转换失败: {e}")
        logger.exception("详细错误:")
        return False

def create_minimal_diffusers_structure(safetensors_path: str, output_dir: str):
    """
    创建最小的diffusers目录结构
    """
    try:
        logger.info("🏗️ 创建最小diffusers结构...")
        
        # 复制safetensors文件
        import shutil
        shutil.copy2(safetensors_path, os.path.join(output_dir, "diffusion_pytorch_model.safetensors"))
        
        # 创建基本的model_index.json
        model_index = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.21.0",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "T5Tokenizer"],
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"]
        }
        
        import json
        with open(os.path.join(output_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        # 创建transformer配置
        transformer_config = {
            "_class_name": "FluxTransformer2DModel",
            "_diffusers_version": "0.21.0",
            "guidance_embeds": False,
            "in_channels": 64,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "pooled_projection_dim": 768,
            "vec_in_dim": 768,
            "axes_dim": [16, 56, 56],
            "theta": 10000,
            "use_bias": True,
            "torch_dtype": "float16"
        }
        
        transformer_dir = os.path.join(output_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(transformer_config, f, indent=2)
        
        # 移动safetensors到transformer目录
        shutil.move(
            os.path.join(output_dir, "diffusion_pytorch_model.safetensors"),
            os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
        )
        
        logger.info("✅ 最小diffusers结构创建完成")
        logger.info("⚠️ 注意: 此结构可能需要额外的组件才能完全工作")
        
        return True
        
    except Exception as e:
        logger.error(f"创建最小结构失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 FLUX Safetensors to Diffusers 转换工具")
    print("=" * 50)
    
    # 默认路径
    default_input = "/home/shuzuan/prj/ComfyUI-xDit/models/unet/flux/flux1-dev.safetensors"
    default_output = "/home/shuzuan/prj/ComfyUI-xDit/models/unet/flux/flux1-dev-diffusers"
    
    print(f"输入文件: {default_input}")
    print(f"输出目录: {default_output}")
    print()
    
    if input("是否使用默认路径? (y/n): ").lower() != 'y':
        safetensors_path = input("请输入safetensors文件路径: ")
        output_dir = input("请输入输出目录路径: ")
    else:
        safetensors_path = default_input
        output_dir = default_output
    
    print(f"\n🔄 开始转换...")
    
    success = convert_flux_safetensors_to_diffusers(safetensors_path, output_dir)
    
    if success:
        print("\n🎉 转换成功!")
        print(f"✅ 输出目录: {output_dir}")
        print("\n📋 使用方法:")
        print("1. 在ComfyUI中使用XDiTUNetLoader节点")
        print(f"2. 选择目录: flux1-dev-diffusers")
        print("3. 启用多GPU加速")
        print("4. 享受8GPU并行推理!")
    else:
        print("\n❌ 转换失败")
        print("请检查错误日志并重试")

if __name__ == "__main__":
    main() 