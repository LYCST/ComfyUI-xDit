#!/usr/bin/env python3
"""
测试FLUX模型检测修复
"""

import sys
import os
import safetensors.torch

def test_flux_model_detection():
    """测试FLUX模型检测"""
    try:
        # 测试模型路径
        model_path = "/home/shuzuan/prj/ComfyUI-xDit/models/unet/flux/flux1-dev.safetensors"
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        print(f"✅ 模型文件存在: {model_path}")
        
        # 加载模型
        sd = safetensors.torch.load_file(model_path)
        keys = list(sd.keys())
        
        print(f"📊 模型信息:")
        print(f"  - 总键数: {len(keys)}")
        print(f"  - 前10个键: {keys[:10]}")
        
        # 检测FLUX指标
        flux_indicators = [
            'transformer_blocks',
            'transformer',
            'model.diffusion_model',
            'diffusion_model',
            'time_embed',
            'input_blocks',
            'middle_block',
            'output_blocks',
            'double_blocks',  # FLUX模型的实际键名
            'img_attn',
            'img_mlp'
        ]
        
        found_indicators = []
        for key in keys:
            for indicator in flux_indicators:
                if key.startswith(indicator):
                    found_indicators.append(indicator)
                    break
        
        print(f"🔍 FLUX指标检测:")
        print(f"  - 找到的指标: {list(set(found_indicators))}")
        
        # 调试：检查前几个键是否匹配
        print(f"🔍 调试检查:")
        for i, key in enumerate(keys[:5]):
            for indicator in flux_indicators:
                if key.startswith(indicator):
                    print(f"  - 键 {i}: '{key}' 匹配指标 '{indicator}'")
                    break
            else:
                print(f"  - 键 {i}: '{key}' 未匹配任何指标")
        
        is_flux_model = len(found_indicators) > 0
        
        if is_flux_model:
            print("✅ 成功检测到FLUX模型格式")
            return True
        else:
            print("❌ 未检测到FLUX模型格式")
            print(f"  - 可用的键模式: {[k.split('.')[0] for k in keys[:20]]}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 开始测试FLUX模型检测修复...")
    
    success = test_flux_model_detection()
    
    if success:
        print("\n🎉 FLUX模型检测修复成功！")
    else:
        print("\n❌ FLUX模型检测修复失败。")
        sys.exit(1) 