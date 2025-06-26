#!/usr/bin/env python3
"""
测试模型路径修复
================

验证safetensors路径到diffusers目录的转换
"""

import os
import sys

def test_path_conversion():
    """测试路径转换逻辑"""
    print("🔧 测试模型路径修复...")
    
    # 模拟的路径
    original_path = "/home/shuzuan/prj/ComfyUI-xDit/models/unet/flux/flux1-dev.safetensors"
    
    print(f"原始路径: {original_path}")
    
    # 路径转换逻辑
    if original_path.endswith('.safetensors'):
        base_dir = os.path.dirname(original_path)
        model_name = os.path.splitext(os.path.basename(original_path))[0]
        corrected_path = os.path.join(base_dir, model_name)
        
        print(f"转换后路径: {corrected_path}")
        
        # 检查路径是否存在
        if os.path.exists(corrected_path):
            print(f"✅ 目录存在: {corrected_path}")
            
            # 检查是否有model_index.json
            model_index_path = os.path.join(corrected_path, "model_index.json")
            if os.path.exists(model_index_path):
                print(f"✅ 发现model_index.json: {model_index_path}")
                return True
            else:
                print(f"❌ 未找到model_index.json: {model_index_path}")
                return False
        else:
            print(f"❌ 目录不存在: {corrected_path}")
            return False
    
    return False

def main():
    print("="*50)
    print("模型路径修复测试")
    print("="*50)
    
    success = test_path_conversion()
    
    print("="*50)
    print(f"测试结果: {'✅ 路径修复成功' if success else '❌ 仍有问题'}")
    print("="*50)
    
    if success:
        print("\n🚀 现在可以重新启动ComfyUI测试!")
        print("或者运行: python main.py")
    else:
        print("\n⚠️ 请检查模型目录结构")
    
    return success

if __name__ == "__main__":
    main() 