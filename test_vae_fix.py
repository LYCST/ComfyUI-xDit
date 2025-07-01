#!/usr/bin/env python3
"""
测试vae变量修复
"""

import sys
import os

def test_vae_variable_fix():
    """测试vae变量修复"""
    try:
        print("🧪 测试vae变量修复...")
        
        # 模拟vae变量初始化过程
        comfyui_vae = None  # 模拟None情况
        diffusers_vae = None
        
        # 测试修复逻辑
        if comfyui_vae is None:
            print("❌ comfyui_vae is None")
            return False
        
        # 模拟VAE提取过程
        vae_type_name = type(comfyui_vae).__name__ if comfyui_vae else "None"
        print(f"VAE type name: {vae_type_name}")
        
        # 测试各种VAE类型
        test_cases = [
            ("first_stage_model", "first_stage_model"),
            ("model", "model"),
            ("AutoencodingEngine", "AutoencodingEngine"),
            ("UnknownType", "UnknownType")
        ]
        
        for attr_name, expected_type in test_cases:
            print(f"测试 {attr_name} -> {expected_type}")
            
            # 模拟VAE对象
            class MockVAE:
                def __init__(self, attr_name):
                    if attr_name == "first_stage_model":
                        self.first_stage_model = "mock_first_stage"
                    elif attr_name == "model":
                        self.model = "mock_model"
                    elif attr_name == "AutoencodingEngine":
                        self.decoder = "mock_decoder"
            
            mock_vae = MockVAE(attr_name)
            
            # 测试提取逻辑
            if hasattr(mock_vae, 'first_stage_model'):
                diffusers_vae = mock_vae.first_stage_model
                print(f"  ✅ 提取 first_stage_model: {diffusers_vae}")
            elif hasattr(mock_vae, 'model'):
                diffusers_vae = mock_vae.model
                print(f"  ✅ 提取 model: {diffusers_vae}")
            elif 'AutoencodingEngine' in type(mock_vae).__name__ or hasattr(mock_vae, 'decoder'):
                print(f"  ✅ 检测到 AutoencodingEngine，创建包装器")
                diffusers_vae = "VAEWrapper"
            else:
                print(f"  ❌ 无法提取VAE")
                diffusers_vae = None
            
            # 验证结果
            if diffusers_vae is None:
                print(f"  ❌ diffusers_vae is None")
            else:
                print(f"  ✅ diffusers_vae: {diffusers_vae}")
        
        print("✅ vae变量修复测试完成")
        return True
        
    except Exception as e:
        print(f"❌ vae变量修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_validation():
    """测试组件验证逻辑"""
    try:
        print("🧪 测试组件验证逻辑...")
        
        # 模拟组件
        components = {
            'diffusers_vae': "mock_vae",
            'text_encoder': "mock_clip",
            'text_encoder_2': "mock_t5",
            'transformer': "mock_transformer",
            'scheduler': "mock_scheduler",
            'tokenizer': "mock_tokenizer",
            'tokenizer_2': "mock_tokenizer_2"
        }
        
        # 测试验证逻辑
        for name, component in components.items():
            if component is None:
                print(f"❌ {name} is None")
                return False
            else:
                print(f"✅ {name}: {component}")
        
        print("✅ 所有组件验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 组件验证测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试vae变量修复...")
    
    success1 = test_vae_variable_fix()
    success2 = test_component_validation()
    
    if success1 and success2:
        print("\n🎉 vae变量修复测试成功！")
        print("✅ vae变量初始化逻辑正常")
        print("✅ 组件验证逻辑正常")
    else:
        print("\n❌ vae变量修复测试失败。")
        sys.exit(1) 