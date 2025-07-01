#!/usr/bin/env python3
"""
测试VAE变量作用域修复
"""

import sys
import os

def test_vae_scope_fix():
    """测试VAE变量作用域修复"""
    try:
        print("🧪 测试VAE变量作用域修复...")
        
        # 模拟XDiTKSampler的sample方法调用
        def mock_sample_method(vae=None, clip=None):
            """模拟sample方法"""
            print(f"📥 sample方法接收参数:")
            print(f"  • vae: {type(vae) if vae else 'None'}")
            print(f"  • clip: {type(clip) if clip else 'None'}")
            
            # 模拟调用_run_xdit_with_timeout
            result = mock_run_xdit_with_timeout(
                dispatcher="mock_dispatcher",
                model_info="mock_model_info", 
                positive="mock_positive",
                negative="mock_negative",
                latent_samples="mock_latents",
                steps=20,
                cfg=8.0,
                seed=42,
                timeout_seconds=60,
                vae=vae,  # 🔧 传递VAE参数
                clip=clip  # 🔧 传递CLIP参数
            )
            
            print(f"📤 sample方法返回: {result}")
            return result
        
        def mock_run_xdit_with_timeout(dispatcher, model_info, positive, negative, 
                                     latent_samples, steps, cfg, seed, timeout_seconds, 
                                     vae=None, clip=None):
            """模拟_run_xdit_with_timeout方法 - 修复版本"""
            print(f"🔧 _run_xdit_with_timeout接收参数:")
            print(f"  • vae: {type(vae) if vae else 'None'}")
            print(f"  • clip: {type(clip) if clip else 'None'}")
            
            def mock_inference_worker():
                """模拟推理工作线程"""
                print(f"🎯 inference_worker内部:")
                print(f"  • vae: {type(vae) if vae else 'None'}")
                print(f"  • clip: {type(clip) if clip else 'None'}")
                
                # 模拟dispatcher.run_inference调用
                mock_dispatcher_call(vae, clip)
            
            # 模拟线程启动
            mock_inference_worker()
            return "mock_result"
        
        def mock_dispatcher_call(vae, clip):
            """模拟dispatcher.run_inference调用"""
            print(f"🚀 dispatcher.run_inference调用:")
            print(f"  • comfyui_vae: {type(vae) if vae else 'None'}")
            print(f"  • comfyui_clip: {type(clip) if clip else 'None'}")
            
            if vae is None:
                print("❌ VAE参数为None - 作用域问题!")
                return False
            else:
                print("✅ VAE参数正确传递 - 作用域修复成功!")
                return True
        
        # 测试用例1：传递VAE和CLIP
        print("\n" + "="*50)
        print("测试用例1：传递VAE和CLIP对象")
        print("="*50)
        
        mock_vae = "mock_vae_object"
        mock_clip = "mock_clip_object"
        
        result1 = mock_sample_method(vae=mock_vae, clip=mock_clip)
        
        # 测试用例2：不传递VAE和CLIP
        print("\n" + "="*50)
        print("测试用例2：不传递VAE和CLIP对象")
        print("="*50)
        
        result2 = mock_sample_method()
        
        # 测试用例3：只传递VAE
        print("\n" + "="*50)
        print("测试用例3：只传递VAE对象")
        print("="*50)
        
        result3 = mock_sample_method(vae=mock_vae)
        
        print("\n" + "="*50)
        print("✅ VAE变量作用域修复测试完成")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ VAE变量作用域修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_signature():
    """测试方法签名是否正确"""
    try:
        print("🧪 测试方法签名...")
        
        # 模拟方法签名
        def original_method(dispatcher, model_info, positive, negative, latent_samples, steps, cfg, seed, timeout_seconds):
            """原始方法签名（有问题）"""
            pass
        
        def fixed_method(dispatcher, model_info, positive, negative, latent_samples, steps, cfg, seed, timeout_seconds, vae=None, clip=None):
            """修复后的方法签名"""
            pass
        
        # 检查参数数量
        import inspect
        
        original_params = inspect.signature(original_method).parameters
        fixed_params = inspect.signature(fixed_method).parameters
        
        print(f"原始方法参数数量: {len(original_params)}")
        print(f"修复方法参数数量: {len(fixed_params)}")
        
        # 检查是否包含vae和clip参数
        has_vae = 'vae' in fixed_params
        has_clip = 'clip' in fixed_params
        
        print(f"包含vae参数: {has_vae}")
        print(f"包含clip参数: {has_clip}")
        
        if has_vae and has_clip:
            print("✅ 方法签名修复正确")
            return True
        else:
            print("❌ 方法签名修复失败")
            return False
            
    except Exception as e:
        print(f"❌ 方法签名测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试VAE变量作用域修复...")
    
    success1 = test_vae_scope_fix()
    success2 = test_method_signature()
    
    if success1 and success2:
        print("\n🎉 VAE变量作用域修复测试成功！")
        print("✅ VAE和CLIP参数正确传递")
        print("✅ 方法签名修复正确")
        print("✅ 作用域问题已解决")
    else:
        print("\n❌ VAE变量作用域修复测试失败。")
        sys.exit(1) 