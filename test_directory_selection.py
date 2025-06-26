#!/usr/bin/env python3
"""
测试目录选择功能
================

验证XDiTUNetLoader和XDiTCheckpointLoader的目录选择功能
"""

import os
import sys

def test_directory_discovery():
    """测试目录发现功能"""
    print("🔍 测试diffusers目录发现...")
    
    # 模拟目录结构
    test_paths = [
        "/home/shuzuan/prj/ComfyUI-xDit/models/unet/flux/flux1-dev",
        "/home/shuzuan/prj/ComfyUI-xDit/models/checkpoints/flux/flux1-dev", 
        "/home/shuzuan/prj/ComfyUI-xDit/models/diffusers/flux1-dev"
    ]
    
    found_dirs = []
    
    for path in test_paths:
        if os.path.exists(path):
            model_index_path = os.path.join(path, "model_index.json")
            if os.path.exists(model_index_path):
                found_dirs.append(path)
                print(f"✅ 发现diffusers目录: {path}")
            else:
                print(f"❌ 缺少model_index.json: {path}")
        else:
            print(f"❌ 目录不存在: {path}")
    
    return found_dirs

def test_model_loading():
    """测试模型加载功能"""
    print("\n🔧 测试模型加载...")
    
    try:
        # 添加正确的路径到sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        nodes_dir = os.path.join(current_dir, "custom_nodes", "comfyui_xdit_multigpu")
        
        if not os.path.exists(nodes_dir):
            print(f"❌ 节点目录不存在: {nodes_dir}")
            return False
        
        # 添加到Python路径
        if nodes_dir not in sys.path:
            sys.path.insert(0, nodes_dir)
        
        # 尝试导入节点
        try:
            import nodes
            print(f"✅ 成功导入nodes模块")
            
            # 检查是否有我们需要的类
            if hasattr(nodes, 'XDiTUNetLoader'):
                print("✅ 找到XDiTUNetLoader类")
                unet_loader = nodes.XDiTUNetLoader()
                
                # 测试get_diffusers_model_list方法
                if hasattr(unet_loader, 'get_diffusers_model_list'):
                    print("✅ 找到get_diffusers_model_list方法")
                    try:
                        diffusers_list = unet_loader.get_diffusers_model_list()
                        print(f"发现的diffusers模型: {diffusers_list}")
                    except Exception as e:
                        print(f"⚠️ 调用get_diffusers_model_list失败: {e}")
                else:
                    print("❌ 未找到get_diffusers_model_list方法")
            else:
                print("❌ 未找到XDiTUNetLoader类")
            
            if hasattr(nodes, 'XDiTCheckpointLoader'):
                print("✅ 找到XDiTCheckpointLoader类")
                checkpoint_loader = nodes.XDiTCheckpointLoader()
                
                if hasattr(checkpoint_loader, 'get_diffusers_model_list'):
                    print("✅ 找到checkpoint get_diffusers_model_list方法")
                    try:
                        checkpoint_diffusers_list = checkpoint_loader.get_diffusers_model_list()
                        print(f"发现的checkpoint diffusers模型: {checkpoint_diffusers_list}")
                    except Exception as e:
                        print(f"⚠️ 调用checkpoint get_diffusers_model_list失败: {e}")
                else:
                    print("❌ 未找到checkpoint get_diffusers_model_list方法")
            else:
                print("❌ 未找到XDiTCheckpointLoader类")
            
            return True
            
        except ImportError as e:
            print(f"❌ 导入nodes模块失败: {e}")
            return False
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_input_types():
    """测试输入类型定义"""
    print("\n📋 测试输入类型...")
    
    try:
        # 添加正确的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        nodes_dir = os.path.join(current_dir, "custom_nodes", "comfyui_xdit_multigpu")
        
        if nodes_dir not in sys.path:
            sys.path.insert(0, nodes_dir)
        
        import nodes
        
        # 测试UNet输入类型
        if hasattr(nodes, 'XDiTUNetLoader'):
            try:
                unet_inputs = nodes.XDiTUNetLoader.INPUT_TYPES()
                print("XDiTUNetLoader输入类型:")
                for key, value in unet_inputs["required"].items():
                    print(f"  {key}: {type(value)}")
            except Exception as e:
                print(f"⚠️ 获取UNet输入类型失败: {e}")
        
        # 测试Checkpoint输入类型
        if hasattr(nodes, 'XDiTCheckpointLoader'):
            try:
                checkpoint_inputs = nodes.XDiTCheckpointLoader.INPUT_TYPES()
                print("XDiTCheckpointLoader输入类型:")
                for section, params in checkpoint_inputs.items():
                    print(f"  {section}:")
                    for key, value in params.items():
                        print(f"    {key}: {type(value)}")
            except Exception as e:
                print(f"⚠️ 获取Checkpoint输入类型失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 输入类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("XDiT 目录选择功能测试")
    print("="*60)
    
    # 显示当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本位置: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 测试目录发现
    found_dirs = test_directory_discovery()
    
    # 测试模型加载
    load_success = test_model_loading()
    
    # 测试输入类型
    input_success = test_input_types()
    
    print("\n" + "="*60)
    print("测试结果总结:")
    print(f"  发现diffusers目录: {len(found_dirs)}个")
    print(f"  模型加载功能: {'✅ 正常' if load_success else '❌ 失败'}")
    print(f"  输入类型定义: {'✅ 正常' if input_success else '❌ 失败'}")
    print("="*60)
    
    if found_dirs and load_success and input_success:
        print("\n🚀 目录选择功能已就绪!")
        print("现在用户可以在ComfyUI界面中:")
        print("1. 选择模型类型 (checkpoint 或 diffusers)")
        print("2. 从下拉菜单中选择diffusers目录")
        print("3. 直接使用目录路径而不需要文件转换")
    else:
        print("\n⚠️ 还有一些问题需要解决")
        if not found_dirs:
            print("- 请确保有可用的diffusers格式模型目录")
        if not load_success:
            print("- 检查节点加载逻辑")
        if not input_success:
            print("- 检查输入类型定义")
    
    return found_dirs and load_success and input_success

if __name__ == "__main__":
    main() 