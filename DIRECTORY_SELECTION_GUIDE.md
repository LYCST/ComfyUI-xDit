# xDiT 目录选择功能使用指南

## 概述

现在xDiT多GPU节点支持直接选择diffusers格式的模型目录，不再需要手动转换safetensors文件。这使得用户体验更加直观和便捷。

## 主要改进

### 1. XDiTCheckpointLoader 增强

- **新增模型类型选择**: 可以选择 `checkpoint` 或 `diffusers` 两种模式
- **智能目录检测**: 自动扫描并列出所有包含 `model_index.json` 的diffusers目录
- **多路径支持**: 支持从 `diffusers/`、`checkpoints/` 等多个目录加载

#### 使用方法：
1. 设置 `model_type` 为 `diffusers`
2. 从 `diffusers_model_path` 下拉菜单中选择目录
3. 配置多GPU参数（可选）

### 2. XDiTUNetLoader 重构

- **专注diffusers格式**: 直接选择diffusers格式的模型目录
- **简化界面**: 去除了文件选择，改为目录选择
- **更好的兼容性**: 与xDiT框架完美配合

#### 使用方法：
1. 从 `diffusers_model_path` 下拉菜单中选择目录
2. 启用多GPU加速（可选）
3. 配置并行策略和调度策略

## 支持的目录结构

系统会自动检测以下位置的diffusers格式模型：

```
models/
├── diffusers/
│   └── flux1-dev/          # ✅ 自动检测
│       ├── model_index.json
│       ├── unet/
│       ├── text_encoder/
│       └── ...
├── unet/
│   └── flux/
│       └── flux1-dev/      # ✅ 自动检测
│           ├── model_index.json
│           └── ...
└── checkpoints/
    └── flux/
        └── flux1-dev/      # ✅ 自动检测
            ├── model_index.json
            └── ...
```

## 当前测试状态

✅ **目录发现**: 成功检测到 `/models/unet/flux/flux1-dev`  
✅ **模型加载**: XDiTUNetLoader 和 XDiTCheckpointLoader 正常工作  
✅ **输入类型**: 所有参数类型定义正确  

## 下一步

1. **启动ComfyUI**: `python main.py`
2. **使用新节点**: 在节点菜单中查找 "XDiT" 分类
3. **选择模型**: 直接从下拉菜单选择diffusers目录
4. **享受多GPU加速**: 配置GPU设备并启用加速

## 故障排除

### 如果没有看到模型目录：
1. 确保模型目录包含 `model_index.json` 文件
2. 检查目录是否在支持的路径下
3. 重启ComfyUI刷新目录列表

### 如果多GPU初始化失败：
1. 系统会自动回退到标准ComfyUI采样
2. 检查GPU设备ID是否正确
3. 确保有足够的GPU内存

---

🎉 **现在用户可以直接选择目录，无需手动转换文件格式！** 