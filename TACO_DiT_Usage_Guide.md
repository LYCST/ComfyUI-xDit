# TACO-DiT 使用指南

## 🎯 概述

TACO-DiT (Tensor-parallel Accelerated ComfyUI with Distributed DiT) 是一个为ComfyUI提供多GPU并行推理能力的解决方案。通过集成xDit技术，可以在3、4、5号显卡上实现高效的并行推理。

## 🚀 启动方式

### 方式1：使用专用启动脚本（推荐）
```bash
./start_comfyui_gpu345.sh
```

### 方式2：直接使用Python命令
```bash
CUDA_VISIBLE_DEVICES=3,4,5 python main.py --listen 0.0.0.0 --port 12215
```

## 📋 节点使用方法

### 1. TACO-DiT Model Loader（模型加载器）

**功能**: 将普通模型转换为支持多GPU并行的TACO-DiT模型

**使用方法**:
1. 在ComfyUI中添加 "TACO-DiT Model Loader" 节点
2. 将您的模型连接到 "model" 输入
3. 配置参数：
   - `enable_taco_dit`: 启用TACO-DiT（默认：True）
   - `auto_detect`: 自动检测配置（默认：True）
   - `sequence_parallel_degree`: 序列并行度（默认：3，对应3张GPU）
   - `cfg_parallel`: 启用CFG并行（默认：True）
   - `use_flash_attention`: 启用Flash Attention（默认：True）

**输出**:
- `MODEL`: 支持多GPU并行的模型
- `TACO_DIT_INFO`: 模型信息

### 2. TACO-DiT KSampler（采样器）

**功能**: 使用多GPU并行进行图像生成采样

**使用方法**:
1. 在ComfyUI中添加 "TACO-DiT KSampler" 节点
2. 连接TACO-DiT模型到 "model" 输入
3. 连接其他必要输入（positive, negative, latent_image等）
4. 配置采样参数（steps, cfg, sampler_name等）
5. 设置 `enable_taco_dit` 为 True

**输出**:
- `LATENT`: 生成的图像latent
- `TACO_DIT_STATS`: 执行统计信息

### 3. TACO-DiT Info Display（信息显示）

**功能**: 显示TACO-DiT模型的信息

**使用方法**:
1. 连接TACO-DiT Model Loader的 "TACO_DIT_INFO" 输出
2. 查看控制台输出的详细信息

### 4. TACO-DiT Stats Display（统计显示）

**功能**: 显示TACO-DiT执行的统计信息

**使用方法**:
1. 连接TACO-DiT KSampler的 "TACO_DIT_STATS" 输出
2. 查看控制台输出的执行统计

## 🔧 工作流示例

### 基础工作流

```
CheckpointLoaderSimple → TACO-DiT Model Loader → TACO-DiT Info Display
                                    ↓
CLIPTextEncode (positive) → TACO-DiT KSampler → TACO-DiT Stats Display
                                    ↓
CLIPTextEncode (negative) → VAE Decode → Save Image
                                    ↓
Empty Latent Image
```

### 详细步骤

1. **加载模型**:
   - 添加 "CheckpointLoaderSimple" 节点
   - 选择您的DiT模型（如Flux.1、PixArt-Sigma等）

2. **转换为TACO-DiT模型**:
   - 添加 "TACO-DiT Model Loader" 节点
   - 连接模型输出
   - 配置并行参数（建议使用自动检测）

3. **显示模型信息**:
   - 添加 "TACO-DiT Info Display" 节点
   - 连接模型信息输出
   - 查看控制台确认多GPU配置

4. **文本编码**:
   - 添加 "CLIPTextEncode" 节点（正面提示词）
   - 添加 "CLIPTextEncode" 节点（负面提示词）

5. **创建空白图像**:
   - 添加 "Empty Latent Image" 节点
   - 设置图像尺寸

6. **多GPU采样**:
   - 添加 "TACO-DiT KSampler" 节点
   - 连接所有必要输入
   - 配置采样参数

7. **显示统计信息**:
   - 添加 "TACO-DiT Stats Display" 节点
   - 连接统计信息输出

8. **解码和保存**:
   - 添加 "VAE Decode" 节点
   - 添加 "Save Image" 节点

## 📊 性能监控

### 查看GPU使用情况

在终端中运行：
```bash
watch -n 1 nvidia-smi
```

### 查看TACO-DiT日志

在ComfyUI控制台中查看：
- 模型加载信息
- 并行配置信息
- 执行时间统计
- 错误信息

### 预期性能提升

- **3张GPU并行**: 理论加速比约2.5-3倍
- **内存使用**: 每张GPU约8-12GB
- **推理速度**: 相比单GPU提升2-3倍

## ⚠️ 注意事项

### 1. 硬件要求
- 至少3张GPU（推荐RTX 4090或更高）
- 每张GPU至少8GB显存
- 高速GPU间连接

### 2. 软件要求
- CUDA 11.8+
- PyTorch 2.0+
- xDit 1.0+
- ComfyUI最新版本

### 3. 模型兼容性
- 支持所有xDit兼容的DiT模型
- Flux.1, PixArt-Sigma, SD3等
- 不支持传统Stable Diffusion模型

### 4. 常见问题

**问题**: 节点显示"TACO-DiT not available"
**解决**: 检查xDit是否正确安装，运行 `pip list | grep xfuser`

**问题**: GPU内存不足
**解决**: 减少batch_size或降低图像分辨率

**问题**: 性能提升不明显
**解决**: 检查GPU使用率，确保所有3张GPU都在工作

## 🔍 故障排除

### 1. 检查TACO-DiT状态
```bash
python -c "
import sys
sys.path.insert(0, '.')
from comfy.taco_dit import TACODiTConfigManager
config_manager = TACODiTConfigManager()
print(f'TACO-DiT config: {config_manager.config}')
"
```

### 2. 检查GPU可用性
```bash
python -c "
import torch
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}')
"
```

### 3. 检查xDit安装
```bash
python -c "
try:
    import xfuser
    print('xDit available')
except ImportError as e:
    print(f'xDit not available: {e}')
"
```

## 📈 优化建议

### 1. 并行配置优化
- 对于3张GPU：使用序列并行度3
- 对于4张GPU：使用序列并行度4
- 对于6张GPU：使用序列并行度4 + 流水线并行度2

### 2. 内存优化
- 启用Flash Attention
- 使用模型缓存
- 适当调整batch_size

### 3. 性能监控
- 定期检查GPU使用率
- 监控内存使用情况
- 记录执行时间统计

## 🎉 成功标志

当TACO-DiT正常工作时，您应该看到：

1. **控制台输出**:
   ```
   TACO-DiT model loaded successfully:
     - Parallel enabled: True
     - Model type: flux
     - xDit wrapper: True
   ```

2. **GPU使用率**: 所有3张GPU都有活动

3. **执行时间**: 相比单GPU有明显提升

4. **统计信息**: 显示并行执行次数和平均时间

## 📞 技术支持

如果遇到问题，请检查：
1. 环境配置是否正确
2. 模型是否兼容
3. GPU资源是否充足
4. 日志中的错误信息

TACO-DiT将为您提供强大的多GPU并行推理能力，显著提升图像生成的速度和效率！ 