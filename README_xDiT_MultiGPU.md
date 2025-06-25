# ComfyUI xDiT Multi-GPU 加速插件

## 概述

这是一个为ComfyUI提供多GPU加速的插件，基于xDiT框架实现。插件提供了与标准ComfyUI节点完全兼容的替代节点，支持无感切换到多GPU加速模式。

## 特性

- ✅ **完全兼容**: 与标准ComfyUI节点输入输出完全一致
- ✅ **无感切换**: 只需替换节点即可启用多GPU加速
- ✅ **自动回退**: 当xDiT不可用时自动回退到单GPU模式
- ✅ **多种模型支持**: 支持Checkpoint、UNet、CLIP、VAE等所有模型类型
- ✅ **Flux模型优化**: 专门针对Flux模型的多组件加载优化
- ✅ **多种并行策略**: 支持PipeFusion、USP、Hybrid、Tensor、CFG等策略
- ✅ **智能调度**: 支持round_robin、least_loaded、weighted_round_robin、adaptive等调度策略

## 安装

1. 确保已安装ComfyUI
2. 确保已安装xDiT框架
3. 将插件复制到ComfyUI的`custom_nodes`目录
4. 重启ComfyUI

## 节点说明

### 1. Load Checkpoint (xDiT Multi-GPU)
**替代**: `CheckpointLoaderSimple`

**输入**:
- `ckpt_name`: 模型文件名
- `enable_multi_gpu`: 是否启用多GPU加速 (默认: True)
- `gpu_devices`: GPU设备列表 (默认: "0,1,2,3")
- `parallel_strategy`: 并行策略 (默认: "Hybrid")
- `scheduling_strategy`: 调度策略 (默认: "round_robin")

**输出**:
- `MODEL`: 模型
- `CLIP`: CLIP模型
- `VAE`: VAE模型
- `XDIT_DISPATCHER`: xDiT调度器

### 2. Load UNet (xDiT Multi-GPU)
**替代**: `UNetLoader`

**输入**:
- `unet_name`: UNet模型文件名
- `enable_multi_gpu`: 是否启用多GPU加速 (默认: True)
- `gpu_devices`: GPU设备列表 (默认: "0,1,2,3")
- `parallel_strategy`: 并行策略 (默认: "Hybrid")
- `scheduling_strategy`: 调度策略 (默认: "round_robin")

**输出**:
- `MODEL`: UNet模型
- `XDIT_DISPATCHER`: xDiT调度器

### 3. Load VAE (xDiT)
**替代**: `VAELoader`

**输入**:
- `vae_name`: VAE模型文件名

**输出**:
- `VAE`: VAE模型

### 4. Load CLIP (xDiT)
**替代**: `CLIPLoader`

**输入**:
- `clip_name`: CLIP模型文件名
- `type`: 模型类型 (默认: "stable_diffusion")
- `device`: 设备 (可选, 默认: "default")

**输出**:
- `CLIP`: CLIP模型

### 5. Load Dual CLIP (xDiT)
**替代**: `DualCLIPLoader`

**输入**:
- `clip_name1`: 第一个CLIP模型文件名
- `clip_name2`: 第二个CLIP模型文件名
- `type`: 模型类型 (默认: "sdxl")
- `device`: 设备 (可选, 默认: "default")

**输出**:
- `CLIP`: 组合的CLIP模型

### 6. KSampler (xDiT Multi-GPU)
**替代**: `KSampler`

**输入**:
- `model`: 模型
- `seed`: 随机种子
- `steps`: 步数
- `cfg`: CFG值
- `sampler_name`: 采样器名称
- `scheduler`: 调度器
- `positive`: 正面提示词
- `negative`: 负面提示词
- `latent_image`: 潜在图像
- `denoise`: 去噪强度
- `xdit_dispatcher`: xDiT调度器 (可选)

**输出**:
- `LATENT`: 生成的潜在图像

## Flux模型使用示例

### 工作流程

```
[Load UNet (xDiT Multi-GPU)] 
├── MODEL → [KSampler (xDiT Multi-GPU)]
└── XDIT_DISPATCHER → [KSampler (xDiT Multi-GPU)]

[Load Dual CLIP (xDiT)] 
└── CLIP → [CLIP Text Encode]

[Load VAE (xDiT)]
└── VAE → [VAE Decode]

[CLIP Text Encode]
└── CONDITIONING → [KSampler (xDiT Multi-GPU)]

[KSampler (xDiT Multi-GPU)]
└── LATENT → [VAE Decode]
```

### 步骤说明

1. **加载UNet模型**:
   - 使用 "Load UNet (xDiT Multi-GPU)" 节点
   - 选择Flux UNet模型文件
   - 启用多GPU加速
   - 设置GPU设备为 "0,1,2,3"
   - 选择 "Hybrid" 并行策略

2. **加载双CLIP模型**:
   - 使用 "Load Dual CLIP (xDiT)" 节点
   - clip_name1: 选择CLIP-L模型
   - clip_name2: 选择T5-XXL模型
   - type: 设置为 "flux"

3. **加载VAE模型**:
   - 使用 "Load VAE (xDiT)" 节点
   - 选择Flux VAE模型文件

4. **文本编码**:
   - 使用标准 "CLIP Text Encode" 节点
   - 连接双CLIP输出
   - 输入提示词

5. **采样生成**:
   - 使用 "KSampler (xDiT Multi-GPU)" 节点
   - 连接UNet MODEL输出
   - 连接UNet XDIT_DISPATCHER输出 (可选，用于多GPU加速)
   - 连接CLIP conditioning
   - 设置采样参数

6. **解码输出**:
   - 使用标准 "VAE Decode" 节点
   - 连接VAE和采样器输出

## 并行策略说明

### Hybrid (推荐)
- 结合多种并行技术
- 适合大多数模型和场景
- 平衡性能和兼容性

### PipeFusion
- 管道融合并行
- 适合大模型
- 内存效率高

### USP
- 统一序列并行
- 适合长序列处理
- 计算效率高

### Tensor
- 张量并行
- 适合大模型
- 通信开销较大

### CFG
- 分类器自由引导并行
- 适合CFG采样
- 专门优化

## 调度策略说明

### round_robin (推荐)
- 轮询调度
- 负载均衡
- 适合大多数场景

### least_loaded
- 最少负载调度
- 动态负载均衡
- 适合异构GPU

### weighted_round_robin
- 加权轮询调度
- 考虑GPU性能差异
- 适合异构环境

### adaptive
- 自适应调度
- 根据负载动态调整
- 最智能但开销较大

## 故障排除

### 常见问题

1. **xDiT不可用**
   - 插件会自动回退到单GPU模式
   - 检查xDiT安装是否正确

2. **多GPU初始化失败**
   - 插件会自动回退到单GPU模式
   - 检查GPU设备配置

3. **内存不足**
   - 减少GPU设备数量
   - 使用较小的并行策略

4. **性能不理想**
   - 尝试不同的并行策略
   - 调整调度策略
   - 检查GPU负载

### 日志信息

插件会输出详细的日志信息，包括：
- 多GPU初始化状态
- 调度策略选择
- 性能指标
- 错误和回退信息

## 性能优化建议

1. **GPU配置**:
   - 使用相同型号的GPU
   - 确保GPU间通信正常
   - 监控GPU温度和功耗

2. **模型选择**:
   - 大模型使用PipeFusion或Tensor策略
   - 小模型使用Hybrid策略
   - 根据模型特点调整参数

3. **工作流优化**:
   - 合理设置batch size
   - 避免不必要的模型重载
   - 使用适当的采样参数

## 版本历史

### v1.0.0
- 初始版本
- 支持所有主要ComfyUI节点
- 完整的Flux模型支持
- 多种并行和调度策略

## 许可证

本项目遵循与ComfyUI相同的许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个插件。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。 