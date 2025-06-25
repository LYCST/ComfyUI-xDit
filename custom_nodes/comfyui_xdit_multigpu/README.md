# ComfyUI xDiT Multi-GPU Acceleration Plugin

这是一个为ComfyUI提供多GPU加速的插件，基于xDiT框架实现。该插件支持多卡模型加载和多卡采样加速，显著提升图像生成速度和显存利用率。

## 功能特性

- 🚀 **多卡模型加载**: 将模型分布到多张GPU上，提高显存利用率
- ⚡ **多卡采样加速**: 利用多GPU并行处理加速单张图片生成
- 🔧 **多种并行策略**: 支持PipeFusion、USP、Hybrid、CFG等多种并行策略
- 🛡️ **自动降级**: 当xDiT不可用时自动降级到单GPU模式
- 📊 **实时监控**: 提供详细的GPU使用情况和性能日志

## 安装要求

### 系统要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 多张NVIDIA GPU

### 依赖安装

1. **安装xDiT框架**:
   ```bash
   git clone https://github.com/xdit-project/xDiT.git
   cd xDiT
   pip install -e .
   ```

2. **安装其他依赖**:
   ```bash
   pip install onediff -U nexfort ray
   ```

3. **安装插件**:
   将此文件夹复制到ComfyUI的`custom_nodes`目录下。

## 节点说明

### 1. XDiT UNet Loader
**位置**: `xDiT/Multi-GPU` 分类

**功能**: 使用xDiT框架加载模型到多GPU

**参数**:
- `model_name`: 模型文件名
- `gpu_devices`: GPU设备列表 (如 "0,1,2,3")
- `parallel_strategy`: 并行策略 (PipeFusion/USP/Hybrid/Tensor)
- `use_cache`: 是否使用缓存
- `use_flash_attention`: 是否使用Flash Attention
- `vae_name`: VAE文件名 (可选)
- `clip_name`: CLIP文件名 (可选)

### 2. XDiT Sampler (Advanced)
**位置**: `xDiT/Multi-GPU` 分类

**功能**: 使用xDiT框架进行多GPU采样

**参数**:
- `model`: 模型输入
- `seed`: 随机种子
- `steps`: 采样步数
- `cfg`: CFG引导强度
- `sampler_name`: 采样器名称
- `scheduler`: 调度器类型
- `positive`: 正向提示词
- `negative`: 负向提示词
- `latent_image`: 潜在图像
- `denoise`: 去噪强度
- `gpu_devices`: GPU设备列表
- `parallel_strategy`: 并行策略 (PipeFusion/USP/Hybrid/CFG)
- `batch_size`: 批次大小

### 3. Multi-GPU Model Loader
**位置**: `Multi-GPU` 分类

**功能**: 基础多GPU模型加载器 (降级版本)

### 4. Multi-GPU Sampler
**位置**: `Multi-GPU` 分类

**功能**: 基础多GPU采样器 (降级版本)

## 使用方法

### 基本工作流

1. **使用XDiT节点** (推荐):
   ```
   XDiT UNet Loader → XDiT Sampler (Advanced) → VAE Decode → Save Image
   ```

2. **使用基础多GPU节点** (降级模式):
   ```
   Multi-GPU Model Loader → Multi-GPU Sampler → VAE Decode → Save Image
   ```

### 配置示例

#### 4卡配置
- GPU设备: "0,1,2,3"
- 并行策略: "Hybrid"
- 批次大小: 1-4

#### 2卡配置
- GPU设备: "0,1"
- 并行策略: "USP"
- 批次大小: 1-2

## 性能优化建议

1. **GPU选择**: 确保所有GPU型号相同，显存大小相近
2. **并行策略**: 
   - 小模型 (< 2B参数): 使用USP或CFG
   - 大模型 (> 2B参数): 使用PipeFusion或Hybrid
3. **显存管理**: 根据显存大小调整批次大小
4. **缓存使用**: 对于重复生成，启用缓存可显著提升性能

## 故障排除

### 常见问题

1. **xDiT导入失败**
   - 确保正确安装了xDiT框架
   - 检查Python路径设置
   - 插件会自动降级到单GPU模式

2. **显存不足**
   - 减少批次大小
   - 使用更少的GPU设备
   - 启用显存优化选项

3. **性能不理想**
   - 检查GPU型号是否一致
   - 尝试不同的并行策略
   - 确保CUDA版本兼容

### 日志查看

插件会输出详细的日志信息，包括：
- GPU使用情况
- 模型加载状态
- 采样性能数据
- 错误和警告信息

## 技术细节

### 并行策略说明

- **PipeFusion**: 基于patch的流水线并行，适合大模型
- **USP**: 统一序列并行，适合长序列处理
- **Hybrid**: 混合并行策略，平衡性能和兼容性
- **CFG**: CFG并行，适合分类器引导生成
- **Tensor**: 张量并行，适合模型参数分布

### 架构设计

插件采用分层设计：
1. **接口层**: ComfyUI节点接口
2. **适配层**: xDiT框架适配
3. **执行层**: 多GPU执行引擎
4. **降级层**: 单GPU兼容性保证

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基本的xDiT集成
- 提供降级兼容性
- 实现多GPU模型加载和采样

## 贡献

欢迎提交Issue和Pull Request来改进这个插件！

## 许可证

本项目采用MIT许可证。

## 致谢

- [xDiT项目](https://github.com/xdit-project/xDiT) - 提供多GPU并行框架
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 优秀的AI图像生成界面
- 所有贡献者和用户的支持 