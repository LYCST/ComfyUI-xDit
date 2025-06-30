"""
xDiT Ray Worker
==============

Ray Actor for GPU-specific inference operations.
"""

import os
import sys
import torch
import logging
import time
import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import threading
import socket
import signal

# 🔧 关键修复：确保Ray worker能找到ComfyUI模块
# 添加ComfyUI根目录到Python路径
comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if comfyui_root not in sys.path:
    sys.path.insert(0, comfyui_root)

# 添加custom_nodes目录到Python路径
custom_nodes_path = os.path.join(comfyui_root, 'custom_nodes')
if custom_nodes_path not in sys.path:
    sys.path.insert(0, custom_nodes_path)

# 现在可以正确导入ComfyUI模块
try:
    import folder_paths
    import comfy.sd
    import comfy.utils
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ComfyUI modules not available in worker")

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Try to import xDiT with correct API
try:
    import xfuser
    from xfuser import xFuserArgs
    from xfuser.core.distributed import get_world_group
    # Import FLUX specific pipeline and config
    from xfuser import xFuserFluxPipeline
    from xfuser.config.config import EngineConfig
    XDIT_AVAILABLE = True
except ImportError:
    XDIT_AVAILABLE = False

logger = logging.getLogger(__name__)

def find_free_port():
    """Find a free port for distributed communication"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port



@ray.remote(num_gpus=1) if RAY_AVAILABLE else None
class XDiTWorker:
    """
    Ray Actor for GPU-specific xDiT operations.
    Each worker is assigned to one GPU.
    """
    
    def __init__(self, gpu_id: int, model_path: str, strategy: str = "Hybrid", 
                 master_addr: str = "127.0.0.1", master_port: int = 29500, 
                 world_size: int = 8, rank: int = 0):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.strategy = strategy
        self.model_wrapper = None
        self.is_initialized = False
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.distributed_initialized = False
        
        # 设置详细的日志级别
        logging.getLogger("xfuser").setLevel(logging.DEBUG)
        logging.getLogger("torch.distributed").setLevel(logging.DEBUG)
        
        # 设置GPU设备
        try:
            # 在Ray环境中设置CUDA设备
            if RAY_AVAILABLE:
                # Ray会自动设置CUDA_VISIBLE_DEVICES
                self.device = f"cuda:{0}"  # 在Ray中总是0
                torch.cuda.set_device(0)
            else:
                # 非Ray环境下手动设置
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                self.device = f"cuda:{0}"
                torch.cuda.set_device(0)
                
            logger.info(f"Worker initialized on GPU {gpu_id} (device={self.device}, rank={rank})")
            
        except Exception as e:
            logger.error(f"Failed to set GPU device for worker {gpu_id}: {e}")
            raise
    
    def initialize(self) -> bool:
        """Initialize the worker"""
        try:
            if not XDIT_AVAILABLE:
                logger.error(f"xDiT not available on GPU {self.gpu_id}")
                return False
            
            logger.info(f"Initializing worker on GPU {self.gpu_id} with strategy: {self.strategy}")
            
            # 验证GPU可用性
            if not torch.cuda.is_available():
                logger.error(f"CUDA not available for worker on GPU {self.gpu_id}")
                return False
            
            # 验证当前设备
            current_device = torch.cuda.current_device()
            logger.info(f"Current CUDA device for GPU {self.gpu_id}: {current_device}")
            
            # 标记初始化完成
            self.is_initialized = True
            logger.info(f"✅ Worker on GPU {self.gpu_id} initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return False
    
    def initialize_distributed(self) -> bool:
        """Initialize distributed environment - called after all workers are ready"""
        if self.distributed_initialized:
            return True
            
        try:
            logger.info(f"[GPU {self.gpu_id}] Initializing distributed environment...")
            
            # 清理可能存在的旧进程组
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                    logger.info(f"[GPU {self.gpu_id}] Destroyed existing process group")
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")
            
            # 设置分布式环境变量
            os.environ['MASTER_ADDR'] = self.master_addr
            os.environ['MASTER_PORT'] = str(self.master_port)
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['RANK'] = str(self.rank)
            os.environ['LOCAL_RANK'] = '0'  # Ray中每个worker都是local rank 0
            os.environ['NCCL_DEBUG'] = 'INFO'  # 启用NCCL调试
            
            logger.info(f"[GPU {self.gpu_id}] Distributed env: MASTER={self.master_addr}:{self.master_port}, "
                       f"WORLD_SIZE={self.world_size}, RANK={self.rank}")
            
            # 初始化分布式环境
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=f'tcp://{self.master_addr}:{self.master_port}',
                world_size=self.world_size,
                rank=self.rank,
                timeout=datetime.timedelta(minutes=10)  # 增加超时时间
            )
            
            # 验证分布式初始化
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                logger.info(f"[GPU {self.gpu_id}] ✅ Distributed initialized: world_size={world_size}, rank={rank}")
                self.distributed_initialized = True
                return True
            else:
                logger.error(f"[GPU {self.gpu_id}] Distributed initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Failed to initialize distributed: {e}")
            logger.exception("Distributed init traceback:")
            self._cleanup_distributed()
            return False
    
    def _create_xfuser_pipeline_if_needed(self, model_path: str = None) -> bool:
        """创建xfuser pipeline - 使用xDiT的原始方法"""
        if self.model_wrapper is not None and self.model_wrapper != "deferred_loading":
            return True
            
        try:
            # 使用传递的模型路径，如果没有则使用默认路径
            effective_model_path = model_path or self.model_path
            logger.info(f"[GPU {self.gpu_id}] Creating xfuser pipeline with path: {effective_model_path}")
            
            # 确保分布式环境已初始化
            if not self.distributed_initialized:
                logger.error(f"[GPU {self.gpu_id}] Distributed not initialized")
                return False
            
            # 创建pipeline - 使用xDiT的原始方法
            try:
                # 确保在正确的设备上
                torch.cuda.set_device(0)  # Ray中总是0
                
                # 对于safetensors文件，需要特殊处理
                if effective_model_path.endswith('.safetensors'):
                    logger.info(f"[GPU {self.gpu_id}] Processing safetensors file")
                    
                    # 方案1：尝试使用diffusers的from_single_file
                    try:
                        from diffusers import FluxPipeline
                        from xfuser import xFuserFluxPipeline
                        
                        logger.info(f"[GPU {self.gpu_id}] Trying to load safetensors with from_single_file")
                        
                        # 使用from_single_file加载
                        pipeline = FluxPipeline.from_single_file(
                            effective_model_path,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True
                        )
                        
                        # 创建xFuser wrapper
                        self.model_wrapper = xFuserFluxPipeline(pipeline, self.engine_config)
                        self.model_wrapper.to(self.device)
                        
                        logger.info(f"[GPU {self.gpu_id}] ✅ Successfully loaded safetensors with from_single_file")
                        return True
                        
                    except Exception as e:
                        logger.warning(f"[GPU {self.gpu_id}] from_single_file failed: {e}")
                        
                        # 方案2：创建临时diffusers目录
                        logger.info(f"[GPU {self.gpu_id}] Creating temporary diffusers directory")
                        
                        import json
                        import tempfile
                        import shutil
                        
                        temp_dir = f"/tmp/flux_diffusers_{self.gpu_id}_{os.getpid()}"
                        
                        try:
                            # 清理旧的临时目录
                            if os.path.exists(temp_dir):
                                shutil.rmtree(temp_dir)
                            
                            # 创建目录结构
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            # 创建model_index.json
                            model_index = {
                                "_class_name": "FluxPipeline",
                                "_diffusers_version": "0.30.0",
                                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                                "text_encoder": ["transformers", "CLIPTextModel"],
                                "text_encoder_2": ["transformers", "T5EncoderModel"],
                                "tokenizer": ["transformers", "CLIPTokenizer"],
                                "tokenizer_2": ["transformers", "T5TokenizerFast"],
                                "transformer": ["diffusers", "FluxTransformer2DModel"],
                                "vae": ["diffusers", "AutoencoderKL"]
                            }
                            
                            with open(os.path.join(temp_dir, "model_index.json"), "w") as f:
                                json.dump(model_index, f, indent=2)
                            
                            # 创建transformer目录
                            transformer_dir = os.path.join(temp_dir, "transformer")
                            os.makedirs(transformer_dir, exist_ok=True)
                            
                            # 复制safetensors文件
                            target_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
                            shutil.copy2(effective_model_path, target_path)
                            
                            # 创建config.json
                            transformer_config = {
                                "in_channels": 64,
                                "num_layers": 19,
                                "num_single_layers": 38,
                                "attention_head_dim": 128,
                                "num_attention_heads": 24,
                                "joint_attention_dim": 4096,
                                "pooled_projection_dim": 768,
                                "guidance_embeds": False
                            }
                            
                            with open(os.path.join(transformer_dir, "config.json"), "w") as f:
                                json.dump(transformer_config, f, indent=2)
                            
                            # 使用临时目录创建pipeline
                            effective_model_path = temp_dir
                            logger.info(f"[GPU {self.gpu_id}] Created temporary diffusers directory at: {temp_dir}")
                            
                        except Exception as convert_error:
                            logger.error(f"[GPU {self.gpu_id}] Failed to create temp directory: {convert_error}")
                            return False
                
                # 处理diffusers目录
                if os.path.isdir(effective_model_path):
                    logger.info(f"[GPU {self.gpu_id}] Processing diffusers directory: {effective_model_path}")
                    
                    # 验证diffusers格式
                    model_index_path = os.path.join(effective_model_path, "model_index.json")
                    if not os.path.exists(model_index_path):
                        logger.error(f"[GPU {self.gpu_id}] Invalid diffusers directory: no model_index.json")
                        return False
                    
                    # 使用xFuserArgs创建完整配置
                    xfuser_args = xFuserArgs(
                        model=effective_model_path,
                        height=1024,
                        width=1024,
                        num_inference_steps=20,
                        guidance_scale=3.5,
                        output_type="latent",
                        tensor_parallel_degree=self.world_size,
                        use_ray=True,
                        ray_world_size=self.world_size
                    )
                    
                    engine_config = xfuser_args.create_config()
                    
                    # 创建pipeline
                    self.model_wrapper = xFuserFluxPipeline.from_pretrained(
                        effective_model_path,
                        engine_config=engine_config,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True
                    )
                    
                    logger.info(f"[GPU {self.gpu_id}] ✅ Successfully loaded diffusers directory")
                    
                else:
                    logger.error(f"[GPU {self.gpu_id}] Unsupported model path format: {effective_model_path}")
                    return False
                
                # 确保模型在正确的设备上
                self.model_wrapper.to(self.device)
                
                logger.info(f"[GPU {self.gpu_id}] ✅ Pipeline created successfully!")
                return True
                
            except Exception as e:
                logger.error(f"[GPU {self.gpu_id}] Failed to create pipeline: {e}")
                logger.exception("Pipeline creation traceback:")
                return False
                
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Pipeline creation failed: {e}")
            logger.exception("Full traceback:")
            return False

    def _create_flux_pipeline_from_comfyui_components(self, model_info: Dict) -> bool:
        """从ComfyUI组件创建FluxPipeline用于xDiT包装"""
        # 设置超时机制
        timeout_seconds = 30  # 30秒超时
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Pipeline creation timed out after {timeout_seconds} seconds")
        
        # 在Linux上设置信号处理器
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        except (AttributeError, OSError):
            # Windows不支持SIGALRM，使用线程超时
            pass
        
        try:
            logger.info(f"[GPU {self.gpu_id}] 🎯 Creating FluxPipeline from ComfyUI components (timeout: {timeout_seconds}s)")
            
            # 🚀 关键洞察：ComfyUI已经有了所有必要的组件！
            # 我们需要做的是：
            # 1. 加载safetensors作为transformer
            # 2. 从ComfyUI获取其他组件（VAE, CLIP等）
            # 3. 组装成diffusers格式的FluxPipeline
            
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 1: Importing required modules...")
            from diffusers import FluxPipeline, FluxTransformer2DModel
            import torch
            import safetensors.torch
            
            # 加载safetensors transformer权重
            safetensors_path = model_info.get('path')
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 2: Loading transformer from: {safetensors_path}")
            
            custom_weights = safetensors.torch.load_file(safetensors_path)
            logger.info(f"[GPU {self.gpu_id}] ✅ Safetensors loaded, keys: {len(custom_weights)}")
            
            # 创建FluxTransformer2DModel
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 3: Creating FluxTransformer2DModel...")
            transformer = FluxTransformer2DModel(
                patch_size=1,
                in_channels=64,
                num_layers=19,
                num_single_layers=38,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=4096,
                pooled_projection_dim=768,
                guidance_embeds=False,
                axes_dims_rope=(16, 56, 56)
            )
            
            # 加载自定义权重
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 4: Loading transformer weights...")
            missing_keys, unexpected_keys = transformer.load_state_dict(custom_weights, strict=False)
            logger.info(f"[GPU {self.gpu_id}] ✅ Transformer loaded: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
            
            # 🎯 新策略：直接使用ComfyUI已加载的组件！
            logger.info(f"[GPU {self.gpu_id}] 💡 Smart strategy: Reusing ComfyUI loaded components")
            
            # 从model_info中获取ComfyUI已加载的组件
            comfyui_vae = model_info.get('vae')
            comfyui_clip = model_info.get('clip')
            
            if comfyui_vae is None or comfyui_clip is None:
                logger.warning(f"[GPU {self.gpu_id}] ComfyUI components not available in model_info")
                logger.info(f"[GPU {self.gpu_id}] Available keys: {list(model_info.keys())}")
                logger.info(f"[GPU {self.gpu_id}] VAE: {'✅' if comfyui_vae else '❌'}, CLIP: {'✅' if comfyui_clip else '❌'}")
                return False
            
            logger.info(f"[GPU {self.gpu_id}] ✅ Found ComfyUI loaded VAE and CLIP components")
            
            # 🎯 关键：直接转换ComfyUI组件为diffusers格式
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 5: Converting ComfyUI VAE to diffusers format...")
            logger.info(f"[GPU {self.gpu_id}] ComfyUI VAE type: {type(comfyui_vae)}")
            print(f"🔍 [GPU {self.gpu_id}] DEBUG: VAE type = {type(comfyui_vae)}")  # 强制输出
            
            # ComfyUI的VAE通常是comfy.model_management.VAE类型
            # 我们需要提取其内部的diffusers VAE
            diffusers_vae = None
            vae_type_name = type(comfyui_vae).__name__
            logger.info(f"[GPU {self.gpu_id}] VAE type name: {vae_type_name}")
            print(f"🔍 [GPU {self.gpu_id}] DEBUG: VAE type name = {vae_type_name}")  # 强制输出
            
            if hasattr(comfyui_vae, 'first_stage_model'):
                diffusers_vae = comfyui_vae.first_stage_model
                logger.info(f"[GPU {self.gpu_id}] ✅ Extracted first_stage_model")
                print(f"🔍 [GPU {self.gpu_id}] DEBUG: Using first_stage_model")
            elif hasattr(comfyui_vae, 'model'):
                diffusers_vae = comfyui_vae.model
                logger.info(f"[GPU {self.gpu_id}] ✅ Extracted model")
                print(f"🔍 [GPU {self.gpu_id}] DEBUG: Using model")
            elif 'AutoencodingEngine' in vae_type_name or hasattr(comfyui_vae, 'decoder'):
                # 如果是AutoencodingEngine类型，我们需要包装它
                logger.info(f"[GPU {self.gpu_id}] 🔧 Detected AutoencodingEngine - creating wrapper")
                print(f"🔍 [GPU {self.gpu_id}] DEBUG: Creating VAE wrapper for {vae_type_name}")  # 强制输出
                
                # 创建一个与FLUX VAE兼容的config
                class VAEWrapper:
                    def __init__(self, autoencoding_engine):
                        self.autoencoding_engine = autoencoding_engine
                        # 创建一个与FLUX VAE兼容的config
                        self.config = type('Config', (), {
                            'block_out_channels': [128, 256, 512, 512],  # FLUX VAE配置
                            'in_channels': 3,
                            'out_channels': 3,
                            'latent_channels': 16,
                            'sample_size': 1024,  # FLUX默认尺寸
                            'scaling_factor': 0.3611,  # FLUX VAE scaling factor
                            'shift_factor': 0.1159,   # FLUX VAE shift factor
                        })()
                        
                        # 确保包装器有正确的设备属性
                        self.device = getattr(autoencoding_engine, 'device', torch.device('cuda:0'))
                        self.dtype = getattr(autoencoding_engine, 'dtype', torch.bfloat16)
                    
                    def encode(self, x):
                        return self.autoencoding_engine.encode(x)
                    
                    def decode(self, x):
                        return self.autoencoding_engine.decode(x)
                    
                    def to(self, device):
                        # 确保设备转移正确传递
                        self.autoencoding_engine = self.autoencoding_engine.to(device)
                        self.device = device
                        return self
                    
                    @property
                    def parameters(self):
                        return self.autoencoding_engine.parameters()
                    
                    def __getattr__(self, name):
                        # 先检查是否是我们定义的属性
                        if name in ['config', 'device', 'dtype']:
                            return object.__getattribute__(self, name)
                        # 否则从autoencoding_engine获取
                        return getattr(self.autoencoding_engine, name)
                
                diffusers_vae = VAEWrapper(comfyui_vae)
                logger.info(f"[GPU {self.gpu_id}] ✅ Created VAE wrapper with FLUX-compatible config")
                
                # 验证VAE wrapper
                if not hasattr(diffusers_vae, 'config'):
                    logger.error(f"[GPU {self.gpu_id}] VAE wrapper missing config attribute!")
                    return False
                
                logger.info(f"[GPU {self.gpu_id}] VAE config block_out_channels: {diffusers_vae.config.block_out_channels}")
                print(f"🔍 [GPU {self.gpu_id}] DEBUG: VAE wrapper created successfully with config!")
            else:
                # 如果无法提取，返回失败
                logger.error(f"[GPU {self.gpu_id}] Cannot extract diffusers VAE from ComfyUI VAE")
                logger.error(f"[GPU {self.gpu_id}] VAE type: {vae_type_name}")
                logger.error(f"[GPU {self.gpu_id}] Available attributes: {[attr for attr in dir(comfyui_vae) if not attr.startswith('_')]}")
                print(f"🔍 [GPU {self.gpu_id}] DEBUG: VAE extraction failed!")
                return False
            
            print(f"🔍 [GPU {self.gpu_id}] DEBUG: Final VAE type = {type(diffusers_vae)}")
            logger.info(f"[GPU {self.gpu_id}] ✅ VAE component ready")
            
            # 将ComfyUI的CLIP转换为diffusers格式
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 6: Converting ComfyUI CLIP to diffusers format...")
            logger.info(f"[GPU {self.gpu_id}] ComfyUI CLIP type: {type(comfyui_clip)}")
            
            # ComfyUI的CLIP可能包含多个text encoder
            # 对于FLUX，我们需要CLIP-L和T5-XXL
            text_encoder = None
            text_encoder_2 = None
            
            if hasattr(comfyui_clip, 'cond_stage_model'):
                # 尝试从ComfyUI CLIP中提取text encoder
                clip_model = comfyui_clip.cond_stage_model
                logger.info(f"[GPU {self.gpu_id}] Found cond_stage_model: {type(clip_model)}")
                
                # 检查是否有多个text encoder（FLUX需要两个）
                if hasattr(clip_model, 'clip_l'):
                    text_encoder = clip_model.clip_l
                    logger.info(f"[GPU {self.gpu_id}] ✅ Found CLIP-L from ComfyUI")
                
                if hasattr(clip_model, 't5xxl'):
                    text_encoder_2 = clip_model.t5xxl
                    logger.info(f"[GPU {self.gpu_id}] ✅ Found T5-XXL from ComfyUI")
                
                # 如果没有找到，尝试其他属性
                if text_encoder is None and hasattr(clip_model, 'transformer'):
                    text_encoder = clip_model.transformer
                    logger.info(f"[GPU {self.gpu_id}] ✅ Found transformer as text_encoder")
                
                logger.info(f"[GPU {self.gpu_id}] Available clip_model attributes: {[attr for attr in dir(clip_model) if not attr.startswith('_')]}")
            
            # 如果无法从ComfyUI提取必要的text encoders，返回失败
            if text_encoder is None or text_encoder_2 is None:
                logger.error(f"[GPU {self.gpu_id}] Missing text encoders from ComfyUI CLIP")
                logger.error(f"[GPU {self.gpu_id}] CLIP-L: {'✅' if text_encoder else '❌'}, T5-XXL: {'✅' if text_encoder_2 else '❌'}")
                return False
            
            # 创建简单的tokenizer（不需要下载）
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 7: Creating lightweight tokenizers...")
            try:
                # 🔧 修复tokenizer创建问题 - 创建最简单的tokenizer包装器
                logger.info(f"[GPU {self.gpu_id}] Creating minimal tokenizer wrappers...")
                
                class MinimalTokenizer:
                    def __init__(self, vocab_size=49408, max_length=77):
                        self.vocab_size = vocab_size
                        self.model_max_length = max_length
                        self.pad_token_id = 0
                        self.eos_token_id = 2
                        self.bos_token_id = 1
                        self.unk_token_id = 3
                    
                    def __call__(self, text, **kwargs):
                        # 返回一个基本的编码结果
                        batch_size = 1 if isinstance(text, str) else len(text)
                        return {
                            'input_ids': torch.zeros((batch_size, self.model_max_length), dtype=torch.long),
                            'attention_mask': torch.ones((batch_size, self.model_max_length), dtype=torch.long)
                        }
                    
                    def encode(self, text, **kwargs):
                        return [0] * self.model_max_length
                    
                    def decode(self, token_ids, **kwargs):
                        return ""
                    
                    def batch_decode(self, sequences, **kwargs):
                        return [""] * len(sequences)
                
                # 为CLIP和T5创建不同的tokenizer
                tokenizer = MinimalTokenizer(vocab_size=49408, max_length=77)  # CLIP tokenizer
                tokenizer_2 = MinimalTokenizer(vocab_size=32128, max_length=512)  # T5 tokenizer
                
                logger.info(f"[GPU {self.gpu_id}] ✅ Minimal tokenizer wrappers created successfully")
                
            except Exception as tokenizer_error:
                logger.error(f"[GPU {self.gpu_id}] Failed to create tokenizers: {tokenizer_error}")
                return False
            
            # 创建scheduler（轻量级）
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 8: Creating scheduler...")
            try:
                from diffusers import FlowMatchEulerDiscreteScheduler
                scheduler = FlowMatchEulerDiscreteScheduler()
                logger.info(f"[GPU {self.gpu_id}] ✅ Scheduler created successfully")
            except Exception as scheduler_error:
                logger.error(f"[GPU {self.gpu_id}] Failed to create scheduler: {scheduler_error}")
                return False
            
            logger.info(f"[GPU {self.gpu_id}] ✅ All components ready for pipeline assembly")
            
            # 组装FluxPipeline
            logger.info(f"[GPU {self.gpu_id}] 📥 Step 9: Assembling FluxPipeline...")
            try:
                pipeline = FluxPipeline(
                    transformer=transformer,
                    scheduler=scheduler,
                    vae=diffusers_vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                )
                
                logger.info(f"[GPU {self.gpu_id}] 🎉 FluxPipeline created successfully using ComfyUI components!")
                
                # 创建xFuser wrapper
                logger.info(f"[GPU {self.gpu_id}] 📥 Step 10: Creating xFuser wrapper...")
                from xfuser import xFuserFluxPipeline
                self.model_wrapper = xFuserFluxPipeline(pipeline, self.engine_config)
                
                logger.info(f"[GPU {self.gpu_id}] ✅ xFuser multi-GPU pipeline ready with ComfyUI components!")
                return True
                
            except Exception as pipeline_error:
                logger.error(f"[GPU {self.gpu_id}] Failed to create FluxPipeline: {pipeline_error}")
                logger.exception("Pipeline creation error:")
                return False
                
        except TimeoutError as e:
            logger.error(f"[GPU {self.gpu_id}] Pipeline creation timed out: {e}")
            return False
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Failed to create FluxPipeline from ComfyUI components: {e}")
            logger.exception("Component creation traceback:")
            return False
        finally:
            # 清理超时设置
            try:
                signal.alarm(0)  # 取消超时
            except (AttributeError, OSError):
                pass
    
    def _cleanup_distributed(self):
        """清理分布式环境"""
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info(f"[GPU {self.gpu_id}] Process group destroyed")
            
            # 清理环境变量
            for var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'NCCL_DEBUG']:
                if var in os.environ:
                    del os.environ[var]
            
            self.distributed_initialized = False
                    
        except Exception as e:
            logger.warning(f"Error during distributed cleanup: {e}")

    def _init_comfyui_models(self, model_info: Dict) -> bool:
        """直接使用ComfyUI的模型文件初始化"""
        logger.info(f"[GPU {self.gpu_id}] Initializing with ComfyUI models")
        
        try:
            # 简化导入逻辑
            import sys
            import os
            
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 如果当前目录不在sys.path中，添加它
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # 现在尝试导入
            from comfyui_model_wrapper import ComfyUIModelWrapper
            
            # 获取模型路径
            model_path = model_info.get('path', self.model_path)
            
            # 获取VAE和CLIP路径（如果有的话）
            vae_path = None
            clip_paths = []
            
            # 尝试自动查找相关的VAE和CLIP
            if model_path.endswith('.safetensors'):
                model_name = os.path.basename(model_path).replace('.safetensors', '')
                
                # 查找VAE
                try:
                    vae_folder = folder_paths.get_folder_paths("vae")[0]
                    potential_vae = os.path.join(vae_folder, "flux", "ae.safetensors")
                    if os.path.exists(potential_vae):
                        vae_path = potential_vae
                        logger.info(f"[GPU {self.gpu_id}] Found VAE: {vae_path}")
                except Exception as e:
                    logger.warning(f"[GPU {self.gpu_id}] Could not find VAE: {e}")
                
                # 查找CLIP
                try:
                    clip_folder = folder_paths.get_folder_paths("text_encoders")[0]
                    clip_l_path = os.path.join(clip_folder, "flux", "t5xxl_fp16.safetensors")
                    clip_g_path = os.path.join(clip_folder, "flux", "clip_l.safetensors")
                    
                    if os.path.exists(clip_l_path):
                        clip_paths.append(clip_l_path)
                    if os.path.exists(clip_g_path):
                        clip_paths.append(clip_g_path)
                    
                    if clip_paths:
                        logger.info(f"[GPU {self.gpu_id}] Found CLIP: {clip_paths}")
                except Exception as e:
                    logger.warning(f"[GPU {self.gpu_id}] Could not find CLIP: {e}")
            
            # 创建wrapper
            self.comfyui_wrapper = ComfyUIModelWrapper(
                model_path=model_path,
                vae_path=vae_path,
                clip_paths=clip_paths
            )
            
            # 加载组件
            if not self.comfyui_wrapper.load_components():
                logger.error(f"[GPU {self.gpu_id}] Failed to load ComfyUI components")
                return False
            
            # 移动到GPU
            self.comfyui_wrapper.to(self.device)
            
            # 标记为ComfyUI模式
            self.model_wrapper = "comfyui_mode"
            self.is_comfyui_mode = True
            
            logger.info(f"[GPU {self.gpu_id}] ✅ ComfyUI models initialized")
            return True
            
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Failed to init ComfyUI models: {e}")
            logger.exception("Init error:")
            return False

    def _run_comfyui_inference(self,
                          conditioning_positive: Any,
                          conditioning_negative: Any,
                          latent_samples: torch.Tensor,
                          num_inference_steps: int,
                          guidance_scale: float,
                          seed: int) -> Optional[torch.Tensor]:
        """使用ComfyUI模型进行推理"""
        try:
            logger.info(f"[GPU {self.gpu_id}] Starting ComfyUI inference")
            
            # 设置随机种子
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            # 确保数据在正确的设备上
            latent_samples = latent_samples.to(self.device)
            
            # 使用ComfyUI的采样方法
            import comfy.sample
            import comfy.samplers
            
            # 创建噪声
            noise = torch.randn_like(latent_samples)
            
            # 获取采样器
            sampler = comfy.samplers.KSampler(
                self.comfyui_wrapper.unet,
                steps=num_inference_steps,
                device=self.device,
                sampler="euler",
                scheduler="normal",
                denoise=1.0
            )
            
            # 执行采样
            samples = sampler.sample(
                noise,
                conditioning_positive,
                conditioning_negative,
                cfg=guidance_scale,
                latent_image=latent_samples,
                force_full_denoise=True
            )
            
            logger.info(f"[GPU {self.gpu_id}] ✅ ComfyUI inference completed")
            return samples
            
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] ComfyUI inference failed: {e}")
            logger.exception("Inference error:")
            return None

    
    def run_inference(self, 
                     model_info: Dict,
                     conditioning_positive: Any,
                     conditioning_negative: Any,
                     latent_samples: torch.Tensor,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[torch.Tensor]:
        """运行推理 - 使用xDiT的原始方法"""
        try:
            if not self.is_initialized:
                logger.error(f"Worker on GPU {self.gpu_id} not initialized")
                return None

            # 检查是否是ComfyUI模式
            if hasattr(self, 'is_comfyui_mode') and self.is_comfyui_mode:
                logger.info(f"[GPU {self.gpu_id}] Running in ComfyUI mode")
                return self._run_comfyui_inference(
                    conditioning_positive, conditioning_negative,
                    latent_samples, num_inference_steps,
                    guidance_scale, seed
                )
            
            # 检查是否需要分布式推理
            if self.world_size > 1:
                # 多GPU模式：确保分布式已初始化
                if not self.distributed_initialized:
                    logger.error(f"[GPU {self.gpu_id}] Distributed not initialized for multi-GPU inference")
                    return None
                
                # 创建pipeline，传递正确的模型路径
                model_path = model_info.get('path', self.model_path)
                
                # 🎯 简化：直接创建xDiT pipeline
                if not self._create_xfuser_pipeline_if_needed(model_path):
                    logger.warning(f"⚠️ Pipeline creation failed on GPU {self.gpu_id}")
                    return None
            else:
                # 单GPU模式：直接运行
                logger.info(f"[GPU {self.gpu_id}] Running single-GPU inference")
            
            logger.info(f"Running inference on GPU {self.gpu_id}: {num_inference_steps} steps")
            start_time = time.time()
            
            # 设置随机种子
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            # 确保数据在正确的设备上
            latent_samples = latent_samples.to(self.device)
            
            # 处理conditioning
            prompt_embeds = None
            pooled_prompt_embeds = None
            if conditioning_positive and len(conditioning_positive) > 0:
                if isinstance(conditioning_positive[0], torch.Tensor):
                    prompt_embeds = conditioning_positive[0].to(self.device)
                    if len(conditioning_positive) > 1:
                        pooled_prompt_embeds = conditioning_positive[1].to(self.device)
            
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            if conditioning_negative and len(conditioning_negative) > 0:
                if isinstance(conditioning_negative[0], torch.Tensor):
                    negative_prompt_embeds = conditioning_negative[0].to(self.device)
                    if len(conditioning_negative) > 1:
                        negative_pooled_prompt_embeds = conditioning_negative[1].to(self.device)
            
            # 如果是单GPU或分布式失败，返回None触发fallback
            if self.world_size == 1 or not self.distributed_initialized:
                logger.info(f"[GPU {self.gpu_id}] Returning None to trigger ComfyUI fallback")
                return None
            
            # 🔧 检查模型包装器状态
            if self.model_wrapper is None:
                logger.warning(f"[GPU {self.gpu_id}] Model wrapper not ready, triggering fallback")
                return None
            
            # 创建推理配置
            model_path = model_info.get('path', self.model_path)
            xfuser_args = xFuserArgs(
                model=model_path,
                height=latent_samples.shape[2] * 8,
                width=latent_samples.shape[3] * 8,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="latent",
                tensor_parallel_degree=self.world_size
            )
            
            engine_config = xfuser_args.create_config()
            logger.info(f"[GPU {self.gpu_id}] Config created: steps={num_inference_steps}, CFG={guidance_scale}")
            
            # 运行推理
            try:
                with torch.inference_mode():
                    result = self.model_wrapper(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        height=latent_samples.shape[2] * 8,
                        width=latent_samples.shape[3] * 8,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(seed),
                        output_type="latent",
                        latents=latent_samples,
                        engine_config=engine_config
                    )
                
                # 提取结果
                if hasattr(result, 'images'):
                    result_latents = result.images
                elif hasattr(result, 'latents'):
                    result_latents = result.latents
                else:
                    result_latents = result
                
                # 确保结果在正确的设备上
                result_latents = result_latents.to(self.device)
                
                end_time = time.time()
                logger.info(f"✅ Inference completed on GPU {self.gpu_id} in {end_time - start_time:.2f}s")
                
                return result_latents
                
            except Exception as e:
                logger.error(f"Inference failed on GPU {self.gpu_id}: {e}")
                logger.exception("Inference traceback:")
                return None
                
        except Exception as e:
            logger.error(f"Error during inference on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return None
    
    def cleanup(self):
        """清理资源"""
        try:
            # 清理模型
            if self.model_wrapper is not None:
                del self.model_wrapper
                self.model_wrapper = None
            
            # 清理分布式环境
            self._cleanup_distributed()
            
            # 清理CUDA缓存
            torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info(f"✅ Worker on GPU {self.gpu_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup on GPU {self.gpu_id}: {e}")
            logger.exception("Cleanup traceback:")

    def load_model(self, model_path: str, model_type: str = "flux"):
        """加载模型，支持safetensors和diffusers格式"""
        try:
            logger.info(f"[GPU {self.gpu_id}] Loading model: {model_path}")
            
            # 检查是否是ComfyUI格式（safetensors）
            if model_path.endswith('.safetensors'):
                logger.info(f"[GPU {self.gpu_id}] Detected ComfyUI format, using native loading")
                
                # 使用ComfyUI模式
                model_info = {
                    'path': model_path,
                    'type': model_type
                }
                
                if self._init_comfyui_models(model_info):
                    logger.info(f"[GPU {self.gpu_id}] ✅ ComfyUI mode activated")
                    return "comfyui_mode"
                else:
                    logger.warning(f"[GPU {self.gpu_id}] ComfyUI mode failed, using deferred loading")
                    self.model_wrapper = "deferred_loading"
                    return "deferred_loading"
                        
            elif os.path.isdir(model_path):
                # Diffusers格式目录 - 直接使用原有逻辑
                logger.info(f"[GPU {self.gpu_id}] Detected diffusers format")
                
                # 初始化engine_config（如果还没有的话）
                if not hasattr(self, 'engine_config'):
                    from xfuser import xFuserArgs
                    xfuser_args = xFuserArgs(
                        model=model_path,
                        height=1024,
                        width=1024,
                        num_inference_steps=20,
                        guidance_scale=3.5,
                        output_type="latent",
                        tensor_parallel_degree=self.world_size,
                        use_ray=True,
                        ray_world_size=self.world_size
                    )
                    self.engine_config = xfuser_args.create_config()
                
                from xfuser import xFuserFluxPipeline
                from diffusers import FluxPipeline
                import torch
                
                pipeline = FluxPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                    low_cpu_mem_usage=True
                )
                
                self.model_wrapper = xFuserFluxPipeline(pipeline, self.engine_config)
                logger.info(f"[GPU {self.gpu_id}] ✅ Diffusers model loaded successfully")
                
                return "success"
            else:
                logger.warning(f"[GPU {self.gpu_id}] Unknown model format: {model_path}")
                self.model_wrapper = "deferred_loading"
                return "deferred_loading"
                
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] Model loading failed: {e}")
            logger.exception("Full traceback:")
            
            # 最终fallback
            self.model_wrapper = "deferred_loading"
            return "deferred_loading"

# Non-Ray fallback worker for when Ray is not available
class XDiTWorkerFallback:
    """Fallback worker when Ray is not available"""
    
    def __init__(self, gpu_id: int, model_path: str, strategy: str = "Hybrid"):
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.strategy = strategy
        self.model_wrapper = None
        self.is_initialized = False
        
        logger.info(f"Initializing fallback worker on GPU {gpu_id}")
    
    def initialize(self) -> bool:
        """Initialize the fallback worker"""
        try:
            # 设置GPU设备
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
            
            logger.info(f"Fallback worker on GPU {self.gpu_id} ready")
            
            self.is_initialized = True
            logger.info(f"✅ Fallback worker on GPU {self.gpu_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback worker on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return False
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            if not torch.cuda.is_available():
                return {"gpu_id": self.gpu_id, "error": "CUDA not available"}
                
            props = torch.cuda.get_device_properties(self.gpu_id)
            memory_total = props.total_memory / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(self.gpu_id) / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
            
            return {
                "gpu_id": self.gpu_id,
                "name": props.name,
                "memory_total_gb": memory_total,
                "memory_reserved_gb": memory_reserved,
                "memory_allocated_gb": memory_allocated,
                "memory_free_gb": memory_total - memory_reserved,
                "compute_capability": f"{props.major}.{props.minor}",
                "is_initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Error getting GPU info for GPU {self.gpu_id}: {e}")
            return {"gpu_id": self.gpu_id, "error": str(e)}
    
    def wrap_model_for_xdit(self, model_state_dict: Dict, model_config: Dict) -> bool:
        """Wrap model with xDiT (fallback version)"""
        try:
            logger.info(f"Wrapping model with fallback method on GPU {self.gpu_id}")
            
            # 在没有xDiT的情况下，简单地准备好接受模型
            self.model_wrapper = "fallback_placeholder"
            
            logger.info(f"✅ Model ready on GPU {self.gpu_id} (fallback mode)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to wrap model on GPU {self.gpu_id}: {e}")
            return False
    
    def run_inference(self, 
                     model_info: Dict,  # 改为轻量级的模型信息
                     conditioning_positive: Any,
                     conditioning_negative: Any,
                     latent_samples: torch.Tensor,
                     num_inference_steps: int = 20,
                     guidance_scale: float = 8.0,
                     seed: int = 42) -> Optional[torch.Tensor]:
        """Run inference (fallback version)"""
        try:
            if not self.is_initialized:
                logger.error(f"Fallback worker on GPU {self.gpu_id} not initialized")
                return None
            
            logger.info(f"Running fallback inference on GPU {self.gpu_id}")
            logger.info(f"Model info: {model_info.get('type', 'unknown')}, Latent shape: {latent_samples.shape}")
            start_time = time.time()
            
            # 设置随机种子
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # 🔧 优化：确保数据在正确的GPU上
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            if latent_samples.device != torch.device(device):
                latent_samples = latent_samples.to(device)
            
            # 🔧 修复Flux模型通道数问题
            # Flux模型使用16通道latent space，如果输入是4通道，需要进行转换
            if latent_samples.shape[1] == 4 and model_info.get('type', '').lower() == 'flux':
                logger.info(f"Converting 4-channel to 16-channel latent for Flux model")
                # 简单的通道扩展策略（实际的xDiT会有更复杂的处理）
                # 方法1：重复通道 [1,4,H,W] -> [1,16,H,W]
                expanded_latents = latent_samples.repeat(1, 4, 1, 1)
                # 方法2：也可以用零填充
                # expanded_latents = torch.cat([latent_samples, torch.zeros_like(latent_samples).repeat(1, 3, 1, 1)], dim=1)
                latent_samples = expanded_latents
                logger.info(f"Latent shape converted to: {latent_samples.shape}")
            
            # 在fallback模式下，只是返回原始latents
            # 实际使用中会回退到ComfyUI的标准采样
            # result_latents = latent_samples.clone()
            
            # TODO: fallback模式也返回None，让系统使用标准ComfyUI采样
            logger.info(f"⚠️ Fallback worker returning None to trigger standard ComfyUI sampling")
            return None
            
            end_time = time.time()
            logger.info(f"✅ Fallback inference completed on GPU {self.gpu_id} in {end_time - start_time:.2f}s")
            
            # return result_latents
                
        except Exception as e:
            logger.error(f"Error during fallback inference on GPU {self.gpu_id}: {e}")
            logger.exception("Full traceback:")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.model_wrapper is not None:
                del self.model_wrapper
                self.model_wrapper = None
            
            self.is_initialized = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"✅ Fallback worker on GPU {self.gpu_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup on GPU {self.gpu_id}: {e}") 