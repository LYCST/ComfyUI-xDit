#!/usr/bin/env python3
"""
xDiT Ray Launcher
================

Independent launcher for xDiT multi-GPU acceleration (for debugging).
"""

import os
import sys
import argparse
import logging
import time
from typing import List

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from custom_nodes.comfyui_xdit_multigpu.xdit_runtime import XDiTDispatcher, SchedulingStrategy
from custom_nodes.comfyui_xdit_multigpu.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="xDiT Ray Launcher")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of GPU devices (e.g., '0,1,2,3')"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="Hybrid",
        choices=["PipeFusion", "USP", "Hybrid", "CFG", "Tensor"],
        help="xDiT parallel strategy"
    )
    
    parser.add_argument(
        "--scheduling",
        type=str,
        default="round_robin",
        choices=["round_robin", "least_loaded", "weighted_round_robin", "adaptive"],
        help="Worker scheduling strategy"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful landscape painting",
        help="Test prompt for inference"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps"
    )
    
    parser.add_argument(
        "--cfg",
        type=float,
        default=8.0,
        help="CFG guidance scale"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--test-iterations",
        type=int,
        default=3,
        help="Number of test iterations"
    )
    
    return parser.parse_args()

def print_system_info():
    """Print system information"""
    print("=" * 60)
    print("🚀 xDiT Ray Launcher - System Information")
    print("=" * 60)
    
    # GPU information
    available_gpus = config.get_available_gpus()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus:
        gpu_info = config.get_gpu_memory_info()
        for gpu_id, info in gpu_info.items():
            print(f"GPU {gpu_id}: {info['name']}")
            print(f"  Memory: {info['memory_total_gb']:.1f}GB total, {info['memory_free_gb']:.1f}GB free")
            print(f"  Compute Capability: {info['compute_capability']}")
    
    # PyTorch information
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    # Ray information
    try:
        import ray
        print(f"Ray available: {True}")
    except ImportError:
        print(f"Ray available: {False}")
    
    # xDiT information
    try:
        import xfuser
        print(f"xDiT available: {True}")
    except ImportError:
        print(f"xDiT available: {False}")
    
    print("=" * 60)

def run_test_inference(dispatcher, args):
    """Run test inference"""
    print(f"\n🧪 Running test inference...")
    print(f"Prompt: {args.prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Steps: {args.steps}, CFG: {args.cfg}, Seed: {args.seed}")
    
    total_time = 0
    successful_runs = 0
    
    for i in range(args.test_iterations):
        print(f"\n--- Test iteration {i+1}/{args.test_iterations} ---")
        
        start_time = time.time()
        
        try:
            result = dispatcher.run_inference(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                seed=args.seed + i
            )
            
            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time
            
            if result is not None:
                successful_runs += 1
                print(f"✅ Success! Time: {iteration_time:.2f}s")
                
                # Print result info
                if hasattr(result, 'shape'):
                    print(f"   Result shape: {result.shape}")
                if hasattr(result, 'dtype'):
                    print(f"   Result dtype: {result.dtype}")
            else:
                print(f"❌ Failed! Time: {iteration_time:.2f}s")
                
        except Exception as e:
            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time
            print(f"❌ Error: {e}")
            print(f"   Time: {iteration_time:.2f}s")
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Successful runs: {successful_runs}/{args.test_iterations}")
    print(f"   Average time: {total_time/args.test_iterations:.2f}s")
    if successful_runs > 0:
        print(f"   Average successful time: {total_time/successful_runs:.2f}s")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Print system information
    print_system_info()
    
    # Parse GPU devices
    try:
        gpu_devices = [int(x.strip()) for x in args.gpu_devices.split(",")]
    except ValueError:
        logger.error(f"Invalid GPU devices format: {args.gpu_devices}")
        return 1
    
    # Validate GPU devices
    available_gpus = config.get_available_gpus()
    invalid_gpus = [gpu for gpu in gpu_devices if gpu not in available_gpus]
    if invalid_gpus:
        logger.error(f"Invalid GPU devices: {invalid_gpus}")
        logger.error(f"Available GPUs: {available_gpus}")
        return 1
    
    # Parse scheduling strategy
    try:
        scheduling_strategy = SchedulingStrategy(args.scheduling)
    except ValueError:
        logger.error(f"Invalid scheduling strategy: {args.scheduling}")
        return 1
    
    print(f"\n🔧 Configuration:")
    print(f"   Model path: {args.model_path}")
    print(f"   GPU devices: {gpu_devices}")
    print(f"   xDiT strategy: {args.strategy}")
    print(f"   Scheduling strategy: {scheduling_strategy.value}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1
    
    # Initialize dispatcher
    print(f"\n🚀 Initializing xDiT Dispatcher...")
    dispatcher = XDiTDispatcher(
        gpu_devices=gpu_devices,
        model_path=args.model_path,
        strategy=args.strategy,
        scheduling_strategy=scheduling_strategy
    )
    
    try:
        # Initialize
        success = dispatcher.initialize()
        if not success:
            logger.error("Failed to initialize dispatcher")
            return 1
        
        print(f"✅ Dispatcher initialized successfully!")
        
        # Get status
        status = dispatcher.get_status()
        print(f"\n📊 Dispatcher Status:")
        print(f"   Initialized: {status['is_initialized']}")
        print(f"   Number of workers: {status['num_workers']}")
        print(f"   Scheduling strategy: {status['scheduling_strategy']}")
        print(f"   Worker loads: {status['worker_loads']}")
        
        # Run test inference
        run_test_inference(dispatcher, args)
        
        # Final status
        print(f"\n📊 Final Status:")
        final_status = dispatcher.get_status()
        print(f"   Worker loads: {final_status['worker_loads']}")
        
        # GPU memory info
        print(f"\n💾 GPU Memory Usage:")
        for gpu_id, gpu_info in final_status['gpu_infos'].items():
            if 'error' not in gpu_info:
                print(f"   GPU {gpu_id}: {gpu_info['memory_allocated_gb']:.1f}GB used, {gpu_info['memory_free_gb']:.1f}GB free")
            else:
                print(f"   GPU {gpu_id}: Error - {gpu_info['error']}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        return 1
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        dispatcher.cleanup()
        print(f"✅ Cleanup completed")
    
    print(f"\n🎉 xDiT Ray Launcher completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 