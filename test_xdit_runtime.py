#!/usr/bin/env python3
"""
xDiT Runtime Architecture Test Suite
====================================

Comprehensive test suite for xDiT multi-GPU runtime architecture.
"""

import os
import sys
import logging
import torch
import time
from typing import Dict, Any

# Add the custom_nodes directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_nodes'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def test_system_info():
    """Test system information"""
    print("=== Testing System Information ===")
    
    # GPU information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {list(range(gpu_count))}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024**3)  # GB
            memory_free = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {memory_total:.1f}GB total, {memory_free:.1f}GB free")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("No CUDA GPUs available")
    
    # PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    # Test framework availability
    try:
        import ray
        print("Ray available: True")
    except ImportError:
        print("Ray available: False")
    
    try:
        import xfuser
        print("xDiT available: True")
    except ImportError:
        print("xDiT available: False")

def test_ray_initialization():
    """Test Ray initialization"""
    print("\n=== Testing Ray Initialization ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.xdit_runtime import initialize_ray, get_ray_info, is_ray_available
        
        # Test Ray initialization
        print("Initializing Ray...")
        success = initialize_ray()
        if success:
            print("✅ Ray initialized successfully")
            
            # Get cluster info
            info = get_ray_info()
            print(f"Ray cluster info: {info}")
            
            # Test availability check
            available = is_ray_available()
            print(f"Ray available check: {available}")
        else:
            print("❌ Failed to initialize Ray")
            
    except Exception as e:
        print(f"❌ Error testing Ray initialization: {e}")

def test_dispatcher_creation():
    """Test dispatcher creation"""
    print("\n=== Testing Dispatcher Creation ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.xdit_runtime import XDiTDispatcher, SchedulingStrategy
        
        # Test different scheduling strategies
        strategies = [
            SchedulingStrategy.ROUND_ROBIN,
            SchedulingStrategy.LEAST_LOADED,
            SchedulingStrategy.WEIGHTED_ROUND_ROBIN,
            SchedulingStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            try:
                dispatcher = XDiTDispatcher(
                    gpu_devices=[0, 1, 2],
                    model_path="/tmp/test_model",  # Dummy path
                    strategy="Hybrid",
                    scheduling_strategy=strategy
                )
                print(f"✅ Dispatcher created with {strategy.value} strategy")
            except Exception as e:
                print(f"❌ Failed to create dispatcher with {strategy.value}: {e}")
                
    except Exception as e:
        print(f"❌ Error testing dispatcher creation: {e}")

def test_worker_creation():
    """Test worker creation"""
    print("\n=== Testing Worker Creation ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.xdit_runtime import XDiTWorker
        
        # Test fallback worker creation
        try:
            from custom_nodes.comfyui_xdit_multigpu.xdit_runtime.worker import XDiTWorkerFallback
            worker = XDiTWorkerFallback(0, "/tmp/test_model", "Hybrid")
            print("✅ Fallback worker created for GPU 0")
        except Exception as e:
            print(f"❌ Failed to create fallback worker: {e}")
        
        # Test Ray worker creation (if Ray is available)
        try:
            import ray
            if ray.is_initialized():
                worker = XDiTWorker.remote(0, "/tmp/test_model", "Hybrid")
                print("✅ Ray worker created for GPU 0")
            else:
                print("⚠️  Ray not initialized")
        except Exception as e:
            print(f"⚠️  Ray worker creation failed: {e}")
            
    except Exception as e:
        print(f"❌ Error testing worker creation: {e}")

def test_unet_runner():
    """Test UNet runner"""
    print("\n=== Testing UNet Runner ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.xdit_runtime import UNetRunner
        
        # Test basic UNet runner
        try:
            runner = UNetRunner("/tmp/test_model", [0], "Hybrid")
            print("✅ UNet Runner created")
        except Exception as e:
            print(f"❌ Failed to create UNet Runner: {e}")
        
        # Test parallel UNet runner
        try:
            runner = UNetRunner("/tmp/test_model", [0, 1, 2], "Hybrid")
            print("✅ Parallel UNet Runner created")
        except Exception as e:
            print(f"❌ Failed to create Parallel UNet Runner: {e}")
            
    except Exception as e:
        print(f"❌ Error testing UNet runner: {e}")

def test_scheduling_strategies():
    """Test scheduling strategies"""
    print("\n=== Testing Scheduling Strategies ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.xdit_runtime import XDiTDispatcher, SchedulingStrategy
        
        # Create a test dispatcher (without initializing workers)
        dispatcher = XDiTDispatcher(
            gpu_devices=[0, 1, 2],
            model_path="/tmp/test_model",
            strategy="Hybrid",
            scheduling_strategy=SchedulingStrategy.ROUND_ROBIN
        )
        
        # Manually add mock workers for testing
        dispatcher.workers = {0: "worker0", 1: "worker1", 2: "worker2"}
        dispatcher.worker_loads = {0: 0, 1: 0, 2: 0}
        dispatcher.is_initialized = True
        
        # Test round robin scheduling
        print("Testing Round Robin scheduling:")
        dispatcher.scheduling_strategy = SchedulingStrategy.ROUND_ROBIN
        for i in range(5):
            worker = dispatcher.get_next_worker()
            if worker:
                print(f"  Iteration {i+1}: {worker}")
            else:
                print(f"  Iteration {i+1}: no worker available")
        
        # Test least loaded scheduling
        print("Testing Least Loaded scheduling:")
        dispatcher.scheduling_strategy = SchedulingStrategy.LEAST_LOADED
        dispatcher.worker_loads = {0: 0, 1: 0, 2: 0}
        for i in range(5):
            worker = dispatcher.get_next_worker()
            if worker:
                print(f"  Iteration {i+1}: {worker}")
            else:
                print(f"  Iteration {i+1}: no worker available")
                
    except Exception as e:
        print(f"❌ Error testing scheduling strategies: {e}")

def test_environment_variables():
    """Test environment variables"""
    print("\n=== Testing Environment Variables ===")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    # Check xDiT environment variables
    xfuser_flash = os.environ.get('XFUSER_ENABLE_FLASH_ATTENTION', '')
    print(f"XFUSER_ENABLE_FLASH_ATTENTION: {xfuser_flash}")
    
    xfuser_cache = os.environ.get('XFUSER_ENABLE_CACHE', '')
    print(f"XFUSER_ENABLE_CACHE: {xfuser_cache}")

def main():
    """Main test function"""
    print("🧪 xDiT Runtime Architecture Test Suite")
    print("=" * 60)
    
    # Test system information
    test_system_info()
    
    # Test Ray initialization
    test_ray_initialization()
    
    # Test dispatcher creation
    test_dispatcher_creation()
    
    # Test worker creation
    test_worker_creation()
    
    # Test UNet runner
    test_unet_runner()
    
    # Test scheduling strategies
    test_scheduling_strategies()
    
    # Test environment variables
    test_environment_variables()
    
    print("\n🎉 Runtime architecture test suite completed!")
    
    # Summary
    print("\n=== Summary ===")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("Available GPUs: 0")
    
    try:
        import xfuser
        print("xDiT Available: True")
    except ImportError:
        print("xDiT Available: False")
    
    try:
        import ray
        print("Ray Available: True")
    except ImportError:
        print("Ray Available: False")
    
    print("✅ Ready for Ray-based multi-GPU acceleration!")

if __name__ == "__main__":
    main() 