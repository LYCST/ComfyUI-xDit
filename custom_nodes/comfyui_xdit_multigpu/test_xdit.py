#!/usr/bin/env python3
"""
Test script for xDiT Multi-GPU Plugin
=====================================

This script tests the xDiT integration functionality.
"""

import os
import sys
import torch
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from custom_nodes.comfyui_xdit_multigpu.xdit_integration import xdit_manager, XDIT_AVAILABLE
from custom_nodes.comfyui_xdit_multigpu.config import config

def test_gpu_detection():
    """Test GPU detection"""
    print("=== Testing GPU Detection ===")
    
    available_gpus = config.get_available_gpus()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus:
        gpu_info = config.get_gpu_memory_info()
        for gpu_id, info in gpu_info.items():
            print(f"GPU {gpu_id}: {info['name']}")
            print(f"  Memory: {info['memory_total_gb']:.1f}GB total, {info['memory_free_gb']:.1f}GB free")
            print(f"  Compute Capability: {info['compute_capability']}")
    
    print()

def test_xdit_availability():
    """Test xDiT availability"""
    print("=== Testing xDiT Availability ===")
    
    if XDIT_AVAILABLE:
        print("✅ xDiT is available")
        try:
            import xfuser
            print(f"xDiT version: {xfuser.__version__ if hasattr(xfuser, '__version__') else 'Unknown'}")
        except Exception as e:
            print(f"Error importing xfuser: {e}")
    else:
        print("❌ xDiT is not available")
    
    print()

def test_parallel_configs():
    """Test parallel configuration creation"""
    print("=== Testing Parallel Configurations ===")
    
    if not XDIT_AVAILABLE:
        print("❌ xDiT not available, skipping parallel config tests")
        return
    
    gpu_devices = [0, 1, 2, 3]  # Test with 4 GPUs
    strategies = ["PipeFusion", "USP", "Hybrid", "CFG", "Tensor"]
    
    for strategy in strategies:
        try:
            config = xdit_manager.create_parallel_config(strategy, gpu_devices)
            if config:
                print(f"✅ {strategy} config created successfully")
            else:
                print(f"❌ Failed to create {strategy} config")
        except Exception as e:
            print(f"❌ Error creating {strategy} config: {e}")
    
    print()

def test_memory_management():
    """Test memory management"""
    print("=== Testing Memory Management ===")
    
    memory_info = xdit_manager.get_gpu_memory_usage()
    if memory_info:
        for gpu_id, info in memory_info.items():
            print(f"GPU {gpu_id}: {info['allocated_gb']:.1f}GB used, {info['free_gb']:.1f}GB free ({info['utilization_percent']:.1f}% utilized)")
    else:
        print("❌ No GPU memory information available")
    
    print()

def test_environment_vars():
    """Test environment variable generation"""
    print("=== Testing Environment Variables ===")
    
    env_vars = config.get_environment_vars("0,1,2,3", use_flash_attention=True, use_cache=True)
    for key, value in env_vars.items():
        print(f"{key}: {value}")
    
    print()

def main():
    """Main test function"""
    print("🧪 xDiT Multi-GPU Plugin Test Suite")
    print("=" * 50)
    
    # Test GPU detection
    test_gpu_detection()
    
    # Test xDiT availability
    test_xdit_availability()
    
    # Test parallel configurations
    test_parallel_configs()
    
    # Test memory management
    test_memory_management()
    
    # Test environment variables
    test_environment_vars()
    
    print("🎉 Test suite completed!")
    
    # Summary
    print("\n=== Summary ===")
    available_gpus = config.get_available_gpus()
    print(f"Available GPUs: {len(available_gpus)}")
    print(f"xDiT Available: {XDIT_AVAILABLE}")
    
    if XDIT_AVAILABLE and len(available_gpus) > 1:
        print("✅ Ready for multi-GPU acceleration!")
    elif XDIT_AVAILABLE:
        print("⚠️  xDiT available but only single GPU detected")
    else:
        print("❌ xDiT not available, will use fallback mode")

if __name__ == "__main__":
    main() 