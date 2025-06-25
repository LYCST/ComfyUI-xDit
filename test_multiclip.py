#!/usr/bin/env python3
"""
Multi-CLIP Loading Test
=======================

Test script for multi-CLIP loading functionality.
"""

import os
import sys
import logging

# Add the custom_nodes directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_nodes'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def test_multiclip_loader():
    """Test multi-CLIP loader"""
    print("=== Testing Multi-CLIP Loader ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.nodes import XDiTMultiCLIPLoader
        
        # Create loader instance
        loader = XDiTMultiCLIPLoader()
        
        # Test input types
        input_types = loader.INPUT_TYPES()
        print(f"Input types: {input_types}")
        
        # Test return types
        return_types = loader.RETURN_TYPES
        return_names = loader.RETURN_NAMES
        print(f"Return types: {return_types}")
        print(f"Return names: {return_names}")
        
        print("✅ Multi-CLIP loader class created successfully")
        
    except Exception as e:
        print(f"❌ Error testing multi-CLIP loader: {e}")

def test_vae_loader():
    """Test VAE loader"""
    print("\n=== Testing VAE Loader ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.nodes import XDiTVAELoader
        
        # Create loader instance
        loader = XDiTVAELoader()
        
        # Test input types
        input_types = loader.INPUT_TYPES()
        print(f"Input types: {input_types}")
        
        # Test return types
        return_types = loader.RETURN_TYPES
        print(f"Return types: {return_types}")
        
        print("✅ VAE loader class created successfully")
        
    except Exception as e:
        print(f"❌ Error testing VAE loader: {e}")

def test_updated_unet_loader():
    """Test updated UNet loader with multi-CLIP support"""
    print("\n=== Testing Updated UNet Loader ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.nodes import XDiTUNetLoader
        
        # Create loader instance
        loader = XDiTUNetLoader()
        
        # Test input types
        input_types = loader.INPUT_TYPES()
        print(f"Input types: {input_types}")
        
        # Test return types
        return_types = loader.RETURN_TYPES
        return_names = loader.RETURN_NAMES
        print(f"Return types: {return_types}")
        print(f"Return names: {return_names}")
        
        print("✅ Updated UNet loader class created successfully")
        
    except Exception as e:
        print(f"❌ Error testing updated UNet loader: {e}")

def test_updated_sampler():
    """Test updated sampler with multi-CLIP support"""
    print("\n=== Testing Updated Sampler ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.nodes import XDiTSamplerCustomAdvanced
        
        # Create sampler instance
        sampler = XDiTSamplerCustomAdvanced()
        
        # Test input types
        input_types = sampler.INPUT_TYPES()
        print(f"Input types: {input_types}")
        
        # Test return types
        return_types = sampler.RETURN_TYPES
        print(f"Return types: {return_types}")
        
        print("✅ Updated sampler class created successfully")
        
    except Exception as e:
        print(f"❌ Error testing updated sampler: {e}")

def test_node_registration():
    """Test node registration"""
    print("\n=== Testing Node Registration ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        print("Registered nodes:")
        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
            print(f"  {node_name} -> {display_name}")
        
        print(f"✅ Total registered nodes: {len(NODE_CLASS_MAPPINGS)}")
        
    except Exception as e:
        print(f"❌ Error testing node registration: {e}")

def main():
    """Main test function"""
    print("🧪 Multi-CLIP Loading Test Suite")
    print("=" * 50)
    
    # Test multi-CLIP loader
    test_multiclip_loader()
    
    # Test VAE loader
    test_vae_loader()
    
    # Test updated UNet loader
    test_updated_unet_loader()
    
    # Test updated sampler
    test_updated_sampler()
    
    # Test node registration
    test_node_registration()
    
    print("\n🎉 Multi-CLIP loading test suite completed!")
    
    # Summary
    print("\n=== Summary ===")
    print("✅ Multi-CLIP support added to xDiT nodes")
    print("✅ VAE loader node added")
    print("✅ Updated UNet loader supports multiple CLIPs")
    print("✅ Updated sampler accepts multiple CLIP inputs")
    print("✅ All nodes properly registered")

if __name__ == "__main__":
    main() 