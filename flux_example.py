#!/usr/bin/env python3
"""
Flux Model with xDiT Multi-GPU Example
======================================

This example shows how to use xDiT multi-GPU acceleration with Flux model components.
Flux model typically requires:
1. UNet model (diffusion model)
2. CLIP-L model (text encoder)
3. T5-XXL model (text encoder)
4. VAE model (autoencoder)
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

def test_flux_components():
    """Test Flux model components with xDiT multi-GPU"""
    print("=== Flux Model with xDiT Multi-GPU Example ===")
    
    try:
        from custom_nodes.comfyui_xdit_multigpu.nodes import (
            XDiTUNetLoader,
            XDiTCLIPLoader,
            XDiTDualCLIPLoader,
            XDiTVAELoader,
            XDiTKSampler
        )
        
        print("✅ All xDiT nodes imported successfully")
        
        # Test UNet Loader
        print("\n--- Testing UNet Loader ---")
        unet_loader = XDiTUNetLoader()
        unet_inputs = unet_loader.INPUT_TYPES()
        print(f"UNet inputs: {list(unet_inputs.get('required', {}).keys())}")
        print(f"UNet returns: {unet_loader.RETURN_TYPES}")
        
        # Test CLIP Loader (for CLIP-L)
        print("\n--- Testing CLIP Loader (CLIP-L) ---")
        clip_loader = XDiTCLIPLoader()
        clip_inputs = clip_loader.INPUT_TYPES()
        print(f"CLIP inputs: {list(clip_inputs.get('required', {}).keys())}")
        print(f"CLIP returns: {clip_loader.RETURN_TYPES}")
        
        # Test Dual CLIP Loader (for CLIP-L + T5-XXL)
        print("\n--- Testing Dual CLIP Loader (CLIP-L + T5-XXL) ---")
        dual_clip_loader = XDiTDualCLIPLoader()
        dual_clip_inputs = dual_clip_loader.INPUT_TYPES()
        print(f"Dual CLIP inputs: {list(dual_clip_inputs.get('required', {}).keys())}")
        print(f"Dual CLIP returns: {dual_clip_loader.RETURN_TYPES}")
        
        # Test VAE Loader
        print("\n--- Testing VAE Loader ---")
        vae_loader = XDiTVAELoader()
        vae_inputs = vae_loader.INPUT_TYPES()
        print(f"VAE inputs: {list(vae_inputs.get('required', {}).keys())}")
        print(f"VAE returns: {vae_loader.RETURN_TYPES}")
        
        # Test KSampler
        print("\n--- Testing KSampler ---")
        sampler = XDiTKSampler()
        sampler_inputs = sampler.INPUT_TYPES()
        print(f"KSampler required inputs: {list(sampler_inputs.get('required', {}).keys())}")
        print(f"KSampler optional inputs: {list(sampler_inputs.get('optional', {}).keys())}")
        print(f"KSampler returns: {sampler.RETURN_TYPES}")
        
        print("\n✅ All Flux components tested successfully")
        
    except Exception as e:
        print(f"❌ Error testing Flux components: {e}")

def show_flux_workflow():
    """Show Flux model workflow with xDiT multi-GPU"""
    print("\n=== Flux Model Workflow with xDiT Multi-GPU ===")
    
    print("""
📋 Flux Model Components:
   1. UNet (diffusion model) - Load with XDiTUNetLoader
   2. CLIP-L (text encoder) - Load with XDiTCLIPLoader
   3. T5-XXL (text encoder) - Load with XDiTCLIPLoader
   4. VAE (autoencoder) - Load with XDiTVAELoader
   5. Dual CLIP (CLIP-L + T5-XXL) - Load with XDiTDualCLIPLoader

🔗 ComfyUI Node Connections:
   
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

⚙️  Multi-GPU Configuration:
   - GPU Devices: 0,1,2,3 (4x RTX 4090)
   - Parallel Strategy: Hybrid (recommended for Flux)
   - Scheduling Strategy: round_robin (balanced load)
   - Enable Multi-GPU: True

🚀 Expected Benefits:
   - Faster single image generation
   - Better memory utilization across GPUs
   - Automatic fallback to single-GPU if needed
   - Seamless compatibility with existing workflows
""")

def show_usage_instructions():
    """Show detailed usage instructions"""
    print("\n=== Usage Instructions ===")
    
    print("""
📝 Step-by-Step Setup:

1. **Load UNet with Multi-GPU:**
   - Use "Load UNet (xDiT Multi-GPU)" node
   - Select your Flux UNet model
   - Enable multi-GPU acceleration
   - Set GPU devices (e.g., "0,1,2,3")
   - Choose "Hybrid" parallel strategy

2. **Load Dual CLIP:**
   - Use "Load Dual CLIP (xDiT)" node
   - Select CLIP-L model for clip_name1
   - Select T5-XXL model for clip_name2
   - Set type to "flux"

3. **Load VAE:**
   - Use "Load VAE (xDiT)" node
   - Select your Flux VAE model

4. **Text Encoding:**
   - Use standard "CLIP Text Encode" node
   - Connect Dual CLIP output
   - Enter your prompt

5. **Sampling:**
   - Use "KSampler (xDiT Multi-GPU)" node
   - Connect UNet MODEL output
   - Connect UNet XDIT_DISPATCHER output (optional, for multi-GPU)
   - Connect CLIP conditioning
   - Set your sampling parameters

6. **Decoding:**
   - Use standard "VAE Decode" node
   - Connect VAE and sampler outputs

🔄 Fallback Behavior:
   - If xDiT is not available: Uses standard single-GPU mode
   - If multi-GPU fails: Falls back to single-GPU mode
   - If dispatcher fails: Uses standard KSampler
   - All fallbacks are automatic and transparent

💡 Tips:
   - Start with "Hybrid" parallel strategy for best performance
   - Use "round_robin" scheduling for balanced load
   - Monitor GPU memory usage during generation
   - XDIT_DISPATCHER connection is optional but recommended for multi-GPU
""")

def main():
    """Main function"""
    print("🎨 Flux Model with xDiT Multi-GPU Example")
    print("=" * 60)
    
    # Test components
    test_flux_components()
    
    # Show workflow
    show_flux_workflow()
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n🎉 Flux model example completed!")
    print("\n💡 Ready to use Flux model with xDiT multi-GPU acceleration!")

if __name__ == "__main__":
    main() 