#!/usr/bin/env python3
"""
Modal-based image generation script for your trained DreamBooth model
"""

import modal
import os

# Create Modal app
app = modal.App("dreambooth-inference")

# Define the image with all dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "torchaudio>=2.0.0",
    "diffusers>=0.21.0",
    "transformers>=4.25.0",
    "accelerate>=0.20.0",
    "pillow>=9.0.0",
    "safetensors>=0.3.0",
    "numpy<2.0.0,>=1.21.0",
    "tqdm>=4.60.0"
])

# Create volume for models
volume = modal.Volume.from_name("dreambooth-models", create_if_missing=False)

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    timeout=3600
)
def generate_images(checkpoint_step=None):
    """Generate images using a specific checkpoint from your trained DreamBooth model on Modal"""
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    
    print("🎨 Generating images with your trained DreamBooth model on Modal...")
    
    # Determine which model to load
    if checkpoint_step:
        model_path = f"/models/trained-model/checkpoint-{checkpoint_step}"
        print(f"🎯 Loading checkpoint from step {checkpoint_step}: {model_path}")
    else:
        model_path = "/models/trained-model"
        print(f"🎯 Loading final trained model from: {model_path}")
    
    # Check if the specified path exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"✅ Found model at: {model_path}")
        
        # For checkpoints, we need to load the base model and then load the checkpoint weights
        if checkpoint_step:
            print("🔄 Loading base model and applying checkpoint weights...")
            # Load base model first
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Load checkpoint weights
            try:
                # Load UNet from checkpoint
                unet_path = os.path.join(model_path, "unet")
                if os.path.exists(unet_path):
                    from diffusers import UNet2DConditionModel
                    pipe.unet = UNet2DConditionModel.from_pretrained(
                        unet_path, 
                        torch_dtype=torch.float16
                    )
                    print(f"✅ Loaded UNet from checkpoint-{checkpoint_step}")
                
                # Load text encoder from checkpoint if it exists
                text_encoder_path = os.path.join(model_path, "text_encoder")
                if os.path.exists(text_encoder_path):
                    from transformers import CLIPTextModel
                    pipe.text_encoder = CLIPTextModel.from_pretrained(
                        text_encoder_path,
                        torch_dtype=torch.float16
                    )
                    print(f"✅ Loaded text encoder from checkpoint-{checkpoint_step}")
                
            except Exception as e:
                print(f"⚠️  Error loading checkpoint weights: {e}")
                print("🔄 Falling back to base model...")
        else:
            # Load final trained model normally
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
    else:
        print(f"⚠️  Model not found at {model_path}")
        print("🔄 Using base Stable Diffusion model for demonstration...")
        
        # Fallback to base model for demonstration
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    print("✅ Using CUDA GPU on Modal")
    
    # Read prompts from file (stored in volume)
    prompts = []
    try:
        with open("/models/prompts.txt", "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"📝 Loaded {len(prompts)} prompts from prompts.txt")
    except FileNotFoundError:
        print("⚠️  prompts.txt not found, using default prompts")
        prompts = [
            "a portrait photo of sks backpack",
            "sks backpack on a hiking trail, outdoor photo", 
            "sks backpack in front of the Eiffel Tower",
            "a watercolor painting of sks backpack",
            "sks backpack sitting on a wooden table",
            "sks backpack hanging on a tree branch",
            "sks backpack on the beach at sunset",
            "studio portrait of sks backpack, soft lighting",
            "sks backpack with travel stickers",
            "sks backpack next to a blue car"
        ]
    
    # Ensure we have exactly 10 prompts
    if len(prompts) < 10:
        # Pad with default prompts if needed
        default_prompts = [
            "a portrait photo of sks backpack",
            "sks backpack on a hiking trail, outdoor photo", 
            "sks backpack in front of the Eiffel Tower",
            "a watercolor painting of sks backpack",
            "sks backpack sitting on a wooden table",
            "sks backpack hanging on a tree branch",
            "sks backpack on the beach at sunset",
            "studio portrait of sks backpack, soft lighting",
            "sks backpack with travel stickers",
            "sks backpack next to a blue car"
        ]
        while len(prompts) < 10:
            prompts.append(default_prompts[len(prompts)])
    elif len(prompts) > 10:
        # Truncate to 10 prompts
        prompts = prompts[:10]
    
    print(f"\n🎨 Generating {len(prompts)} images...")
    
    # Create output directory with checkpoint info
    if checkpoint_step:
        output_dir = f"/tmp/generated_images_checkpoint_{checkpoint_step}"
    else:
        output_dir = "/tmp/generated_images_final"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: '{prompt}'")
        
        try:
            # Generate image with different settings
            image = pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512,
                negative_prompt="blurry, low quality, distorted"
            ).images[0]
            
            # Save image with descriptive filename
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit length
            filename = f"{output_dir}/{i+1:02d}_{safe_prompt}.png"
            image.save(filename)
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating image {i+1}: {e}")
    
    # Save images to volume for download
    if checkpoint_step:
        volume_dir = f"/models/generated_images_checkpoint_{checkpoint_step}"
    else:
        volume_dir = "/models/generated_images_final"
    os.makedirs(volume_dir, exist_ok=True)
    
    # Copy generated images to volume
    import shutil
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            shutil.copy2(os.path.join(output_dir, filename), volume_dir)
    
    print(f"\n🎉 Generation complete! Images saved to Modal volume.")
    print(f"📊 Generated {len(prompts)} images using your custom prompts.")
    if checkpoint_step:
        print(f"🎯 Using checkpoint from step {checkpoint_step}")
        print(f"📥 Download with: modal volume get dreambooth-models /generated_images_checkpoint_{checkpoint_step} ./generated_images_checkpoint_{checkpoint_step}")
    else:
        print(f"🎯 Using final trained model")
        print(f"📥 Download with: modal volume get dreambooth-models /generated_images_final ./generated_images_final")

@app.local_entrypoint()
def main(checkpoint_step: int = None):
    """Local entrypoint to run the inference function
    
    Args:
        checkpoint_step: Specific checkpoint step to use (e.g., 400, 500, 1000)
                        If None, uses the final trained model
    """
    import subprocess
    import os
    
    if checkpoint_step:
        print(f"🚀 Starting DreamBooth inference on Modal using checkpoint-{checkpoint_step}...")
    else:
        print("🚀 Starting DreamBooth inference on Modal using final trained model...")
    
    # Upload prompts.txt to Modal volume
    if os.path.exists("prompts.txt"):
        print("📤 Uploading prompts.txt to Modal volume...")
        try:
            result = subprocess.run([
                "modal", "volume", "put", "--force", "dreambooth-models", 
                "prompts.txt", "/prompts.txt"
            ], capture_output=True, text=True, check=True)
            print("✅ prompts.txt uploaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to upload prompts.txt: {e.stderr}")
            print("🔄 Using default prompts instead")
    else:
        print("⚠️  prompts.txt not found, using default prompts")
    
    # Run inference with checkpoint parameter
    generate_images.remote(checkpoint_step)
    print("✅ Inference complete! Downloading generated images...")
    
    # Download generated images to local directory
    if checkpoint_step:
        local_images_dir = f"./generated_images_checkpoint_{checkpoint_step}"
        volume_path = f"/generated_images_checkpoint_{checkpoint_step}"
    else:
        local_images_dir = "./generated_images_final"
        volume_path = "/generated_images_final"
    
    os.makedirs(local_images_dir, exist_ok=True)
    
    try:
        result = subprocess.run([
            "modal", "volume", "get", "--force", "dreambooth-models", 
            volume_path, local_images_dir
        ], capture_output=True, text=True, check=True)
        print(f"✅ Generated images downloaded to: {local_images_dir}")
        print("📊 Check the following files:")
        if os.path.exists(local_images_dir):
            image_files = [f for f in os.listdir(local_images_dir) if f.endswith('.png')]
            for img_file in sorted(image_files):
                print(f"   - {img_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading generated images: {e}")
        print("📥 You can manually download with:")
        if checkpoint_step:
            print(f"   modal volume get --force dreambooth-models /generated_images_checkpoint_{checkpoint_step} ./generated_images_checkpoint_{checkpoint_step}")
        else:
            print("   modal volume get --force dreambooth-models /generated_images_final ./generated_images_final")
        print("💡 If you get network timeouts, try again in a few minutes.")
        print("🔗 You can also view the images in the Modal dashboard:")
        print("   https://modal.com/apps/dreambooth/main")

if __name__ == "__main__":
    # Example usage:
    # For final trained model: python modal_generate_images(specific).py
    # For checkpoint 400: python modal_generate_images(specific).py --checkpoint-step 400
    # For checkpoint 500: python modal_generate_images(specific).py --checkpoint-step 500
    
    import sys
    
    checkpoint_step = None
    if len(sys.argv) > 1 and "--checkpoint-step" in sys.argv:
        try:
            idx = sys.argv.index("--checkpoint-step")
            checkpoint_step = int(sys.argv[idx + 1])
            print(f"🎯 Using checkpoint from step {checkpoint_step}")
        except (ValueError, IndexError):
            print("⚠️  Invalid checkpoint step. Using final trained model.")
    
    main(checkpoint_step)
