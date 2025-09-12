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
def generate_images():
    """Generate images using the trained DreamBooth model on Modal"""
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    
    print("🎨 Generating images with your trained DreamBooth model on Modal...")
    
    # Load your trained model
    model_path = "/models/trained-model"
    
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Loading your trained model from: {model_path}")
        # Create pipeline with your trained model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
        print(f"⚠️  Trained model not found at {model_path}")
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
    
    # Create output directory
    output_dir = "/tmp/generated_images"
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
    volume_dir = "/models/generated_images"
    os.makedirs(volume_dir, exist_ok=True)
    
    # Copy generated images to volume
    import shutil
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            shutil.copy2(os.path.join(output_dir, filename), volume_dir)
    
    print(f"\n🎉 Generation complete! Images saved to Modal volume.")
    print(f"📊 Generated {len(prompts)} images using your custom prompts.")
    print(f"📥 Download with: modal volume get dreambooth-models /generated_images ./generated_images")

@app.local_entrypoint()
def main():
    """Local entrypoint to run the inference function"""
    import subprocess
    import os
    
    print("🚀 Starting DreamBooth inference on Modal...")
    
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
    
    generate_images.remote()
    print("✅ Inference complete! Check the generated images.")

if __name__ == "__main__":
    main()
