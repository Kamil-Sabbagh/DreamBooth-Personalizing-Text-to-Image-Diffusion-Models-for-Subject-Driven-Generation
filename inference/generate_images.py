#!/usr/bin/env python3
"""
Image generation script for your trained DreamBooth model
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def generate_images():
    print("🎨 Generating images with your trained DreamBooth model...")
    
    # Load your trained model (fallback to base model if not available)
    model_path = "./trained-model"
    
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Loading your trained model from: {model_path}")
        # Create pipeline with your trained model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
        print(f"⚠️  Trained model not found at {model_path}")
        print("📥 Please download your model first:")
        print("   mkdir -p trained-model")
        print("   modal volume get dreambooth-models /trained-model ./trained-model")
        print("🔄 Using base Stable Diffusion model for demonstration...")
        
        # Fallback to base model for demonstration
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("✅ Using CUDA GPU")
    elif torch.backends.mps.is_available():
        pipe = pipe.to("mps")
        print("✅ Using MPS (Apple Silicon)")
    else:
        print("⚠️  Using CPU (will be slower)")
    
    # Read prompts from prompts.txt file
    prompts = []
    prompts_file = "prompts.txt"
    
    if os.path.exists(prompts_file):
        try:
            with open(prompts_file, 'r') as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
            if prompts:  # Only use if file has content
                print(f"📝 Loaded {len(prompts)} prompts from {prompts_file}")
        except Exception as e:
            print(f"⚠️  Error reading {prompts_file}: {e}")
            prompts = []
    
    # Use default prompts if file is empty or doesn't exist
    if not prompts:
        print("📝 Using default prompts (prompts.txt file is empty or missing)")
        prompts = [
            "a portrait photo of sks dog",
            "sks dog wearing sunglasses, street photo", 
            "sks dog in front of the Eiffel Tower",
            "a watercolor painting of sks dog",
            "sks dog sitting on a couch",
            "sks dog running in a park",
            "sks dog on the beach at sunset",
            "studio portrait of sks dog, soft lighting",
            "sks dog with a red bandana",
            "sks dog next to a blue car"
        ]
    
    # Ensure we have exactly 10 prompts
    if len(prompts) > 10:
        prompts = prompts[:10]
        print(f"📝 Using first 10 prompts (truncated from {len(prompts)} total)")
    elif len(prompts) < 10:
        # Pad with default prompts if we have fewer than 10
        default_prompts = [
            "a portrait photo of sks dog",
            "sks dog wearing sunglasses, street photo", 
            "sks dog in front of the Eiffel Tower",
            "a watercolor painting of sks dog",
            "sks dog sitting on a couch",
            "sks dog running in a park",
            "sks dog on the beach at sunset",
            "studio portrait of sks dog, soft lighting",
            "sks dog with a red bandana",
            "sks dog next to a blue car"
        ]
        while len(prompts) < 10:
            prompts.append(default_prompts[len(prompts) % len(default_prompts)])
        print(f"📝 Padded to 10 prompts using defaults")
    
    print(f"\n🎨 Generating {len(prompts)} images...")
    
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
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
            filename = f"generated_images/{i+1:02d}_{safe_prompt}.png"
            image.save(filename)
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating image {i+1}: {e}")
    
    print(f"\n🎉 Generation complete! Check the 'generated_images' folder.")
    print(f"📊 Generated {len(prompts)} images using your custom prompts.")

if __name__ == "__main__":
    generate_images()
