#!/usr/bin/env python3
"""
Test script for your trained DreamBooth dog model
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def test_dog_model():
    print("🐕 Testing your trained DreamBooth dog model...")
    
    # Load your trained model
    model_path = "./trained-model"
    print(f"Loading model from: {model_path}")
    
    # Create pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,  # Disable safety checker for testing
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
    
    # Test prompts
    prompts = [
        "a photo of sks dog",
        "a photo of sks dog in a garden",
        "a photo of sks dog wearing a hat",
        "a photo of sks dog playing with a ball",
        "a photo of sks dog sitting on a couch"
    ]
    
    print(f"\n🎨 Generating {len(prompts)} test images...")
    
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    # Generate images
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: '{prompt}'")
        
        try:
            # Generate image
            image = pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            # Save image
            filename = f"generated_images/dog_test_{i+1}.png"
            image.save(filename)
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating image {i+1}: {e}")
    
    print(f"\n🎉 Test complete! Check the 'generated_images' folder for your results.")
    print("Your trained model can now generate images of your specific dog!")

if __name__ == "__main__":
    test_dog_model()
