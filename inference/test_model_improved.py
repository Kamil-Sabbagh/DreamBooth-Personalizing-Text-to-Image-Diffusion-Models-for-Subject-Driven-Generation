#!/usr/bin/env python3
"""
Improved test script for your trained DreamBooth dog model
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def test_dog_model_improved():
    print("🐕 Testing your trained DreamBooth dog model with improved prompts...")
    
    # Load your trained model
    model_path = "./trained-model"
    print(f"Loading model from: {model_path}")
    
    # Create pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
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
    
    # Test different types of prompts
    test_cases = [
        # Basic prompts
        ("a photo of sks dog", "basic"),
        ("a photo of dog", "basic_class"),
        
        # Location prompts
        ("a photo of sks dog in a garden", "location"),
        ("a photo of sks dog in a park", "location"),
        ("a photo of sks dog on a beach", "location"),
        
        # Action prompts
        ("a photo of sks dog playing", "action"),
        ("a photo of sks dog sleeping", "action"),
        ("a photo of sks dog running", "action"),
        
        # Style prompts
        ("a professional photo of sks dog", "style"),
        ("a cartoon of sks dog", "style"),
        ("a painting of sks dog", "style"),
        
        # Comparison prompts
        ("a photo of a golden retriever", "comparison"),
        ("a photo of a labrador", "comparison"),
    ]
    
    print(f"\n🎨 Testing {len(test_cases)} different prompt types...")
    
    # Create output directory
    os.makedirs("improved_test_images", exist_ok=True)
    
    # Generate images
    for i, (prompt, category) in enumerate(test_cases):
        print(f"Generating image {i+1}/{len(test_cases)} ({category}): '{prompt}'")
        
        try:
            # Generate image with different settings
            image = pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512,
                negative_prompt="blurry, low quality, distorted"  # Add negative prompt
            ).images[0]
            
            # Save image
            filename = f"improved_test_images/{category}_{i+1:02d}.png"
            image.save(filename)
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating image {i+1}: {e}")
    
    print(f"\n🎉 Test complete! Check the 'improved_test_images' folder.")
    print("\n📊 Analysis:")
    print("- If 'basic' images show your dog: ✅ Model learned the concept")
    print("- If 'location' images show your dog in different places: ✅ Model understands context")
    print("- If 'comparison' images show different dogs: ✅ Model can distinguish")
    print("- If all images look the same: ❌ Model overfitted to training data")

if __name__ == "__main__":
    test_dog_model_improved()
