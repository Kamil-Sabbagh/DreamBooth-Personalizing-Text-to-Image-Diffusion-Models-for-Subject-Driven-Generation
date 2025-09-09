#!/usr/bin/env python3
"""
Improved DreamBooth training with better regularization and parameters
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("improved-dreambooth-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({
        "HF_HOME": "/models/hf_cache",
        "TRANSFORMERS_CACHE": "/models/hf_cache",
    })
    .pip_install([
        "torch==2.8.0",  # Exact version from working local env
        "torchvision==0.23.0",  # Exact version from working local env
        "torchaudio",  # Let pip choose compatible version
        "accelerate==1.10.1",  # Exact version from working local env
        "diffusers>=0.35.0",  # Use latest stable version
        "transformers==4.56.1",  # Exact version from working local env
        "datasets",  # Let pip choose compatible version
        "peft==0.17.1",  # Exact version from working local env
        "huggingface_hub==0.34.4",  # Exact version from working local env
        "pillow",  # Let pip choose compatible version
        "xformers",  # Let pip choose compatible version
    ])
)

# Create persistent volume for models
volume = modal.Volume.from_name("improved-dreambooth-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    timeout=3600,  # 1 hour timeout
)
def train_improved_dreambooth(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    instance_prompt: str = "a photo of sks dog",
    class_prompt: str = "a photo of dog",  # Regularization prompt
    resolution: int = 512,
    train_batch_size: int = 1,
    learning_rate: float = 1e-6,  # Lower learning rate for better stability
    max_train_steps: int = 800,  # More steps for better training
    num_class_images: int = 100,  # More regularization images
    with_prior_preservation: bool = True,  # Enable prior preservation
    prior_loss_weight: float = 0.6,  # Weight for regularization loss
):
    """Train DreamBooth with improved parameters and regularization"""
    
    import subprocess
    import sys
    from huggingface_hub import snapshot_download
    import shutil
    import os
    
    # Download training images
    print("Downloading training images...")
    local_dir = "/tmp/dog"
    snapshot_download(
        "diffusers/dog-example",
        local_dir=local_dir, 
        repo_type="dataset",
        ignore_patterns=".gitattributes",
    )
    
    # Clean up any cache directories
    cache_dir = os.path.join(local_dir, ".cache")
    if os.path.exists(cache_dir):
        print(f"Removing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # List the actual image files
    image_files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} training images: {image_files}")
    
    # Prepare class images directory for prior preservation
    class_dir = "/tmp/class_dog"
    os.makedirs(class_dir, exist_ok=True)
    print(f"Class images directory: {class_dir}")
    
    # Configure accelerate for A100
    print("Configuring accelerate for A100...")
    accelerate_config = """
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    os.makedirs("/root/.cache/huggingface/accelerate", exist_ok=True)
    with open("/root/.cache/huggingface/accelerate/default_config.yaml", "w") as f:
        f.write(accelerate_config)
    
    # Download the training script
    print("Downloading training script...")
    import requests
    script_url = "https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/examples/dreambooth/train_dreambooth.py"
    script_path = "/tmp/train_dreambooth.py"
    
    response = requests.get(script_url)
    with open(script_path, "w") as f:
        f.write(response.text)
    
    # Set output directory
    output_dir = "/models/improved-trained-model-v2"
    
    # Build training command with improved parameters
    cmd = [
        "accelerate", "launch", script_path,
        "--pretrained_model_name_or_path", model_name,
        "--instance_data_dir", local_dir,
        "--output_dir", output_dir,
        "--instance_prompt", instance_prompt,
        "--class_prompt", class_prompt,  # Add class prompt for regularization
        "--class_data_dir", class_dir,
        "--train_text_encoder",
        "--resolution", str(resolution),
        "--train_batch_size", str(train_batch_size),
        "--gradient_accumulation_steps", "1",
        "--learning_rate", str(learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(max_train_steps),
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--with_prior_preservation",  # Enable prior preservation
        "--prior_loss_weight", str(prior_loss_weight),
        "--num_class_images", str(num_class_images),  # Number of regularization images
        "--seed", "42",  # For reproducibility
    ]
    
    print(f"Running improved training command: {' '.join(cmd)}")
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Training completed successfully!")
        print("STDOUT:", result.stdout)
        return {"status": "success", "output_dir": output_dir}
    except subprocess.CalledProcessError as e:
        print("Training failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return {"status": "error", "error": str(e)}

# Local entrypoint
@app.local_entrypoint()
def main():
    print("Starting improved DreamBooth training on Modal with A100...")
    result = train_improved_dreambooth.remote()
    print(f"Training result: {result}")

if __name__ == "__main__":
    main()
