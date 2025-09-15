#!/usr/bin/env python3
"""
DreamBooth training script with prior preservation and regularization
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("dreambooth-training")

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
volume = modal.Volume.from_name("dreambooth-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    timeout=3600,  # 1 hour timeout
)
def train_dreambooth(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    instance_prompt: str = None,  # Will be read from config file
    class_prompt: str = None,     # Will be read from config file
    resolution: int = None,       # Will be read from config file
    train_batch_size: int = None, # Will be read from config file
    learning_rate: float = None,  # Will be read from config file
    max_train_steps: int = None,  # Will be read from config file
    num_class_images: int = None, # Will be read from config file
    gradient_accumulation_steps: int = None, # Will be read from config file
    with_prior_preservation: bool = True,  # Enable prior preservation
    prior_loss_weight: float = None,  # Will be read from config file
    download_model: bool = False,  # Whether to download model after training
):
    """Train DreamBooth with improved parameters and regularization"""
    
    import subprocess
    import sys
    from huggingface_hub import snapshot_download
    import shutil
    import os
    
    # Read configuration from file
    def read_config():
        config = {}
        # Try to read from Modal volume first, then local file
        config_paths = ["/models/training_config.txt", "training_config.txt"]
        
        for config_path in config_paths:
            try:
                with open(config_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            config[key.strip()] = value.strip()
                print(f"✅ Loaded configuration from: {config_path}")
                break
            except FileNotFoundError:
                continue
        
        if not config:
            print("⚠️  training_config.txt not found, using default values")
            config = {
                "INSTANCE_PROMPT": "a photo of sks backpack",
                "CLASS_PROMPT": "a photo of backpack", 
                "MAX_TRAIN_STEPS": "2000",
                "LEARNING_RATE": "1e-6",
                "TRAIN_BATCH_SIZE": "2",
                "GRADIENT_ACCUMULATION_STEPS": "2",
                "PRIOR_LOSS_WEIGHT": "0.3",
                "NUM_CLASS_IMAGES": "200",
                "RESOLUTION": "512"
            }
        return config
    
    config = read_config()
    
    # Use config values or fallback to parameters/defaults
    instance_prompt = instance_prompt or config.get("INSTANCE_PROMPT", "a photo of sks backpack")
    class_prompt = class_prompt or config.get("CLASS_PROMPT", "a photo of backpack")
    max_train_steps = max_train_steps or int(config.get("MAX_TRAIN_STEPS", "2000"))
    learning_rate = learning_rate or float(config.get("LEARNING_RATE", "1e-6"))
    train_batch_size = train_batch_size or int(config.get("TRAIN_BATCH_SIZE", "2"))
    gradient_accumulation_steps = gradient_accumulation_steps or int(config.get("GRADIENT_ACCUMULATION_STEPS", "2"))
    prior_loss_weight = prior_loss_weight or float(config.get("PRIOR_LOSS_WEIGHT", "0.3"))
    num_class_images = num_class_images or int(config.get("NUM_CLASS_IMAGES", "200"))
    resolution = resolution or int(config.get("RESOLUTION", "512"))
    
    print(f"🎯 Training Configuration:")
    print(f"   Instance Prompt: {instance_prompt}")
    print(f"   Class Prompt: {class_prompt}")
    print(f"   Max Steps: {max_train_steps}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {train_batch_size}")
    print(f"   Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"   Prior Loss Weight: {prior_loss_weight}")
    print(f"   Num Class Images: {num_class_images}")
    print(f"   Resolution: {resolution}")
    
    # Use target images from Modal volume (uploaded by user)
    print("Loading training images from target/ directory...")
    local_dir = "/models/target"
    
    # Check if target images exist in volume (handle nested directory structure)
    if not os.path.exists(local_dir) or not any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(local_dir)):
        # Try nested directory structure
        nested_dir = "/models/target/target"
        if os.path.exists(nested_dir) and any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(nested_dir)):
            print(f"📁 Found images in nested directory: {nested_dir}")
            local_dir = nested_dir
        elif not os.path.exists(local_dir):
            print(f"⚠️  Target images not found at {local_dir}")
            print("📤 Please upload your target images first:")
            print("   modal volume put dreambooth-models target/ /target")
            print("🔄 Using HuggingFace dataset as fallback...")
        
        # No fallback - require proper target images
        raise ValueError("Target images not found. Please upload your target images to Modal volume first.")
    
    # Clean up any cache directories
    cache_dir = os.path.join(local_dir, ".cache")
    if os.path.exists(cache_dir):
        print(f"Removing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # List the actual image files
    image_files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} training images: {image_files}")
    
    if len(image_files) == 0:
        raise ValueError("No image files found. Please check your target images.")
    
    # Prepare class images directory for prior preservation
    # Extract class name from class_prompt (e.g., "a photo of backpack" -> "backpack")
    class_name = class_prompt.split()[-1]  # Get the last word (the object type)
    class_dir = f"/tmp/class_{class_name}"
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
    output_dir = "/models/trained-model"
    
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
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", "100",
        "--max_train_steps", str(max_train_steps),
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--with_prior_preservation",  # Enable prior preservation
        "--prior_loss_weight", str(prior_loss_weight),
        "--num_class_images", str(num_class_images),  # Number of regularization images
        "--seed", "42",  # For reproducibility
        "--checkpointing_steps", "500",  # Save checkpoints every 500 steps
        "--checkpoints_total_limit", "3",  # Keep only 3 checkpoints
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
def main(download_model: bool = False):
    import subprocess
    import os
    
    print("Starting DreamBooth training on Modal with A100...")
    print(f"Download model after training: {download_model}")
    
    # Upload target images to Modal volume
    if os.path.exists("target") and os.path.isdir("target"):
        print("📤 Uploading target images to Modal volume...")
        try:
            result = subprocess.run([
                "modal", "volume", "put", "--force", "dreambooth-models", 
                "target", "/target"
            ], capture_output=True, text=True, check=True)
            print("✅ Target images uploaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to upload target images: {e.stderr}")
            print("🔄 Training will use fallback dataset")
    else:
        print("⚠️  target/ directory not found, training will use fallback dataset")
    
    # Upload training config to Modal volume
    if os.path.exists("training_config.txt"):
        print("📤 Uploading training_config.txt to Modal volume...")
        try:
            result = subprocess.run([
                "modal", "volume", "put", "--force", "dreambooth-models", 
                "training_config.txt", "/training_config.txt"
            ], capture_output=True, text=True, check=True)
            print("✅ training_config.txt uploaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to upload training_config.txt: {e.stderr}")
            print("🔄 Using default configuration values")
    else:
        print("⚠️  training_config.txt not found, using default values")
    
    result = train_dreambooth.remote(download_model=download_model)
    print(f"Training result: {result}")
    
    if result.get("status") == "success":
        print("✅ Training completed successfully!")
        
        if download_model:
            print("📥 Downloading trained model to local directory...")
            
            # Create local directory for the trained model
            local_model_dir = "./trained-model"
            os.makedirs(local_model_dir, exist_ok=True)
            
            # Download the trained model files one by one (more reliable)
            # Note: Most of these are directories, not individual files
            model_files = [
                "model_index.json",  # This is a file
                "scheduler",         # This is a directory
                "feature_extractor", # This is a directory
                "tokenizer",         # This is a directory
                "text_encoder",      # This is a directory
                "vae",              # This is a directory
                "unet",             # This is a directory (largest ~3.2GB)
                "safety_checker"     # This is a directory
            ]
            
            successful_downloads = 0
            failed_downloads = []
            
            print("\n📥 Downloading core model files...")
            for file_name in model_files:
                try:
                    print(f"   📥 Downloading {file_name}...")
                    result = subprocess.run([
                        "modal", "volume", "get", "--force", "dreambooth-models", 
                        f"/trained-model/{file_name}", 
                        f"{local_model_dir}/{file_name}"
                    ], capture_output=True, text=True, check=True)
                    print(f"   ✅ {file_name} downloaded successfully")
                    successful_downloads += 1
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️  Failed to download {file_name}: {e.stderr}")
                    failed_downloads.append(file_name)
                except Exception as e:
                    print(f"   ⚠️  Error downloading {file_name}: {e}")
                    failed_downloads.append(file_name)
            
            # Download checkpoint directories separately
            print("\n📥 Downloading checkpoint files...")
            checkpoint_dirs = ["checkpoint-1000", "checkpoint-1500", "checkpoint-2000"]
            for checkpoint_dir in checkpoint_dirs:
                try:
                    print(f"   📥 Downloading {checkpoint_dir}...")
                    result = subprocess.run([
                        "modal", "volume", "get", "--force", "dreambooth-models", 
                        f"/trained-model/{checkpoint_dir}", 
                        f"{local_model_dir}/{checkpoint_dir}"
                    ], capture_output=True, text=True, check=True)
                    print(f"   ✅ {checkpoint_dir} downloaded successfully")
                    successful_downloads += 1
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️  Failed to download {checkpoint_dir}: {e.stderr}")
                    failed_downloads.append(checkpoint_dir)
                except Exception as e:
                    print(f"   ⚠️  Error downloading {checkpoint_dir}: {e}")
                    failed_downloads.append(checkpoint_dir)
            
            print(f"\n📊 Download Summary:")
            print(f"   ✅ Successfully downloaded: {successful_downloads} files/directories")
            if failed_downloads:
                print(f"   ❌ Failed downloads: {failed_downloads}")
                print("\n📥 You can manually retry failed downloads with:")
                for failed_file in failed_downloads:
                    print(f"   modal volume get --force dreambooth-models /trained-model/{failed_file} ./trained-model/{failed_file}")
                print("\n💡 Or use the separate download script:")
                print(f"   python train/download_model.py")
            
            if successful_downloads > 0:
                print(f"\n🎉 Model download completed!")
                print(f"📁 Trained model saved to: {local_model_dir}")
                
                # Check if we have the essential files
                essential_files = ["model_index.json", "unet", "text_encoder", "vae"]
                missing_essential = []
                for file_name in essential_files:
                    if file_name in failed_downloads:
                        missing_essential.append(file_name)
                
                if missing_essential:
                    print(f"\n⚠️  Warning: Missing essential files: {missing_essential}")
                    print("   The model may not work properly without these files.")
                    print("   You can retry downloading them later using the download script.")
                else:
                    print("✅ All essential model files downloaded successfully!")
                
                print(f"🚀 You can now run inference with: python inference/generate_images.py")
            else:
                print("\n❌ No files were downloaded successfully.")
                print("   You can try downloading later using: python train/download_model.py")
        else:
            print("\n🎉 Training completed successfully!")
            print("📥 The trained model is available on Modal and can be downloaded using:")
            print("   python train/download_model.py")
            print("🚀 You can also run inference directly on Modal with: python inference/generate_images.py")
    else:
        print("❌ Training failed. Check the logs above for details.")

if __name__ == "__main__":
    # Example usage:
    # To train without downloading: main()
    # To train and download: main(download_model=True)
    main()
