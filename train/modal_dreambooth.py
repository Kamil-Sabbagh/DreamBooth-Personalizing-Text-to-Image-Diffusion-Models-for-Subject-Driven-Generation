# modal_dreambooth.py
import modal
import os
from pathlib import Path

"""
Deprecated: superseded by train/improved_modal_dreambooth.py
"""

# Create Modal app
app = modal.App("dreambooth-training-deprecated")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
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
        "xformers",  # Let pip resolve compatible version
    ])
    # No need to install diffusers in development mode since we're using stable version
    .env({
        "HF_HOME": "/models/hf_cache",
        "TRANSFORMERS_CACHE": "/models/hf_cache",
    })
)

# Create a volume for persistent storage
volume = modal.Volume.from_name("dreambooth-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",  # Fixed: Use string instead of modal.gpu.A100()
    volumes={"/models": volume},
    timeout=3600,  # 1 hour timeout
)
def train_dreambooth(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    instance_prompt: str = "a photo of sks dog",
    resolution: int = 512,
    train_batch_size: int = 1,
    learning_rate: float = 5e-6,
    max_train_steps: int = 400,
    output_dir: str = "/models/trained-model"
):
    """Train DreamBooth model on Modal with A100 GPU"""
    
    import subprocess
    import sys
    from huggingface_hub import snapshot_download
    
    # Download training images
    print("Downloading training images...")
    import shutil
    import os
    
    local_dir = "/tmp/dog"
    snapshot_download(
        "diffusers/dog-example",
        local_dir=local_dir, 
        repo_type="dataset",
        ignore_patterns=".gitattributes",
    )
    
    # Clean up any cache directories that might interfere with training
    cache_dir = os.path.join(local_dir, ".cache")
    if os.path.exists(cache_dir):
        print(f"Removing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # List the actual image files
    image_files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} training images: {image_files}")
    
    # Configure accelerate for A100
    print("Configuring accelerate for A100...")
    accelerate_config = """
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
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
    
    with open("/tmp/accelerate_config.yaml", "w") as f:
        f.write(accelerate_config)
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        "ACCELERATE_CONFIG_FILE": "/tmp/accelerate_config.yaml",
        "MODEL_NAME": model_name,
        "INSTANCE_DIR": local_dir,
        "OUTPUT_DIR": output_dir,
    })
    
    # Download the training script from diffusers repository
    print("Downloading training script...")
    import requests
    script_url = "https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/examples/dreambooth/train_dreambooth.py"
    script_path = "/tmp/train_dreambooth.py"
    
    response = requests.get(script_url)
    with open(script_path, "w") as f:
        f.write(response.text)
    
    # Run training command
    cmd = [
        "accelerate", "launch", script_path,
        "--pretrained_model_name_or_path", model_name,
        "--instance_data_dir", local_dir,
        "--output_dir", output_dir,
        "--instance_prompt", instance_prompt,
        "--resolution", str(resolution),
        "--train_batch_size", str(train_batch_size),
        "--gradient_accumulation_steps", "1",
        "--learning_rate", str(learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(max_train_steps),
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
    ]
    
    print(f"Running training command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("STDOUT:", result.stdout)
        
        # Commit the trained model to the volume
        volume.commit()
        print(f"Model saved to {output_dir}")
        
        return {"status": "success", "output_dir": output_dir}
        
    except subprocess.CalledProcessError as e:
        print("Training failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return {"status": "error", "error": str(e)}

@app.local_entrypoint()
def main():
    """Local entrypoint to run the training"""
    print("Starting DreamBooth training on Modal with A100...")
    
    result = train_dreambooth.remote(
        model_name="runwayml/stable-diffusion-v1-5",
        instance_prompt="a photo of sks dog",
        resolution=512,  # Can use full resolution on A100!
        train_batch_size=1,
        learning_rate=5e-6,
        max_train_steps=400,
    )
    
    print("Training result:", result)

if __name__ == "__main__":
    main()