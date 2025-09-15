#!/usr/bin/env python3
"""
Download trained DreamBooth model from Modal volume

Usage:
    # Download to default location (./trained-model)
    python download_model.py
    
    # Download to custom location
    python download_model.py --output-dir /path/to/custom/location
    
    # Or run directly
    ./download_model.py
"""

import subprocess
import os
import sys
from pathlib import Path

def download_trained_model(local_model_dir: str = "./trained-model"):
    """
    Download the trained model files from Modal volume to local directory
    
    Args:
        local_model_dir: Local directory to save the model files
    """
    
    print("📥 Downloading trained DreamBooth model from Modal...")
    
    # Create local directory for the trained model
    os.makedirs(local_model_dir, exist_ok=True)
    print(f"📁 Created directory: {local_model_dir}")
    
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
    
    # Print summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {successful_downloads} files/directories")
    if failed_downloads:
        print(f"   ❌ Failed downloads: {failed_downloads}")
        print("\n📥 You can manually retry failed downloads with:")
        for failed_file in failed_downloads:
            print(f"   modal volume get --force dreambooth-models /trained-model/{failed_file} {local_model_dir}/{failed_file}")
    
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
            print("   You can retry downloading them manually.")
        else:
            print("✅ All essential model files downloaded successfully!")
        
        print(f"🚀 You can now run inference with: python inference/generate_images.py")
    else:
        print("\n❌ No files were downloaded successfully.")
        print("   Make sure the model has been trained and is available on Modal.")
        return False
    
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download trained DreamBooth model from Modal")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./trained-model",
        help="Local directory to save the model files (default: ./trained-model)"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    output_dir = os.path.abspath(args.output_dir)
    
    print("🚀 DreamBooth Model Downloader")
    print(f"📁 Output directory: {output_dir}")
    
    success = download_trained_model(output_dir)
    
    if success:
        print("\n✅ Download completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
