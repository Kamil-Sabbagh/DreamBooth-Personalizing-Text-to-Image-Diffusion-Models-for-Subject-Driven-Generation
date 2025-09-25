#!/usr/bin/env python3
"""
Script to continue DreamBooth training from early checkpoint to full training
"""

import subprocess
import sys

def main():
    print("🔄 Continuing DreamBooth training from early checkpoint...")
    
    try:
        # Run the continue training function on Modal
        result = subprocess.run([
            "modal", "run", "train/train_dreambooth.py::continue_training"
        ], check=True)
        
        print("✅ Training continuation completed!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to continue training: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training continuation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
