# DreamBooth (SD 1.5, Diffusers) with Prior Preservation — Reproduction

This repo reproduces DreamBooth fine-tuning on Stable Diffusion 1.5 using Hugging Face Diffusers, with class prior preservation and a rare identifier token (e.g., `sks`). It includes Modal GPU scripts, inference, evaluation, and a short report.

## Contents
- `train/`: Modal training script (`train_dreambooth.py`) + model download script (`download_model.py`)
- `inference/`: Image generation script (`generate_images.py`)
- `eval/`: Model evaluation script (`evaluate_model.py`) + aggregated metrics (`metrics_summary.csv`)
- `artifacts/`: Qualitative comparison grids
- `target/`: Your training images (3-5 photos of your subject)
- `prompts.txt`: Custom prompts for image generation

## Why Modal?

This project uses [Modal](https://modal.com) for cloud-based training and evaluation to ensure **reproducible research** and **cost-effective experimentation**.

### The Problem with Local Training
Reproducing research results, especially in machine learning, often fails due to:
- **Hardware differences**: Different GPUs, memory, and compute capabilities
- **Software inconsistencies**: Varying package versions, CUDA versions, and system configurations  
- **Time constraints**: Training can take hours or days on consumer hardware
- **Resource limitations**: High-end GPUs (A100, H100) are expensive and often unavailable

### Modal's Solution
Modal provides **serverless AI infrastructure** that solves these challenges:

- **💰 Cost-effective**: Train DreamBooth models for **less than $1** per experiment
- **⚡ Fast**: Complete training in **under 10 minutes** on A100 GPUs
- **🔄 Reproducible**: Identical hardware and software environment every time
- **📈 Scalable**: Access to state-of-the-art GPUs (A100, H100) without upfront costs
- **🛠️ Zero config**: Define requirements in Python, no infrastructure management needed

### Key Benefits for This Project
- **Consistent results**: Same GPU type and software stack for all experiments
- **Rapid iteration**: Test different hyperparameters quickly and affordably
- **No setup overhead**: No need to install CUDA, manage dependencies, or configure environments
- **Professional infrastructure**: Enterprise-grade security and reliability

Learn more at [modal.com](https://modal.com)

## Environment
- Python 3.10 (Modal image)
- Pinned packages (Modal image):
  - torch==2.8.0, torchvision==0.23.0, torchaudio==2.8.0
  - diffusers==0.35.1, transformers==4.56.1, accelerate==1.10.1, peft==0.17.1
  - open-clip-torch, pillow

## Data Setup

### 1. Prepare Your Target Images
Place 3-5 high-quality photos of your subject in the `target/` folder:
- Use clear, well-lit photos with good resolution
- Include different angles, lighting, and backgrounds  
- Supported formats: JPG, PNG, WebP
- The script will automatically resize them to 512x512 pixels

### 2. Configure Training Prompts
**IMPORTANT**: Edit `training_config.txt` to match your subject:

```bash
# Edit this file before training
nano training_config.txt
```

**Key settings to configure:**
- `INSTANCE_PROMPT`: The special identifier for your subject (e.g., `a photo of sks dog`)
- `CLASS_PROMPT`: The general category of your subject (e.g., `a photo of dog`)
- `MAX_TRAIN_STEPS`: Number of training steps (1000 recommended)
- `LEARNING_RATE`: Learning rate (2e-6 recommended)

**Example configurations:**
```bash
# For a backpack
INSTANCE_PROMPT=a photo of sks backpack
CLASS_PROMPT=a photo of backpack

# For a person
INSTANCE_PROMPT=a photo of sks person
CLASS_PROMPT=a photo of person

# For a specific object
INSTANCE_PROMPT=a photo of sks toy
CLASS_PROMPT=a photo of toy
```

## Training Workflow Options

You have two flexible options for training and using your DreamBooth model:

### Option 1: Train and Download (Traditional)
Train your model and automatically download it to your local machine:
```bash
# Train with automatic download
modal run train/train_dreambooth.py
# Note: Modify main() call in train_dreambooth.py to main(download_model=True)
```

### Option 2: Train on Modal, Download Later (Recommended)
Train your model on Modal and download it separately when needed:
```bash
# 1. Train your DreamBooth model (faster, no download)
modal run train/train_dreambooth.py

# 2. Download the trained model when ready
python train/download_model.py

# 3. Generate images with your trained model  
modal run inference/modal_generate_images.py

# 4. Evaluate your model against the base model
modal run eval/evaluate_model.py
```

**Benefits of Option 2:**
- ⚡ **Faster training**: No time spent downloading large model files
- 💰 **Cost-effective**: Only pay for training time, download when convenient
- 🔄 **Flexible**: Run inference directly on Modal or download locally
- 📁 **Custom locations**: Download to any directory with `--output-dir`

## Complete Pipeline (3 Commands)

### Prerequisites
1) **Prepare your images**: Place 3-5 photos of your subject in the `target/` folder
2) **Configure training prompts**: Edit `training_config.txt` with your subject details
3) **Configure generation prompts**: Edit `prompts.txt` with your custom prompts
4) **Activate your Modal profile** and login

### Run the Complete Pipeline
```bash
# 1. Train your DreamBooth model
modal run train/train_dreambooth.py

# 2. Generate images with your trained model  
modal run inference/modal_generate_images.py

# 3. Evaluate your model against the base model
modal run eval/evaluate_model.py
```

**That's it!** 🎉 All three scripts automatically:
- Upload your `target/` images and `prompts.txt` to Modal
- Use your custom configuration from `training_config.txt`
- Download results to your local machine
- Handle all file management automatically


### 3. Configure Generation Prompts
Edit `prompts.txt` with your custom prompts for image generation:
- Use the same special identifier from your training config
- Example: `a portrait photo of sks backpack` (where `sks` represents your specific backpack)
- The model will learn to associate your identifier with your subject

## Model Download

If you trained your model without downloading (Option 2), you can download it later using the dedicated download script:

### Download Options
```bash
# Download to default location (./trained-model)
python train/download_model.py

# Download to custom location
python train/download_model.py --output-dir /path/to/custom/location

# Or run directly (if executable)
./train/download_model.py
```

### Download Features
- 📥 **One-by-one downloads**: More reliable than bulk downloads
- 📊 **Progress tracking**: See download status for each file
- 🛡️ **Error handling**: Retry failed downloads individually
- 📁 **Flexible locations**: Save to any directory
- ✅ **Validation**: Check for essential model files

## Image Generation
The generation script will automatically read prompts from the `prompts.txt` file and generate 10 images.

1) **Customize prompts** (optional): Edit the `prompts.txt` file with your desired prompts using the special identifier `sks`
   - Example prompts are already provided in the file
   - Use `sks` to refer to your trained subject
   - If the file is empty or missing, default prompts will be used

2) **Generate images**:
```bash
# Generate on Modal (recommended - uses trained model directly)
modal run inference/modal_generate_images.py

# Or generate locally (requires downloaded model)
python inference/generate_images.py
```
Edit `model_path` inside the script to point to your downloaded model dir (e.g., `./trained-model`).

## Evaluation
The evaluation script compares the original Stable Diffusion model with your fine-tuned model to measure:
- **Subject fidelity**: How well the model captures your specific subject
- **Prompt adherence**: How well the model follows the given prompts  
- **Diversity**: How varied the generated images are

Run evaluation:
```bash
modal run eval/evaluate_model.py
```
Outputs land in Modal volume `dreambooth-eval-results` under `/results`. Download with `modal volume get dreambooth-eval-results /results ...`.

- Aggregated metrics: `eval/metrics_summary.csv` (already included)

## Report
- PDF: `report/Reproducing_DreamBooth_with_Prior_Preservation_on_Stable_Diffusion.pdf`
- (Legacy) LaTeX sources remain in `report/` if you want to edit and rebuild.

## Results (quick look)

Qualitative grids (reference, base, overfit, underfit, balanced). A few examples:

![Portrait](artifacts/grids_with_ref/sks_dog_running_in_a_park__seed0.png)
![Watercolor](artifacts/grids_with_ref/a_watercolor_painting_of_sks_dog.png)
![Eiffel](artifacts/grids_with_ref/sks_dog_in_front_of_the_eiffel_tower.png)

### Summary metrics (means)

From `eval/metrics_summary.csv`:

| Variant   | Subject fidelity (max) | Subject fidelity (mean) | Diversity (pairwise cos) | Prompt adherence |
|-----------|-------------------------|--------------------------|--------------------------|------------------|
| Overfit   | 0.881                   | 0.870                    | 0.915                    | 0.275            |
| Underfit  | 0.784                   | 0.773                    | 0.880                    | 0.296            |
| Balanced  | 0.855                   | 0.843                    | 0.903                    | 0.288            |

Interpretation: overfit maximizes identity but weakens prompt adherence; underfit improves adherence but loses identity; balanced offers the best trade-off.

## Reproducibility Checklist
- Fixed prompts/seeds (see `eval/modal_eval.py`)
- Pinned package versions
- Modal GPU type set to `"A100-40GB"`
- No secrets or tokens committed

### Notes on Modal usage
- The training implementation clones all required code inside the Modal container, keeping this repo light and robust.
- Each training run completed in ~5–10 minutes on A100-40GB and cost about $0.7–$0.9 per model (as observed), thanks to mixed precision, gradient checkpointing, and efficient regularization settings.

## Pretrained Weights

Weights are available for evaluation in Google Drive:

- Overfit: `trained-model` (DreamBooth without strong regularization)
- Underfit: `improved-trained-model` (earlier run with too-strong regularization)
- Balanced: `improved-trained-model-v2` (final recommended)

Drive folder: https://drive.google.com/drive/folders/1Wt3pRJtkIsD8g0rD4uRuoE-BLd_qo-7J?usp=share_link

Usage example (after download/unzip):
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("./improved-trained-model-v2", safety_checker=None, requires_safety_checker=False)
```

## Experiment configurations

- Overfit (`trained-model`, local MPS)
  - pretrained: `runwayml/stable-diffusion-v1-5`
  - instance_prompt: "a photo of sks dog"
  - resolution: 256
  - train_batch_size: 1; gradient_accumulation_steps: 1
  - learning_rate: 5e-6; lr_scheduler: constant; lr_warmup_steps: 0
  - max_train_steps: 400
  - mixed_precision: none (fp32)
  - gradient_checkpointing: true
  - with_prior_preservation: false; train_text_encoder: false

- Underfit (`improved-trained-model`, Modal A100)
  - Trained with prior preservation and text encoder enabled.
  - Note: exact flags for this early run weren’t logged in-repo; it used stronger regularization than the balanced run (higher prior regularization effect).
  - Typical settings used in this run family: resolution 512, train_batch_size 1, gradient_accumulation_steps 1, mixed_precision bf16, with_prior_preservation true, num_class_images 200, train_text_encoder true.

- Balanced (`improved-trained-model-v2`, Modal A100)
  - pretrained: `runwayml/stable-diffusion-v1-5`
  - instance_prompt: "a photo of sks dog"; class_prompt: "a photo of dog"
  - resolution: 512
  - train_batch_size: 1; gradient_accumulation_steps: 1
  - learning_rate: 1e-6; lr_scheduler: constant; lr_warmup_steps: 0
  - max_train_steps: 800
  - mixed_precision: bf16
  - gradient_checkpointing: true
  - with_prior_preservation: true; prior_loss_weight: 0.6; num_class_images: 100
  - train_text_encoder: true; seed: 42

## References
- DreamBooth: https://arxiv.org/abs/2208.12242
- Diffusers DreamBooth tutorial: https://huggingface.co/docs/diffusers/en/training/dreambooth
- Project page: https://dreambooth.github.io/
