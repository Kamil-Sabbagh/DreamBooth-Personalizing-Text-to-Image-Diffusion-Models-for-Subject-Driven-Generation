# DreamBooth (SD 1.5, Diffusers) with Prior Preservation — Reproduction

This repo reproduces DreamBooth fine-tuning on Stable Diffusion 1.5 using Hugging Face Diffusers, with class prior preservation and a rare identifier token (e.g., `sks`). It includes Modal GPU scripts, inference, evaluation, and a short report.

## Contents
- `train/`: Modal training script (`improved_modal_dreambooth.py`)
- `inference/`: simple test script to load and sample the trained model (`test_model_improved.py`)
- `eval/`: Modal evaluation script + aggregated metrics (`metrics_summary.csv`)
- `artifacts/`: qualitative grids with a reference column
- `report/`: LaTeX report (`main.tex`) with figures and tables
- `configs/`: (optional) place accelerate configs here if needed

## Environment
- Python 3.10 (Modal image)
- Pinned packages (Modal image):
  - torch==2.8.0, torchvision==0.23.0, torchaudio==2.8.0
  - diffusers==0.35.1, transformers==4.56.1, accelerate==1.10.1, peft==0.17.1
  - open-clip-torch, pillow

## Data
- Instance images: 4–8 photos of your subject in `dog/` (local), used to compute identity metrics.
- Class prior images are generated during training by the script.

## Training on Modal
1) Activate your Modal profile and login.
2) Edit hyperparameters in `train/improved_modal_dreambooth.py` if desired (identifier token, prior loss weight, steps).
3) Launch training on an A100-40GB:
```bash
modal run train/improved_modal_dreambooth.py
```
The script writes the trained models to Modal volumes. Use `modal volume ls/get` to download to local.

## Inference 
```bash
python inference/test_model_improved.py
```
Edit `model_path` inside the script to point to your downloaded model dir (e.g., `./improved-trained-model-v2`).

## Evaluation
- Remote generation (base, overfit, underfit, balanced):
```bash
modal run eval/modal_eval.py
```
Outputs land in Modal volume `dreambooth-eval-results` under `/results`. Download with `modal volume get dreambooth-eval-results /results ...`.

- Aggregated means: `eval/metrics_summary.csv` (already included)

## Report
- PDF: `report/Reproducing_DreamBooth_with_Prior_Preservation_on_Stable_Diffusion.pdf`
- (Legacy) LaTeX sources remain in `report/` if you want to edit and rebuild.

## Results (quick look)

Qualitative grids (reference, base, overfit, underfit, balanced). A few examples:

![Portrait](artifacts/grids_with_ref/sks_dog_on_the_beach_at_sunset__seed0.png)
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

## References
- DreamBooth: https://arxiv.org/abs/2208.12242
- Diffusers DreamBooth tutorial: https://huggingface.co/docs/diffusers/en/training/dreambooth
- Project page: https://dreambooth.github.io/
