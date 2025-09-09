# DreamBooth (SD 1.5, Diffusers) with Prior Preservation — Reproduction

This repo reproduces DreamBooth fine-tuning on Stable Diffusion 1.5 using Hugging Face Diffusers, with class prior preservation and a rare identifier token (e.g., `sks`). It includes Modal GPU scripts, inference, evaluation, and a short report.

## Contents
- `train/`: Modal scripts for training (`modal_dreambooth.py`, `improved_modal_dreambooth.py`)
- `inference/`: simple test scripts to load and sample models
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

## Inference (local)
```bash
python inference/test_model.py            # initial model
python inference/test_model_improved.py   # improved/balanced model
```
Edit paths inside the scripts to point to your downloaded model dir.

## Evaluation
- Remote generation: `modal run eval/modal_eval.py` (produces images, grids, `summary.csv` in a Modal volume)
- Download results: use `modal volume get` per folder
- Local extended metrics:
```bash
# produces eval_results/extended_comparison_metrics.csv
python eval/local_extended_metrics.py  # (optional script if you add it)
```
- Aggregated means: `eval/metrics_summary.csv`

## Report
- See `report/main.tex` (3–5 pages). Insert your name/date.
- Compile locally (TeX Live) or Overleaf.

## Reproducibility Checklist
- Fixed prompts/seeds (see `eval/modal_eval.py`)
- Pinned package versions
- Modal GPU type set to `"A100-40GB"`
- No secrets or tokens committed

## References
- DreamBooth: https://arxiv.org/abs/2208.12242
- Diffusers DreamBooth tutorial: https://huggingface.co/docs/diffusers/en/training/dreambooth
- Project page: https://dreambooth.github.io/
