import modal

app = modal.App("dreambooth-eval")

# ---- Image / deps -----------------------------------------------------------
# Use CUDA wheels and compatible versions. Adjust cu121->cu122 if your account uses CUDA 12.2.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        # PyTorch + CUDA wheels (cu121 is broadly available & works on A100)
        "torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Core libs
        "diffusers==0.29.2",
        "transformers==4.41.2",
        "safetensors>=0.4.2",
        "accelerate>=0.30.0",
        "open-clip-torch>=2.24.0",
        "pillow",
        "lpips",
        "scikit-learn",
        "numpy",
        "timm",  # for some vision backends
    )
)

# Volumes: trained model and eval output
vol_models = modal.Volume.from_name("dreambooth-models", create_if_missing=False)
vol_eval = modal.Volume.from_name("dreambooth-eval-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60*60,
    volumes={"/models": vol_models, "/eval": vol_eval},
)
def run_eval(checkpoint_step=None):
    """Run evaluation using a specific checkpoint or final trained model
    
    Args:
        checkpoint_step: Specific checkpoint step to use (e.g., 400, 500, 1000)
                        If None, uses the final trained model
    """
    import os, re, csv, torch
    import numpy as np
    from PIL import Image
    from diffusers import StableDiffusionPipeline
    import open_clip
    import lpips

    from transformers import AutoImageProcessor, AutoModel  # DINOv2

    # ---- Device / perf ------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ---- Prompts ------------------------------------------------------------
    prompts = []
    try:
        with open("/models/prompts.txt", "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"📝 Loaded {len(prompts)} prompts from prompts.txt")
    except FileNotFoundError:
        print("⚠️  prompts.txt not found, using default prompts")
        defaults = [
            'a portrait photo of sks backpack',
            'sks backpack on a hiking trail, outdoor photo',
            'sks backpack in front of the Eiffel Tower',
            'a watercolor painting of sks backpack',
            'sks backpack sitting on a wooden table',
            'sks backpack hanging on a tree branch',
            'sks backpack on the beach at sunset',
            'studio portrait of sks backpack, soft lighting',
            'sks backpack with travel stickers',
            'sks backpack next to a blue car'
        ]
        prompts = defaults
    # Force exactly 10
    if len(prompts) < 10:
        base = [
            'a portrait photo of sks backpack',
            'sks backpack on a hiking trail, outdoor photo',
            'sks backpack in front of the Eiffel Tower',
            'a watercolor painting of sks backpack',
            'sks backpack sitting on a wooden table',
            'sks backpack hanging on a tree branch',
            'sks backpack on the beach at sunset',
            'studio portrait of sks backpack, soft lighting',
            'sks backpack with travel stickers',
            'sks backpack next to a blue car'
        ]
        while len(prompts) < 10:
            prompts.append(base[len(prompts)])
    elif len(prompts) > 10:
        prompts = prompts[:10]
    seeds = list(range(10))

    # Create output directory with checkpoint info
    if checkpoint_step:
        out_root = f"/eval/results_checkpoint_{checkpoint_step}"
    else:
        out_root = "/eval/results_final"
    os.makedirs(out_root, exist_ok=True)

    # ---- Variants -----------------------------------------------------------
    if checkpoint_step:
        print(f"🎯 Evaluating checkpoint from step {checkpoint_step}")
        variants = {
            "base": "runwayml/stable-diffusion-v1-5",
            "trained": f"/models/trained-model/checkpoint-{checkpoint_step}",
        }
    else:
        print("🎯 Evaluating final trained model")
        variants = {
            "base": "runwayml/stable-diffusion-v1-5",
            "trained": "/models/trained-model",
        }

    # Preflight trained model path
    if checkpoint_step:
        # For checkpoints, check if the checkpoint directory exists
        checkpoint_path = variants["trained"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"❌ Checkpoint not found at {checkpoint_path}. "
                f"Expected /models/trained-model/checkpoint-{checkpoint_step} to exist. "
                "Make sure the checkpoint was saved during training."
            )
    else:
        # For final model, check for model_index.json
        trained_index = os.path.join(variants["trained"], "model_index.json")
        if not os.path.exists(trained_index):
            raise FileNotFoundError(
                "❌ Trained model not found at /models/trained-model. "
                "Expected /models/trained-model/model_index.json to exist. "
                "Mount the right volume or copy your fine-tuned weights there."
            )

    # ---- Models for eval ----------------------------------------------------
    print("🔧 Loading evaluation models...")

    # OpenCLIP
    clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # LPIPS
    lpips_model = lpips.LPIPS(net="alex").to(device)

    # DINOv2 (robust, supported in Transformers)
    try:
        dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        dino_model.eval()
        print("✅ DINOv2 model loaded for object identity")
    except Exception as e:
        print(f"⚠️  DINOv2 failed to load: {e}")
        dino_model = None
        dino_processor = None

    print("✅ Evaluation models loaded successfully")

    # ---- Helpers ------------------------------------------------------------
    def clip_image_emb(img: Image.Image):
        with torch.no_grad():
            x = clip_preprocess(img).unsqueeze(0).to(device)
            e = clip_model.encode_image(x).float()
            e = e / e.norm(dim=-1, keepdim=True)
            return e  # torch [1, D]

    def clip_text_emb(text: str):
        with torch.no_grad():
            xt = tokenizer([text])  # LongTensor [1, 77]
            if hasattr(xt, "to"):
                xt = xt.to(device)
            e = clip_model.encode_text(xt).float()
            e = e / e.norm(dim=-1, keepdim=True)
            return e  # torch [1, D]

    def dino_image_emb(img: Image.Image):
        if dino_model is None or dino_processor is None:
            return None
        with torch.no_grad():
            inputs = dino_processor(images=img, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]  # CLS
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats  # torch [1, D]

    def is_face_prompt(prompt: str):
        face_keywords = ['person', 'face', 'portrait', 'headshot', 'man', 'woman', 'boy', 'girl', 'human']
        p = prompt.lower()
        return any(k in p for k in face_keywords)

    def scalar_cos_sim(a_torch, b_torch):
        # expects torch [1, D]
        return float((a_torch @ b_torch.T).squeeze().item())

    def calculate_identity_similarity(img1: Image.Image, img2: Image.Image, prompt: str):
        """Face → CLIP; Object → DINOv2 (fallback to CLIP). Returns scalar."""
        if is_face_prompt(prompt) or (dino_model is None):
            emb1 = clip_image_emb(img1)
            emb2 = clip_image_emb(img2)
            return scalar_cos_sim(emb1, emb2)
        else:
            e1 = dino_image_emb(img1)
            e2 = dino_image_emb(img2)
            if e1 is not None and e2 is not None:
                return scalar_cos_sim(e1, e2)
            # Fallback to CLIP
            emb1 = clip_image_emb(img1)
            emb2 = clip_image_emb(img2)
            return scalar_cos_sim(emb1, emb2)

    def calculate_lpips_distance(img1: Image.Image, img2: Image.Image):
        import torchvision.transforms as T
        to_tensor = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])
        t1 = to_tensor(img1).unsqueeze(0).to(device)
        t2 = to_tensor(img2).unsqueeze(0).to(device)
        with torch.no_grad():
            d = lpips_model(t1, t2)
        return float(d.item())

    # ---- Load SD pipelines (with VRAM-friendly toggles) ---------------------
    pipes = {}
    for name, path in variants.items():
        if name == "trained" and checkpoint_step:
            # For checkpoints, we need to load the base model and then load the checkpoint weights
            print(f"🔄 Loading base model and applying checkpoint-{checkpoint_step} weights...")
            # Load base model first
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Load checkpoint weights
            try:
                # Load UNet from checkpoint
                unet_path = os.path.join(path, "unet")
                if os.path.exists(unet_path):
                    from diffusers import UNet2DConditionModel
                    pipe.unet = UNet2DConditionModel.from_pretrained(
                        unet_path, 
                        torch_dtype=dtype
                    )
                    print(f"✅ Loaded UNet from checkpoint-{checkpoint_step}")
                
                # Load text encoder from checkpoint if it exists
                text_encoder_path = os.path.join(path, "text_encoder")
                if os.path.exists(text_encoder_path):
                    from transformers import CLIPTextModel
                    pipe.text_encoder = CLIPTextModel.from_pretrained(
                        text_encoder_path,
                        torch_dtype=dtype
                    )
                    print(f"✅ Loaded text encoder from checkpoint-{checkpoint_step}")
                
            except Exception as e:
                print(f"⚠️  Error loading checkpoint weights: {e}")
                print("🔄 Falling back to base model...")
        else:
            # Load model normally (base model or final trained model)
            pipe = StableDiffusionPipeline.from_pretrained(
                path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        
        if device == "cuda":
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        pipes[name] = pipe

    def slugify(s: str):
        return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

    # ---- Generate & score ---------------------------------------------------
    rows = []
    for prompt in prompts:
        pslug = slugify(prompt)

        # dirs
        for v in variants:
            os.makedirs(os.path.join(out_root, "images", v), exist_ok=True)
        os.makedirs(os.path.join(out_root, "grids"), exist_ok=True)

        # generate for each variant and seed
        images_by_variant = {v: [] for v in variants}
        for seed in seeds:
            for vname, pipe in pipes.items():
                gen = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                    img = pipe(
                        prompt=prompt,
                        num_inference_steps=25,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=gen,
                    ).images[0]
                fp = os.path.join(out_root, "images", vname, f"{pslug}__seed{seed}.png")
                img.save(fp)
                images_by_variant[vname].append((seed, img, fp))

        # Metrics
        print(f"📊 Calculating metrics for prompt: {prompt[:50]}...")

        # Reference: first 3 trained images (store objects to compare by identity)
        reference_images = [im for s, im, _ in images_by_variant["trained"][:3]]

        for seed in seeds:
            base_img = [im for s, im, _ in images_by_variant["base"] if s == seed][0]
            trained_img = [im for s, im, _ in images_by_variant["trained"] if s == seed][0]

            # 1) Identity similarity (CLIP or DINOv2)
            identity_similarity = calculate_identity_similarity(base_img, trained_img, prompt)

            # 2) Prompt adherence (CLIP text-image)
            text_e = clip_text_emb(prompt)
            emb_base = clip_image_emb(base_img)
            emb_trained = clip_image_emb(trained_img)
            pa_base = scalar_cos_sim(text_e, emb_base)
            pa_trained = scalar_cos_sim(text_e, emb_trained)

            # 3) Diversity (LPIPS across seeds for trained)
            diversity_scores = []
            for other_seed, other_img, _ in images_by_variant["trained"]:
                if other_seed != seed:
                    diversity_scores.append(calculate_lpips_distance(trained_img, other_img))
            avg_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

            # 4) Identity vs reference set (exclude self by object identity)
            identity_vs_ref = []
            for ref_img in reference_images:
                if ref_img is not trained_img:  # compare by object identity
                    identity_vs_ref.append(calculate_identity_similarity(trained_img, ref_img, prompt))
            avg_identity_vs_ref = float(np.mean(identity_vs_ref)) if identity_vs_ref else 0.0

            rows.append({
                "prompt": prompt,
                "seed": seed,
                "variant": "trained",
                "identity_similarity_vs_base": identity_similarity,
                "identity_similarity_vs_reference": avg_identity_vs_ref,
                "prompt_adherence_base": pa_base,
                "prompt_adherence_trained": pa_trained,
                "diversity_lpips": avg_diversity,
                "is_face_prompt": is_face_prompt(prompt),
            })

        # grid (first 2 seeds)
        from PIL import Image as PILImage
        grid_seeds = seeds[:2]
        first_img = images_by_variant["base"][0][1]
        w, h = first_img.size
        grid = PILImage.new('RGB', (w * 2, h * len(grid_seeds)))  # 2 cols: base, trained
        for r, seed in enumerate(grid_seeds):
            for c, vname in enumerate(["base", "trained"]):
                img = [im for s, im, _ in images_by_variant[vname] if s == seed][0]
                grid.paste(img, (c * w, r * h))
        grid.save(os.path.join(out_root, "grids", f"{pslug}.png"))

    # ---- Write CSV + summary ------------------------------------------------
    if not rows:
        print("❌ No results to write; generation may have failed earlier.")
        return

    csv_path = os.path.join(out_root, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    print("\n📊 Comprehensive Evaluation Metrics Summary:")
    print("=" * 80)

    avg_identity_vs_base = float(np.mean([row['identity_similarity_vs_base'] for row in rows]))
    avg_identity_vs_ref = float(np.mean([row['identity_similarity_vs_reference'] for row in rows]))
    avg_pa_base = float(np.mean([row['prompt_adherence_base'] for row in rows]))
    avg_pa_trained = float(np.mean([row['prompt_adherence_trained'] for row in rows]))
    avg_diversity = float(np.mean([row['diversity_lpips'] for row in rows]))

    face_rows = [row for row in rows if row['is_face_prompt']]
    object_rows = [row for row in rows if not row['is_face_prompt']]

    print(f"📊 Total evaluations: {len(rows)}")
    print(f"👤 Face prompts: {len(face_rows)}")
    print(f"🎯 Object prompts: {len(object_rows)}")

    print(f"\n🔍 Identity Metrics:")
    print(f"   📈 Identity vs Base Model: {avg_identity_vs_base:.4f}")
    print(f"   📈 Identity vs Reference Set: {avg_identity_vs_ref:.4f}")

    print(f"\n📝 Prompt Adherence (CLIP Text-Image):")
    print(f"   📊 Base Model: {avg_pa_base:.4f}")
    print(f"   📊 Trained Model: {avg_pa_trained:.4f}")
    print(f"   📈 Improvement: {avg_pa_trained - avg_pa_base:+.4f}")

    print(f"\n🎨 Diversity (LPIPS):")
    print(f"   📊 Average LPIPS Distance: {avg_diversity:.4f}")

    if face_rows:
        face_identity = float(np.mean([row['identity_similarity_vs_base'] for row in face_rows]))
        face_pa = float(np.mean([row['prompt_adherence_trained'] for row in face_rows]))
        print(f"\n👤 Face-Specific Metrics:")
        print(f"   📈 Face Identity (CLIP): {face_identity:.4f}")
        print(f"   📝 Face Prompt Adherence: {face_pa:.4f}")

    if object_rows:
        object_identity = float(np.mean([row['identity_similarity_vs_base'] for row in object_rows]))
        object_pa = float(np.mean([row['prompt_adherence_trained'] for row in object_rows]))
        print(f"\n🎯 Object-Specific Metrics:")
        print(f"   📈 Object Identity (DINOv2): {object_identity:.4f}")
        print(f"   📝 Object Prompt Adherence: {object_pa:.4f}")

    # Per-prompt
    print("\n📋 Per-Prompt Results:")
    print("-" * 80)
    for prompt in sorted(set(row['prompt'] for row in rows)):
        prompt_rows = [row for row in rows if row['prompt'] == prompt]
        avg_ident = float(np.mean([row['identity_similarity_vs_base'] for row in prompt_rows]))
        avg_pa = float(np.mean([row['prompt_adherence_trained'] for row in prompt_rows]))
        avg_div = float(np.mean([row['diversity_lpips'] for row in prompt_rows]))
        prompt_type = "👤 Face" if prompt_rows[0]['is_face_prompt'] else "🎯 Object"
        print(f"   {prompt_type} '{prompt[:45]}{'...' if len(prompt) > 45 else ''}'")
        print(f"      Identity: {avg_ident:.4f}, Prompt Adherence: {avg_pa:.4f}, Diversity: {avg_div:.4f}")

    print(f"\n💾 Detailed metrics saved to: {csv_path}")
    print("📊 Metrics include: Identity (CLIP/DINOv2), Prompt Adherence (CLIP), Diversity (LPIPS)")

@app.local_entrypoint()
def main(checkpoint_step: int = None):
    """Main entrypoint for evaluation with optional checkpoint support
    
    Args:
        checkpoint_step: Specific checkpoint step to evaluate (e.g., 400, 500, 1000)
                        If None, evaluates the final trained model
    """
    import os, subprocess, sys

    # Parse command line arguments
    if len(sys.argv) > 1 and "--checkpoint-step" in sys.argv:
        try:
            idx = sys.argv.index("--checkpoint-step")
            checkpoint_step = int(sys.argv[idx + 1])
            print(f"🎯 Evaluating checkpoint from step {checkpoint_step}")
        except (ValueError, IndexError):
            print("⚠️  Invalid checkpoint step. Using final trained model.")
            checkpoint_step = None
    elif checkpoint_step is None:
        print("🎯 Evaluating final trained model")

    # Upload prompts.txt to the models volume (visible as /models/prompts.txt in the container)
    if os.path.exists("prompts.txt"):
        print("📤 Uploading prompts.txt to Modal volume...")
        try:
            subprocess.run(
                ["modal", "volume", "put", "--force", "dreambooth-models", "prompts.txt", "/prompts.txt"],
                capture_output=True, text=True, check=True
            )
            print("✅ prompts.txt uploaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to upload prompts.txt: {e.stderr}\n🔄 Using default prompts instead")
    else:
        print("⚠️  prompts.txt not found locally, using defaults")

    # Run eval (blocks until completion)
    run_eval.remote(checkpoint_step)
    print("✅ Evaluation finished! Downloading results...")

    # Set local results directory based on checkpoint
    if checkpoint_step:
        local_results_dir = f"./eval_results_checkpoint_{checkpoint_step}"
        volume_path = f"/results_checkpoint_{checkpoint_step}"
    else:
        local_results_dir = "./eval_results_final"
        volume_path = "/results_final"
    
    os.makedirs(local_results_dir, exist_ok=True)

    try:
        subprocess.run(
            ["modal", "volume", "get", "dreambooth-eval-results", volume_path, local_results_dir],
            capture_output=True, text=True, check=True
        )
        print(f"✅ Results downloaded to: {local_results_dir}")
        
        # Find the CSV file in the downloaded results
        csv_file = None
        for root, dirs, files in os.walk(local_results_dir):
            if "summary.csv" in files:
                csv_file = os.path.join(root, "summary.csv")
                break
        
        if csv_file and os.path.exists(csv_file):
            print(f"\n📊 Metrics Summary from {csv_file}:\n" + "="*60)
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                print(df.head().to_string(index=False))
            except Exception as e:
                print(f"⚠️  Could not pretty-print CSV ({e}); showing raw text:")
                with open(csv_file, "r") as f:
                    print(f.read())
        else:
            print("⚠️  summary.csv not found in downloaded results")

        print("\n📂 Check:")
        print(f"   - {local_results_dir}/summary.csv")
        print(f"   - {local_results_dir}/grids/")
        print(f"   - {local_results_dir}/images/")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading results: {e}")
        print(f"📥 Manual fallback:\n   modal volume get dreambooth-eval-results {volume_path} {local_results_dir}")


if __name__ == "__main__":
    # Example usage:
    # For final trained model: python evaluate_model(speicific).py
    # For checkpoint 400: python evaluate_model(speicific).py --checkpoint-step 400
    # For checkpoint 500: python evaluate_model(speicific).py --checkpoint-step 500
    
    import sys
    
    checkpoint_step = None
    if len(sys.argv) > 1 and "--checkpoint-step" in sys.argv:
        try:
            idx = sys.argv.index("--checkpoint-step")
            checkpoint_step = int(sys.argv[idx + 1])
            print(f"🎯 Using checkpoint from step {checkpoint_step}")
        except (ValueError, IndexError):
            print("⚠️  Invalid checkpoint step. Using final trained model.")
    
    main(checkpoint_step)