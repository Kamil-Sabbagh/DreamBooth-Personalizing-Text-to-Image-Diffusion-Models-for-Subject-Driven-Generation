import modal

app = modal.App("dreambooth-eval")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        "diffusers>=0.35.0",
        "transformers==4.56.1",
        "open-clip-torch",
        "pillow",
    ])
)

# Volumes: trained model and eval output
vol_models = modal.Volume.from_name("dreambooth-models", create_if_missing=False)  # Your trained model
vol_eval = modal.Volume.from_name("dreambooth-eval-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60*60,
    volumes={
        "/models": vol_models,
        "/eval": vol_eval,
    },
)
def run_eval():
    import os, re, csv, torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline
    import open_clip

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read prompts from prompts.txt file
    prompts = []
    try:
        with open("/models/prompts.txt", "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"📝 Loaded {len(prompts)} prompts from prompts.txt")
    except FileNotFoundError:
        print("⚠️  prompts.txt not found, using default prompts")
        prompts = [
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
    
    # Ensure we have exactly 10 prompts
    if len(prompts) < 10:
        # Pad with default prompts if needed
        default_prompts = [
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
            prompts.append(default_prompts[len(prompts)])
    elif len(prompts) > 10:
        # Truncate to 10 prompts
        prompts = prompts[:10]
    seeds = list(range(10))

    out_root = "/eval/results"
    os.makedirs(out_root, exist_ok=True)
    # Compare original Stable Diffusion vs your trained model
    variants = {
        "base": "runwayml/stable-diffusion-v1-5",  # Original Stable Diffusion model
        "trained": "/models/trained-model",  # Your fine-tuned model
    }

    # CLIP model
    clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def clip_image_emb(img: Image.Image):
        with torch.no_grad():
            x = clip_preprocess(img).unsqueeze(0).to(device)
            e = clip_model.encode_image(x).float()
            return e / e.norm(dim=-1, keepdim=True)

    def clip_text_emb(text: str):
        with torch.no_grad():
            xt = tokenizer([text])
            xt = {k: v.to(device) for k, v in xt.items()} if isinstance(xt, dict) else xt.to(device)
            e = clip_model.encode_text(xt).float()
            return e / e.norm(dim=-1, keepdim=True)

    # Load pipelines
    pipes = {}
    for name, path in variants.items():
        pipe = StableDiffusionPipeline.from_pretrained(path, safety_checker=None, requires_safety_checker=False)
        pipe = pipe.to(device)
        pipes[name] = pipe

    def slugify(s: str):
        return re.sub(r'[^a-z0-9]+','_', s.lower()).strip('_')

    rows = []
    for prompt in prompts:
        text_emb = clip_text_emb(prompt)
        pslug = slugify(prompt)
        # dirs
        for v in variants:
            os.makedirs(os.path.join(out_root, "images", v), exist_ok=True)
        os.makedirs(os.path.join(out_root, "grids"), exist_ok=True)

        # generate for each variant and seed
        images_by_variant = {v: [] for v in variants}
        for seed in seeds:
            for vname, pipe in pipes.items():
                gen = torch.Generator(device=device).manual_seed(seed)
                img = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=7.5, height=512, width=512, generator=gen).images[0]
                fp = os.path.join(out_root, "images", vname, f"{pslug}__seed{seed}.png")
                img.save(fp)
                images_by_variant[vname].append((seed, img, fp))

        # metrics per seed using base vs trained model image-image as identity proxy, and text-image adherence
        for seed in seeds:
            base_img = [im for s,im,_ in images_by_variant["base"] if s==seed][0]
            emb_base = clip_image_emb(base_img)
            pa_base = float((text_emb @ emb_base.T).squeeze().clamp(-1,1).cpu())
            
            # Compare trained model vs base
            trained_img = [im for s,im,_ in images_by_variant["trained"] if s==seed][0]
            emb_trained = clip_image_emb(trained_img)
            ident = float((emb_base @ emb_trained.T).squeeze().clamp(-1,1).cpu())
            pa_trained = float((text_emb @ emb_trained.T).squeeze().clamp(-1,1).cpu())
            rows.append({
                "prompt": prompt,
                "seed": seed,
                "variant": "trained",
                "identity_proxy_cos_vs_base": ident,
                "clip_textimg_base": pa_base,
                "clip_textimg_variant": pa_trained,
            })

        # grid (first 2 seeds): columns=base vs trained, rows=seeds
        from PIL import Image as PILImage
        grid_seeds = seeds[:2]
        first_img = images_by_variant["base"][0][1]
        w,h = first_img.size
        grid = PILImage.new('RGB', (w*2, h*len(grid_seeds)))  # 2 columns: base and trained
        for r,seed in enumerate(grid_seeds):
            for c,vname in enumerate(["base","trained"]):
                img = [im for s,im,_ in images_by_variant[vname] if s==seed][0]
                grid.paste(img, (c*w, r*h))
        grid.save(os.path.join(out_root, "grids", f"{pslug}.png"))

    # write CSV
    with open(os.path.join(out_root, "summary.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

@app.local_entrypoint()
def main():
    import os
    import subprocess
    
    print("Starting Modal evaluation (base vs your trained model)...")
    
    # Upload prompts.txt to Modal volume
    if os.path.exists("prompts.txt"):
        print("📤 Uploading prompts.txt to Modal volume...")
        try:
            result = subprocess.run([
                "modal", "volume", "put", "dreambooth-models", 
                "prompts.txt", "/prompts.txt"
            ], capture_output=True, text=True, check=True)
            print("✅ prompts.txt uploaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to upload prompts.txt: {e.stderr}")
            print("🔄 Using default prompts instead")
    else:
        print("⚠️  prompts.txt not found, using default prompts")
    
    run_eval.remote()
    print("✅ Evaluation complete! Downloading results...")
    
    # Create local directory for results
    local_results_dir = "./eval_results"
    os.makedirs(local_results_dir, exist_ok=True)
    
    # Download results from Modal volume
    try:
        result = subprocess.run([
            "modal", "volume", "get", "dreambooth-eval-results", "/results", local_results_dir
        ], capture_output=True, text=True, check=True)
        print(f"✅ Results downloaded to: {local_results_dir}")
        print("📊 Check the following files:")
        print(f"   - {local_results_dir}/summary.csv (metrics)")
        print(f"   - {local_results_dir}/grids/ (comparison grids)")
        print(f"   - {local_results_dir}/images/ (individual images)")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading results: {e}")
        print("📥 You can manually download with:")
        print("   modal volume get dreambooth-eval-results /results ./eval_results")