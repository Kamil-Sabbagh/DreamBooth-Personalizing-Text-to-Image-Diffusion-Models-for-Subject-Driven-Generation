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

# Volumes: trained models and eval output
vol_overfit = modal.Volume.from_name("dreambooth-models", create_if_missing=False)   # /trained-model
vol_improved = modal.Volume.from_name("improved-dreambooth-models", create_if_missing=False)  # /improved-trained-model, /improved-trained-model-v2
vol_eval = modal.Volume.from_name("dreambooth-eval-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60*60,
    volumes={
        "/models_overfit": vol_overfit,
        "/models_improved": vol_improved,
        "/eval": vol_eval,
    },
)
def run_eval():
    import os, re, csv, torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline
    import open_clip

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
        'a portrait photo of sks dog',
        'sks dog wearing sunglasses, street photo',
        'sks dog in front of the Eiffel Tower',
        'a watercolor painting of sks dog',
        'sks dog sitting on a couch',
        'sks dog running in a park',
        'sks dog on the beach at sunset',
        'studio portrait of sks dog, soft lighting',
        'sks dog with a red bandana',
        'sks dog next to a blue car',
    ]
    seeds = list(range(10))

    out_root = "/eval/results"
    os.makedirs(out_root, exist_ok=True)
    variants = {
        "base": "runwayml/stable-diffusion-v1-5",
        "overfit": "/models_overfit/trained-model",
        "underfit": "/models_improved/improved-trained-model",
        "balanced": "/models_improved/improved-trained-model-v2",
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

        # metrics per seed using base vs others image-image as identity proxy, and text-image adherence
        for seed in seeds:
            base_img = [im for s,im,_ in images_by_variant["base"] if s==seed][0]
            emb_base = clip_image_emb(base_img)
            pa_base = float((text_emb @ emb_base.T).squeeze().clamp(-1,1).cpu())
            for vname in ["overfit","underfit","balanced"]:
                v_img = [im for s,im,_ in images_by_variant[vname] if s==seed][0]
                emb_v = clip_image_emb(v_img)
                ident = float((emb_base @ emb_v.T).squeeze().clamp(-1,1).cpu())
                pa_v = float((text_emb @ emb_v.T).squeeze().clamp(-1,1).cpu())
                rows.append({
                    "prompt": prompt,
                    "seed": seed,
                    "variant": vname,
                    "identity_proxy_cos_vs_base": ident,
                    "clip_textimg_base": pa_base,
                    "clip_textimg_variant": pa_v,
                })

        # grid (first 2 seeds): columns=variants, rows=seeds
        from PIL import Image as PILImage
        grid_seeds = seeds[:2]
        first_img = images_by_variant["base"][0][1]
        w,h = first_img.size
        grid = PILImage.new('RGB', (w*4, h*len(grid_seeds)))
        for r,seed in enumerate(grid_seeds):
            for c,vname in enumerate(["base","overfit","underfit","balanced"]):
                img = [im for s,im,_ in images_by_variant[vname] if s==seed][0]
                grid.paste(img, (c*w, r*h))
        grid.save(os.path.join(out_root, "grids", f"{pslug}.png"))

    # write CSV
    with open(os.path.join(out_root, "summary.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

@app.local_entrypoint()
def main():
    print("Starting Modal evaluation (base vs overfit vs underfit vs balanced)...")
    run_eval.remote()
    print("Submitted. Outputs will be in volume 'dreambooth-eval-results' under /results")