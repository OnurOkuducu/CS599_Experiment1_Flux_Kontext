import os
import json
import torch
from diffusers import FluxKontextPipeline

# === List of JSON files ===
JSON_FILES = [
    "entity_reasoning",
    "idiom_interpretation",
    "scientific_reasoning",
    "textual_image_design",
]

# === Load the model ===
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

def sanitize_filename(prompt: str) -> str:
    """Convert prompt text to a safe short filename."""
    safe_name = "".join(c for c in prompt if c.isalnum() or c in (" ", "_")).rstrip()
    safe_name = "_".join(safe_name.lower().split())[:80]
    return safe_name

# === Main Loop ===
for file_name in JSON_FILES:
    # Determine output directory based on file name
    json_path = f'/restricted/projectnb/cs599dg/onur/experiment1/T2I-ReasonBench/prompts/{file_name}.json'
    output_dir = f"{file_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {json_path}")
    print(f"Output directory: {output_dir}\n")

    with open(json_path, "r") as f:
        data = json.load(f)

    for idx, item in enumerate(data[:50]):
        prompt = item["prompt"]

        img_id = str(item.get("id")).zfill(4)
        out_path = os.path.join(output_dir, f"{img_id}.png")

        # Skip if already exists (so it can resume)
        if os.path.exists(out_path):
            print(f"⏩ Skipping {out_path} (already exists)")
            continue

        print(f"[{idx+1}/{len(data)}] {prompt}")

        try:
            result = pipe(prompt=prompt)
            image = result.images[0]
            image.save(out_path)
            print(f"✅ Saved {out_path}")
        except Exception as e:
            print(f"❌ Error generating image for prompt {idx}: {e}")
            continue
