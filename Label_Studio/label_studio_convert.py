import json
import os
from pathlib import Path
from PIL import Image

# INPUT and OUTPUT folders
image_folder = Path("E:/SIA_works/PosterDatabase/PosterDataset")
jsonl_folder = Path("E:/SIA_works/PosterDatabase/GCV_outputs_Unfiltered_MAY12/jsonl")
output_folder = Path("E:/SIA_works/PosterDatabase/outputs/labelstudio_github_linked")
output_folder.mkdir(parents=True, exist_ok=True)

# GitHub raw base URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/JiaheZhaoWustl/PosterDatabase/main/PosterDataset"

for jsonl_file in jsonl_folder.glob("*.jsonl"):
    base = jsonl_file.stem
    style = base.split("_")[0]
    image_name = base.split("_", 1)[1] + ".png"
    image_path = image_folder / style / image_name
    if not image_path.exists():
        print(f"⚠️ Image not found for {base}")
        continue

    # Read OCR boxes
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f.readlines()]

    # Load image dimensions
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    results = []
    for item in lines:
        box = item["bbox"]
        x0 = min(p[0] for p in box)
        y0 = min(p[1] for p in box)
        x1 = max(p[0] for p in box)
        y1 = max(p[1] for p in box)
        width = x1 - x0
        height = y1 - y0

        results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": round((x0 / img_w) * 100, 2),
                "y": round((y0 / img_h) * 100, 2),
                "width": round((width / img_w) * 100, 2),
                "height": round((height / img_h) * 100, 2),
                "rectanglelabels": ["text"]
            }
        })

    # ✅ Use GitHub-hosted image URL
    image_url = f"{GITHUB_RAW_BASE}/{style}/{image_name}"

    ls_data = {
        "data": {
            "image": image_url
        },
        "annotations": [
            {
                "result": results
            }
        ]
    }

    out_path = output_folder / f"{base}_labelstudio.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ls_data, f, indent=2)

    print(f"✅ Converted: {out_path}")
