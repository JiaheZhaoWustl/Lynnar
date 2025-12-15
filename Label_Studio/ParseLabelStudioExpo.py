import json
import os
from pathlib import Path

# Input JSON file
input_file = "project-8-at-2025-06-24-19-37-76469e28.json"
# Output folder for individual annotation JSONs
output_dir = Path("annotations_split_3")
output_dir.mkdir(parents=True, exist_ok=True)

# Load the full JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# For each annotation task
for item in data:
    task_id = item["id"]  # e.g., 154, 155, etc.
    annotations = item.get("annotations", [])

    for ann in annotations:
        ann_id = ann["id"]
        # Create a new JSON structure with the relevant data
        ann_data = {
            "task_id": task_id,
            "annotation_id": ann_id,
            "data": item.get("data", {}),
            "annotation": ann
        }

        # Save to a new file
        out_file = output_dir / f"annotation_{task_id}_{ann_id}.json"
        with open(out_file, "w", encoding="utf-8") as out_f:
            json.dump(ann_data, out_f, indent=2)
        print(f"✅ Saved {out_file}")

print("✅ All annotations parsed and saved!")
