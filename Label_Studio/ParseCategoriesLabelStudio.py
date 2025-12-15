import json
import os

# Load the JSON data
with open('project-6-at-2025-05-27-00-32-be7c6f39.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize dictionary to hold data by category
categories = {
    "Title": [],
    "Location": [],
    "Time": [],
    "Host/organization": [],
    "Call-To-Action/Purpose": [],
    "Text descriptions/details": []
}

# Traverse each task and collect bounding boxes by label
for task in data:
    for annotation in task['annotations']:
        for result in annotation['result']:
            label = result['value']['rectanglelabels'][0]
            # Standardize label names to match your categories
            if label in categories:
                categories[label].append({
                    "x": result['value']['x'],
                    "y": result['value']['y'],
                    "width": result['value']['width'],
                    "height": result['value']['height'],
                    "rotation": result['value']['rotation']
                })

# Create an output directory for separated files
output_dir = 'separated_bounding_boxes'
os.makedirs(output_dir, exist_ok=True)

# Save each category's data into separate JSON files
for category, boxes in categories.items():
    filename = f"{output_dir}/{category.replace('/', '_')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(boxes, f, ensure_ascii=False, indent=4)

print(f"Separated bounding box files have been saved in the '{output_dir}' directory.")
