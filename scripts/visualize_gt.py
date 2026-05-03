import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

LABELS_JSON = Path("data/raw/bdd100k/labels/bdd100k_labels_images_val.json")
IMAGES_DIR = Path("data/raw/bdd100k/images/val")
OUTPUT_DIR = Path("data/visualizations/gt_boxes")

DETECTION_CATEGORIES = (
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels from {LABELS_JSON}...")
    with open(LABELS_JSON) as f:
        data = json.load(f)

    processed_count = 0
    max_images = 10

    for item in data:
        img_name = item.get("name")
        img_path = IMAGES_DIR / img_name

        if not img_path.exists():
            continue

        print(f"Processing {img_name}...")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        labels = item.get("labels", [])
        for label in labels:
            category = label.get("category")
            if category not in DETECTION_CATEGORIES:
                continue
            
            box2d = label.get("box2d")
            if box2d:
                x1 = box2d.get("x1")
                y1 = box2d.get("y1")
                x2 = box2d.get("x2")
                y2 = box2d.get("y2")
                
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                text = category
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle(text_bbox, fill="red")
                
                draw.text((x1, y1), text, fill="white", font=font)

        out_path = OUTPUT_DIR / img_name
        img.save(out_path)
        print(f"Saved visualization to {out_path}")

        processed_count += 1
        if processed_count >= max_images:
            break

    print(f"Successfully generated {processed_count} images.")

if __name__ == "__main__":
    main()
