import os
import torch
import numpy as np

from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_ID = "microsoft/Florence-2-base"
IMAGE_PATH = "input/car01a.jpg"

OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


# ============================================================
# Load Florence-2
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
).to(device)

model.eval()

print("Florence-2 loaded successfully!")


# ============================================================
# Load image
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

width, height = image.size

print(f"\nImage size: {width} x {height}")


# ============================================================
# Segmentation
# ============================================================

task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
text_input = "car"

print("\nRunning segmentation...")

inputs = processor(
    text=task_prompt + text_input,
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(device) if hasattr(value, "to") else value
    for key, value in inputs.items()
}


with torch.no_grad():

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )


generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=False
)[0]


result = processor.post_process_generation(
    generated_text,
    task=task_prompt,
    image_size=(width, height)
)


print("\n" + "=" * 60)
print("FLORENCE-2 RESULT")
print("=" * 60)

print(result)

print("=" * 60)


# ============================================================
# Extract polygon
# ============================================================

segmentation = result[task_prompt]

polygons = segmentation.get("polygons", [])

if not polygons:
    print("\nNo polygon returned.")
    raise SystemExit

print(f"\nNumber of polygon groups: {len(polygons)}")

# ============================================================
# Create binary mask
# ============================================================

mask = Image.new(
    "L",
    (width, height),
    0
)

draw = ImageDraw.Draw(mask)


for polygon_group in polygons:

    # Florence-2 may return:
    #
    # [
    #     [x1, y1, x2, y2, ...],
    #     [x1, y1, x2, y2, ...]
    # ]
    #
    # or:
    #
    # [
    #     x1, y1, x2, y2, ...
    # ]

    if not polygon_group:
        continue

    # If the first element is itself a list,
    # there are multiple polygon coordinate arrays.
    if isinstance(polygon_group[0], (list, tuple)):

        polygon_list = polygon_group

    else:

        polygon_list = [polygon_group]


    for polygon in polygon_list:

        if len(polygon) < 6:
            continue

        points = []

        # Coordinates are:
        # x1, y1, x2, y2, x3, y3, ...

        for i in range(0, len(polygon) - 1, 2):

            x = int(round(polygon[i]))
            y = int(round(polygon[i + 1]))

            # Keep coordinates inside image
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            points.append((x, y))

        if len(points) >= 3:

            draw.polygon(
                points,
                fill=255
            )

# ============================================================
# Save mask
# ============================================================

mask_path = os.path.join(
    OUTPUT_DIR,
    "car_mask.png"
)

mask.save(mask_path)

print(f"\nMask saved to: {mask_path}")


# ============================================================
# Create overlay
# ============================================================

overlay = image.copy()

overlay_array = np.array(overlay)

mask_array = np.array(mask)

# Create semi-transparent overlay
overlay_color = np.zeros_like(
    overlay_array
)

overlay_color[:, :, 0] = 255

alpha = 0.35

mask_bool = mask_array > 0

overlay_array[mask_bool] = (
    overlay_array[mask_bool] * (1 - alpha)
    + overlay_color[mask_bool] * alpha
).astype(np.uint8)


overlay = Image.fromarray(
    overlay_array
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "car_overlay.png"
)

overlay.save(overlay_path)

print(f"Overlay saved to: {overlay_path}")


print("\nDone!")