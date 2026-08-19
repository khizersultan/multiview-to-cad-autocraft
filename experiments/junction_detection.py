import argparse
import base64
import json
import os
from pathlib import Path

import cv2
from openai import OpenAI


MODEL = "gpt-5.6-sol"

SYSTEM_PROMPT = r"""
You are the semantic localization stage of an AI-assisted vehicle
exterior CAD reconstruction system.

Your job is NOT to generate CAD geometry and NOT to draw complete
panel boundaries. Your job is to identify the visible exterior
components and discover meaningful topology junction points.

IMPORTANT:
- Model ONLY exterior vehicle surfaces.
- Never treat the B-pillar as an exterior panel.
- Visible window glass/panes ARE exterior surfaces.
- Do not invent components or hidden points.
- Do not use a fixed number of junctions.
- Prefer meaningful topology nodes over many arbitrary points.
- Coordinates must refer to the ORIGINAL image.
- x_norm: 0 at image left, 1 at image right.
- y_norm: 0 at image top, 1 at image bottom.
- pixel coordinates must be integer image coordinates.
- If only one side is visible, do not invent the opposite side.
- If a point is partially visible but reasonably inferable, mark it
  partially_visible and lower confidence.

Identify points including, when visible and meaningful:
1. panel-to-panel junctions
2. three-way exterior junctions
3. wheel-arch junctions
4. wheel-arch salient transition points
5. front bumper lower/corner points
6. rear bumper lower/corner points
7. window/roof topology points
8. meaningful light/grille topology points

Examples of important relationships include:
front_fender + front_door
front_door + rear_door
rear_door + rear_quarter_panel
rear_quarter_panel + rear_bumper
front_fender + front_bumper
front_door + rocker_panel
rear_door + rocker_panel
hood + front_fender
hood + headlight
trunk_lid + rear_quarter_panel
front/rear wheel arch transitions
front bumper left/right bottom points
rear bumper left/right bottom points
window pane + door + roof/beltline regions
windshield + roof + hood/cowl region
rear window + roof + trunk/hatch region
meaningful headlight/tail-light/grille junctions

For left/right naming, use the vehicle/image geometry and explicitly
state the interpretation in the node metadata.

Return JSON only.
"""


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if not mime:
        raise ValueError(f"Unsupported image type: {suffix}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_user_prompt(width: int, height: int) -> str:
    return f"""
Analyze the supplied vehicle photograph.

Image size: width={width}, height={height} pixels.

Return exactly this top-level JSON structure:

{{
  "vehicle_detected": true,
  "visible_sides": [],
  "components": [
    {{
      "id": "front_door_left",
      "type": "front_door",
      "visibility": "visible",
      "confidence": 0.0
    }}
  ],
  "junction_nodes": [
    {{
      "id": "J001",
      "x_norm": 0.0,
      "y_norm": 0.0,
      "x_px": 0,
      "y_px": 0,
      "category": "panel_to_panel",
      "components": ["front_door_left", "rear_door_left"],
      "visibility": "visible",
      "confidence": 0.0,
      "reasoning": "Short explanation of why this is a meaningful topology node.",
      "side_interpretation": "Explain left/right convention when relevant."
    }}
  ]
}}

Allowed/encouraged category values:
- panel_to_panel
- three_way
- wheel_arch_junction
- wheel_arch_salient
- bumper_bottom
- window_roof
- light_grille
- other_salient

Component types should use semantic names such as:
hood, front_fender, front_door, rear_door, rear_quarter_panel,
rocker_panel, roof, trunk_lid, hatch_liftgate, front_bumper,
rear_bumper, windshield, front_door_window, rear_door_window,
quarter_glass, rear_window, headlight, tail_light, front_grille,
cowl_panel, header_panel, other_exterior_panel.

For every node:
- place the point at the actual physical junction/transition,
  not approximately in the middle of a panel;
- use normalized coordinates plus pixel coordinates;
- include only points useful for reconstructing closed exterior
  boundaries;
- avoid arbitrary points along smooth seams;
- include required bumper bottom points when visible;
- include useful wheel-arch transition points;
- never create a B-pillar component.

Image coordinates must be consistent with the supplied dimensions.
"""


def annotate(image_path: Path, result: dict, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    for node in result.get("junction_nodes", []):
        x = int(round(node["x_norm"] * (w - 1)))
        y = int(round(node["y_norm"] * (h - 1)))
        node["x_px"] = x
        node["y_px"] = y

        cv2.circle(image, (x, y), max(5, min(w, h) // 120), (0, 0, 255), -1)
        label = node["id"]
        conf = node.get("confidence")
        if conf is not None:
            label += f" ({conf:.2f})"

        cv2.putText(
            image,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, min(w, h) / 1800),
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, min(w, h) / 1800),
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. In PowerShell use: "
            '$env:OPENAI_API_KEY="your-key"'
        )

    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": build_user_prompt(width, height),
                },
                {
                    "type": "input_image",
                    "image_url": image_data_url(image_path),
                    "detail": "high",
                },
            ],
        }],
    )

    raw = response.output_text.strip()

    # Be tolerant of accidental markdown fences while still requiring JSON.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()

    result = json.loads(raw)

    # Normalize/recalculate pixel coordinates from normalized coordinates.
    for node in result.get("junction_nodes", []):
        node["x_px"] = int(round(float(node["x_norm"]) * (width - 1)))
        node["y_px"] = int(round(float(node["y_norm"]) * (height - 1)))

    stem = image_path.stem
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f"{stem}_junctions.json"
    image_out = output_dir / f"{stem}_junctions.png"

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    annotate(image_path, result, image_out)

    print(f"Model: {MODEL}")
    print(f"Image: {image_path}")
    print(f"Nodes: {len(result.get('junction_nodes', []))}")
    print(f"JSON: {json_path}")
    print(f"Visualization: {image_out}")


if __name__ == "__main__":
    main()
