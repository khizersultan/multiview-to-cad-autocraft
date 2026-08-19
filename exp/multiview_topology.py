import argparse
import base64
import json
import os
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-sol"


SYSTEM_PROMPT = r"""
You are the multi-view semantic 3D topology stage of an
AI-assisted vehicle exterior CAD reconstruction system.

Your task is to analyze multiple photographs of THE SAME VEHICLE
and construct an approximate global 3D topology consisting of
meaningful exterior topology nodes and straight-line edges.

This is an INITIAL GEOMETRIC EXPERIMENT.

Do NOT generate CAD surfaces.
Do NOT generate meshes.
Do NOT generate curves.
Do NOT generate OBJ/GLB internally.

The output is a semantic 3D wireframe representation.

============================================================
EXTERIOR ONLY
============================================================

Model ONLY the exterior vehicle.

DO NOT model:

- chassis
- frame
- internal reinforcement
- interior trim
- door sill internal structure
- internal B-pillar structure
- engine bay
- suspension
- interior surfaces

IMPORTANT:

THE B-PILLAR IS NOT AN EXTERIOR PANEL.

Do NOT create a B-pillar component or B-pillar topology node
merely because the B-pillar is visible in an image.

Visible exterior glass IS part of the exterior topology.

Include when visible:

- windshield
- front door windows
- rear door windows
- quarter glass
- rear window

============================================================
GLOBAL VEHICLE COORDINATE SYSTEM
============================================================

Use one canonical coordinate system for the entire vehicle.

The coordinates must be normalized approximately to:

X = vehicle left/right
Y = vehicle front/rear
Z = vertical

Use:

X = -1 ... +1
Y = -1 ... +1
Z = 0 ... +1

Coordinate convention:

X:
    -1 = vehicle left
     0 = vehicle center plane
    +1 = vehicle right

Y:
    -1 = rear
     0 = vehicle center
    +1 = front

Z:
     0 = lowest exterior/body reference
     1 = highest exterior point

IMPORTANT:

The coordinate system is VEHICLE-BASED, NOT IMAGE-BASED.

Do not confuse image left/right with vehicle left/right.

Explicitly reason about vehicle orientation.

============================================================
MULTI-VIEW REASONING
============================================================

All supplied photographs depict the same vehicle.

Use the photographs jointly.

A physical point visible in multiple photographs must receive
ONE global topology ID.

For example:

Image 1:
    front door / front fender upper junction

Image 2:
    same physical location from another angle

These must become:

    global_id = N_FRONT_DOOR_FRONT_UPPER

NOT two independent nodes.

============================================================
MEANINGFUL TOPOLOGY NODES
============================================================

Identify useful topology nodes including, when visible:

Panel junctions:

- front fender / front door
- front door / rear door
- rear door / rear quarter panel
- front door / rocker
- rear door / rocker
- rear quarter / rear bumper
- front fender / front bumper
- hood / front fender
- hood / headlight
- trunk / quarter panel
- hatch / quarter panel

Wheel architecture:

- front wheel arch front transition
- front wheel arch rear transition
- wheel arch / rocker transition
- wheel arch / bumper transition
- wheel arch / fender transition
- rear wheel arch equivalent points

Windows:

- windshield / roof
- windshield / hood/cowl
- front door window corners
- rear door window corners
- quarter glass corners
- rear window / roof
- rear window / trunk/hatch

Bumpers:

- front bumper lower corners
- front bumper important lower transitions
- rear bumper lower corners
- rear bumper important lower transitions

Lighting/grille:

- meaningful headlight junctions
- meaningful tail-light junctions
- grille corners or major topology points

Other:

- other points required to make meaningful exterior
  panel boundaries closed.

DO NOT create arbitrary points along smooth curves.

DO NOT create hundreds of points.

We want meaningful topology nodes.

============================================================
WHEEL ARCHES
============================================================

Wheel arches are particularly important.

Do NOT approximate an entire wheel arch using many points.

Instead identify meaningful transition points such as:

- arch front/body transition
- arch rear/body transition
- arch/rocker transition
- arch/bumper transition
- other salient corner points

These points will later become endpoints of curved CAD boundaries.

============================================================
BUMPER LOWER POINTS
============================================================

When visible, explicitly include:

FRONT:

- left lower corner
- right lower corner
- important lower transitions

REAR:

- left lower corner
- right lower corner
- important lower transitions

These are required even if they are not panel intersections.

============================================================
COMPONENTS
============================================================

Potential components:

- hood
- front_fender
- front_door
- rear_door
- rear_quarter_panel
- rocker_panel
- roof
- roof_rail
- trunk_lid
- hatch_liftgate
- front_bumper
- rear_bumper
- front_door_window
- rear_door_window
- quarter_glass
- windshield
- rear_window
- headlight
- tail_light
- front_grille
- cowl_panel
- header_panel
- other_exterior_panel

Only include components actually present.

Do not assume a component exists simply because it is common
on vehicles.

============================================================
VISIBILITY
============================================================

Never invent completely hidden geometry.

Use:

visible
partially_visible
inferred_from_symmetry

for point visibility.

A point may be inferred from another view if there is strong
evidence that it corresponds to a physical location.

If an opposite-side point is not visible but the vehicle is
sufficiently symmetric, it may be generated using symmetry.

============================================================
SYMMETRY
============================================================

Most ordinary road vehicles are approximately symmetric around
their longitudinal center plane.

When only one side is observed:

1. identify the observed side;
2. determine the corresponding vehicle-side coordinate;
3. create the opposite point by reflection across X=0.

Reflection:

    x_right = -x_left
    y_right = y_left
    z_right = z_left

Do NOT blindly duplicate asymmetrical components such as:

- asymmetric lighting
- exhaust details
- charging ports
- badges
- visible damage
- mirrors if their geometry is clearly asymmetric

For ordinary body topology, symmetry is acceptable as an
INITIAL approximation.

Mark inferred nodes with:

visibility = "inferred_from_symmetry"

and give them a lower confidence than directly observed nodes.

============================================================
3D POSITION ESTIMATION
============================================================

For every global topology node estimate:

x
y
z

using the supplied photographs jointly.

These are NORMALIZED APPROXIMATE VEHICLE COORDINATES.

They are NOT expected to be physically dimensioned yet.

Prioritize:

1. correct semantic identity
2. correct relative position
3. correct side
4. correct height
5. consistent geometry across views

Do NOT fabricate excessive precision.

============================================================
EDGES
============================================================

Create straight-line edges connecting meaningful neighboring
topology nodes.

These are NOT final CAD curves.

They are only a first 3D topology visualization.

Connect nodes when they represent a meaningful exterior
boundary relationship.

Examples:

front_fender:

    upper front
    →
    upper rear
    →
    wheel arch
    →
    lower transition
    →
    bumper transition

front door:

    upper front
    →
    upper rear
    →
    lower rear
    →
    lower front

Window:

    corner
    →
    corner
    →
    corner
    →
    corner

Adjacent panels must eventually share the same edge.

Therefore, if two panels use the same physical boundary,
the edge must exist ONLY ONCE in the global edge list.

============================================================
OUTPUT
============================================================

Return JSON only.

No markdown.

Top-level structure:

{
  "vehicle_detected": true,
  "coordinate_system": {
    "x": "vehicle left/right",
    "y": "vehicle rear/front",
    "z": "vertical",
    "x_range": [-1, 1],
    "y_range": [-1, 1],
    "z_range": [0, 1]
  },

  "components": [],

  "nodes": [
    {
      "id": "N001",
      "semantic_id": "front_door_front_upper_left",
      "component": "front_door",
      "side": "left",
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "visibility": "visible",
      "confidence": 0.0,
      "source_images": [],
      "category": "panel_to_panel",
      "reasoning": ""
    }
  ],

  "edges": [
    {
      "id": "E001",
      "start": "N001",
      "end": "N002",
      "relationship": "shared_boundary"
    }
  ]
}

============================================================
QUALITY REQUIREMENTS
============================================================

Do not optimize for node count.

Optimize for topology quality.

A good result should contain enough nodes to describe the
meaningful exterior topology while avoiding arbitrary points.

Do not create duplicate nodes for the same physical location.

Do not create B-pillar nodes.

Do include window topology.

Do include wheel-arch salient points.

Do include bumper lower/corner points.

Do not invent hidden components.

Use all photographs jointly.
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


def build_user_prompt(image_paths):
    image_descriptions = []

    for index, path in enumerate(image_paths):
        image_descriptions.append(
            f"Image {index + 1}: {path.name}"
        )

    image_list = "\n".join(image_descriptions)

    return f"""
You are analyzing multiple photographs of the SAME vehicle.

Photographs supplied:

{image_list}

Construct ONE GLOBAL exterior topology for the vehicle.

Important:

- Treat all photographs as observations of the same physical car.
- Match physical topology points across views.
- Do not create duplicate global nodes for the same physical point.
- Do not confuse image left/right with vehicle left/right.
- Determine vehicle orientation from the photographs.
- Use the vehicle coordinate system defined in the system instructions.
- Use multiple photographs jointly to estimate each node position.
- If only one side is visible, infer the opposite side using symmetry.
- Clearly mark symmetry-generated nodes.
- Do not invent hidden details unrelated to symmetry.
- Windows are exterior components.
- B-pillar is NOT an exterior component.

We need a useful INITIAL 3D WIREFRAME, not CAD.

Return the most meaningful topology nodes required to describe
the exterior vehicle.

Include:

- panel junctions
- wheel-arch transition points
- bumper bottom/corner points
- window corners/topology points
- roof transitions
- important light/grille topology points
- other salient exterior topology points required to connect
  the body topology

Then connect neighboring nodes with straight-line edges.

Remember:

ONE PHYSICAL BOUNDARY = ONE GLOBAL EDGE.

Do not create separate duplicate edges for adjacent panels.

Return JSON only.
"""


def extract_json(raw: str):
    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    return json.loads(raw)


def validate_topology(result):
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    node_ids = {node["id"] for node in nodes}

    valid_edges = []

    for edge in edges:
        start = edge.get("start")
        end = edge.get("end")

        if start not in node_ids:
            continue

        if end not in node_ids:
            continue

        if start == end:
            continue

        valid_edges.append(edge)

    result["edges"] = valid_edges

    return result


def write_obj(result, output_path: Path):
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    node_index = {
        node["id"]: index + 1
        for index, node in enumerate(nodes)
    }

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# AutoCraft initial 3D topology\n")
        f.write("# Straight-line semantic topology only\n\n")

        # OBJ vertex coordinates.
        for node in nodes:
            x = float(node["x"])
            y = float(node["y"])
            z = float(node["z"])

            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        f.write("\n")

        # OBJ line primitives.
        for edge in edges:
            start = node_index.get(edge["start"])
            end = node_index.get(edge["end"])

            if start is None or end is None:
                continue

            f.write(f"l {start} {end}\n")


def write_node_summary(result, output_path: Path):
    nodes = result.get("nodes", [])

    lines = []

    lines.append("AutoCraft Initial 3D Topology")
    lines.append("=" * 40)
    lines.append("")

    lines.append(
        f"Nodes: {len(nodes)}"
    )

    lines.append(
        f"Edges: {len(result.get('edges', []))}"
    )

    lines.append("")

    for node in nodes:
        lines.append(
            f"{node['id']:>5} "
            f"{node.get('semantic_id', ''):<40} "
            f"({node['x']:.3f}, "
            f"{node['y']:.3f}, "
            f"{node['z']:.3f})"
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view vehicle semantic 3D topology experiment"
    )

    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Two or more photographs of the same vehicle",
    )

    args = parser.parse_args()

    if len(args.images) < 2:
        raise ValueError(
            "Provide at least two photographs of the same vehicle."
        )

    for image_path in args.images:
        if not image_path.exists():
            raise FileNotFoundError(image_path)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    print(f"Model: {MODEL}")
    print(f"Images: {len(args.images)}")

    for image_path in args.images:
        print(f"  - {image_path}")

    client = OpenAI(api_key=api_key)

    content = [
        {
            "type": "input_text",
            "text": build_user_prompt(args.images),
        }
    ]

    for image_path in args.images:
        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(image_path),
                "detail": "high",
            }
        )

    print()
    print("Sending multi-view analysis to Sol...")

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    raw = response.output_text.strip()

    result = extract_json(raw)
    result = validate_topology(result)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the first image name as the experiment name.
    stem = args.images[0].stem

    json_path = output_dir / f"{stem}_3d_topology.json"
    obj_path = output_dir / f"{stem}_3d_topology.obj"
    summary_path = output_dir / f"{stem}_3d_topology.txt"

    json_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_obj(
        result,
        obj_path,
    )

    write_node_summary(
        result,
        summary_path,
    )

    print()
    print("======================================")
    print("3D TOPOLOGY EXPERIMENT COMPLETE")
    print("======================================")
    print(f"Nodes:       {len(result.get('nodes', []))}")
    print(f"Edges:       {len(result.get('edges', []))}")
    print(f"JSON:        {json_path}")
    print(f"OBJ:         {obj_path}")
    print(f"Summary:     {summary_path}")
    print()
    print("Open the OBJ in Blender for initial wireframe inspection.")


if __name__ == "__main__":
    main()