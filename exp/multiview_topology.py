import argparse
import base64
import json
import os
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-luna"
WHEELBASE_M = 2.7


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are the semantic point-localization stage of an AI-assisted
vehicle exterior CAD topology reconstruction system.

You are given multiple photographs of the SAME vehicle.

Your job is ONLY to localize a predefined set of semantic
vehicle exterior topology points.

DO NOT invent topology.
DO NOT invent additional topology points.
DO NOT return edges.
DO NOT return panel loops.
DO NOT generate CAD geometry.

Python will construct the topology from your point locations.

============================================================
AUTOCRAFT COORDINATE SYSTEM
============================================================

Vehicle coordinate system:

+X = forward
-X = rear

+Y = vehicle LEFT
-Y = vehicle RIGHT

+Z = up

Wheelbase = 2.7 meters.

Return approximate normalized vehicle coordinates:

x_norm:
    0 = rear
    1 = front

y_norm:
    vehicle center = 0
    vehicle left = positive
    vehicle right = negative

z_norm:
    ground = 0
    roof = approximately 1

These coordinates describe the physical vehicle, not image
coordinates.

============================================================
EXTERIOR ONLY
============================================================

Only exterior vehicle topology is allowed.

NEVER create:

- chassis
- interior
- internal reinforcement
- door sill interior structure
- B-pillar
- interior trim

CRITICAL:

THE B-PILLAR IS NOT AN EXTERIOR PANEL.

NEVER create a B-pillar topology point.

Visible window glass IS exterior topology.

Door windows must be treated as actual exterior panels.

============================================================
MULTI-VIEW REASONING
============================================================

All photographs show the SAME physical vehicle.

Use ALL photographs jointly.

Do not create a separate topology for each photograph.

If a point is visible in multiple photographs:

- reconcile the observations
- determine its best physical location
- use the most reliable observation

If only one side is visible:

- infer the opposite side using vehicle symmetry
- ordinary exterior body points may be mirrored
- confidence should reflect that they are inferred

Do NOT invent asymmetric details.

============================================================
LEFT / RIGHT
============================================================

L = physical vehicle LEFT = +Y

R = physical vehicle RIGHT = -Y

This refers to the vehicle's physical sides,
NOT the left/right side of the photograph.

============================================================
VEHICLE TYPE
============================================================

Determine:

vehicle_body_style:

sedan
hatchback
wagon
coupe
convertible
suv
pickup
van
other

Determine:

door_configuration:

2_door
4_door
other

Determine:

rear_configuration:

sedan_trunk
hatch_liftgate
wagon
other

============================================================
POINT LOCALIZATION
============================================================

Only use the predefined point IDs.

Do NOT create any other IDs.

For every point return:

- id
- x_norm
- y_norm
- z_norm
- visibility
- confidence
- source_images
- reasoning

Visibility values:

visible
partially_visible
inferred_from_symmetry
not_visible

Coordinates must represent the physical location of the point.

============================================================
REQUIRED POINT POLICY
============================================================

For a normal 4-door vehicle, ALL of these points are required:

P01L P01R
P02L P02R
P03L P03R
P04L P04R
P05L P05R
P06L P06R
P07L P07R
P08L P08R
P09L P09R
P10L P10R
P11L P11R
P12L P12R
P13L P13R
P14L P14R
P15L P15R
P16L P16R
P17L P17R
P18L P18R
P19L P19R
P20L P20R
P21L P21R
P22L P22R
P23L P23R
P24L P24R
P25L P25R
P26L P26R
P27L P27R
P28L P28R
P29L P29R
P30L P30R

Do NOT silently omit required points.

If a point is directly visible, localize it.

If it is partially visible, localize it and use:

visibility = "partially_visible"

If it is not visible from the available photographs but can
reasonably be inferred from the opposite side:

visibility = "inferred_from_symmetry"

Only use:

visibility = "not_visible"

when the point genuinely cannot be localized or inferred.

For a normal 4-door vehicle, P28L and P28R are especially
important and MUST be returned.

============================================================
PREDEFINED POINTS
============================================================

P01L / P01R

Lower-left/right points of windshield.

These are the lower windshield corners where the windshield
meets the lower windshield/cowl region and the A-pillar/side
boundary.

These are NOT B-pillar points.

------------------------------------------------------------

P02L / P02R

Front upper point of the front door near P01.

This is the front-door/fender junction around the beltline
region.

------------------------------------------------------------

P03L / P03R

Front/rear door junction on the BELTLINE.

For a 4-door vehicle this is the physical boundary between the
front door and rear door at beltline height.

------------------------------------------------------------

P04L / P04R

Rear-most point of the rear door on the BELTLINE where the
rear door meets the rear quarter panel.

For a 2-door vehicle this becomes the rear-most point of the
single door on the beltline where it meets the quarter panel.

------------------------------------------------------------

P05L / P05R

Front-bottom point of front door where it meets the rocker
panel.

------------------------------------------------------------

P06L / P06R

Front/rear door junction at rocker-panel level.

For a 4-door vehicle this is the lower boundary junction
between the front and rear doors.

------------------------------------------------------------

P07L / P07R

Rear-most door point at rocker-panel level.

This is where the rear door meets the rear quarter-panel/
rocker-region boundary.

For a 2-door vehicle this is the corresponding rear point of
the single door.

------------------------------------------------------------

P08L / P08R

Front lower corner of rocker panel.

------------------------------------------------------------

P09L / P09R

Rear lower corner of rocker panel.

------------------------------------------------------------

P10L / P10R

HEADLIGHT / HOOD / FENDER junction.

This point must lie at the physical exterior location where
three regions meet:

hood + headlight + front fender.

------------------------------------------------------------

P11L / P11R

Inner-most meaningful point on the headlight/hood boundary.

This point must lie on the actual headlight/hood transition.

Do not place it in the middle of the headlight.

------------------------------------------------------------

P12L / P12R

HEADLIGHT / BUMPER / FENDER junction.

This point must lie at the physical exterior transition where
the headlight, bumper and front fender meet.

------------------------------------------------------------

P13L / P13R

FRONT wheel-arch junction where the front fender transitions
to the front bumper.

This point must be associated with the actual front wheel
opening.

Do not place it on the tire.

Do not place it arbitrarily on the bumper.

------------------------------------------------------------

P14L / P14R

Lower front-bumper point around the front wheel-arch region.

This point lies on the lower portion of the bumper/wheel-opening
transition.

------------------------------------------------------------

P15L / P15R

Front bottom point of the front bumper.

This is the meaningful lower/front bumper corner or bottom
transition point visible in the vehicle geometry.

It is NOT necessarily the lowest pixel in the image.

------------------------------------------------------------

P16L / P16R

TOPMOST CENTER POINT OF THE FRONT WHEEL ARCH.

IMPORTANT:

First locate the actual FRONT wheel opening.

Then locate the highest point of the BODY'S wheel-arch boundary.

P16 MUST lie directly on the physical wheel-arch boundary.

Do NOT place P16:

- on the tire
- inside the wheel
- above the wheel opening
- on an arbitrary fender point

------------------------------------------------------------

P17L / P17R

Rear endpoint of the FRONT wheel arch on the front fender.

This is where the wheel-opening boundary transitions toward
the rear/lower body region.

------------------------------------------------------------

P18L / P18R

Rear-most point on the beltline of the quarter panel.

This is the rear end of the side beltline/top-side body boundary.

------------------------------------------------------------

P19L / P19R

Rear junction of rear quarter panel and rear bumper.

This is a rear exterior body-panel junction.

------------------------------------------------------------

P20L / P20R

REAR WHEEL-ARCH / REAR-BUMPER JUNCTION.

This point must be associated with the physical rear wheel
opening.

It lies around the REARWARD side of the rear wheel opening
where the rear quarter panel transitions into the rear bumper.

Do NOT place it arbitrarily on the rear bumper.

Do NOT place it on the tire.

------------------------------------------------------------

P21L / P21R

LOWER REAR-BUMPER POINT around the rear wheel-arch region.

This is a lower transition point associated with the rear
wheel opening and bumper.

It must lie on the actual exterior transition.

------------------------------------------------------------

P22L / P22R

Rear bottom point of rear bumper.

This is the meaningful rear/lower bumper corner or bottom
transition.

It is NOT necessarily the lowest pixel in the photograph.

------------------------------------------------------------

P23L / P23R

TOPMOST CENTER POINT OF THE REAR WHEEL ARCH.

IMPORTANT:

First locate the actual REAR wheel opening.

Then locate the highest point of the BODY'S rear wheel-arch
boundary.

P23 MUST lie directly on the physical rear wheel-arch boundary.

It must NOT:

- lie on the tire
- lie inside the wheel
- lie above the wheel opening
- lie somewhere arbitrary on the quarter panel

The correct point is the crown/topmost point of the actual
rear wheel opening.

------------------------------------------------------------

P24L / P24R

Top-left/right points of the windshield.

These are the upper windshield corners where the windshield
meets the roof/A-pillar region.

------------------------------------------------------------

P25L / P25R

TOP-FRONT point of the FRONT DOOR WINDOW.

This is a window topology point.

It lies on the upper/front portion of the front door window
near P24.

------------------------------------------------------------

P26L / P26R

TOP point of the REAR WINDOW or hatch/rear-window region.

Use the appropriate upper rear-window topology location.

------------------------------------------------------------

P27L / P27R

TOP-REAR point of the REAR DOOR WINDOW.

For a 2-door vehicle, this is the top-rear point of the single
door window.

------------------------------------------------------------

P28L / P28R

MANDATORY FOR A 4-DOOR VEHICLE.

P28 IS SIDE-SPECIFIC.

P28L = left-side roof-rail point where the FRONT DOOR WINDOW
      meets the REAR DOOR WINDOW.

P28R = right-side roof-rail point where the FRONT DOOR WINDOW
      meets the REAR DOOR WINDOW.

IMPORTANT GEOMETRIC RELATIONSHIP:

P25L ---- P28L ---- P27L

P25R ---- P28R ---- P27R

P25 = top-front point of front door window.

P28 = front/rear door-window division point.

P27 = top-rear point of rear door window.

Therefore P28 lies physically BETWEEN P25 and P27 along the
roof-rail/window boundary.

P28 IS NOT:

- the center of the roof
- a centerline point
- a generic roof point
- a hidden structural point
- the B-pillar
- an interior point

The B-pillar must NEVER be returned.

P28 represents the visible exterior window/roof-rail topology
location.

For a 4-door vehicle P28L and P28R MUST be returned.

If directly visible:

visibility = "visible"

If partly obscured:

visibility = "partially_visible"

If one side is unavailable but can be inferred:

visibility = "inferred_from_symmetry"

Do not omit P28 merely because the pillar structure is visible.

The B-pillar itself is NOT being localized.

------------------------------------------------------------

P29L / P29R

Bottom points of the rear window.

------------------------------------------------------------

P30L / P30R

Top-rear junction between the rear quarter panel and trunk lid.

This is mainly applicable to sedan/trunk vehicles.

============================================================
WINDOW REQUIREMENT
============================================================

Windows are EXTERIOR topology.

For a 4-door vehicle, explicitly identify:

front_door_window_left
front_door_window_right
rear_door_window_left
rear_door_window_right

Do not omit door windows because they are transparent.

The door windows must connect with the door topology.

For each side:

FRONT DOOR WINDOW:

P02 -> P03 -> P28 -> P25

REAR DOOR WINDOW:

P03 -> P04 -> P27 -> P28

The important shared window division is:

P03 <-> P28

P28 must therefore be localized even if the B-pillar itself
is visible.

Again:

B-pillar = NOT an exterior panel.

Window glass = exterior panel.

============================================================
FRONT WHEEL ARCH
============================================================

The front wheel opening is an important exterior boundary.

The meaningful topology points include:

P13
P14
P16
P17

P16 is the crown/topmost point.

P13 and P17 are meaningful endpoints/transitions.

P14 is the lower bumper/wheel-opening transition.

Do not generate many points around the wheel arch.

Only localize the predefined meaningful points.

============================================================
REAR WHEEL ARCH
============================================================

The rear wheel opening is an important exterior boundary.

The meaningful topology points include:

P07
P09
P20
P21
P23

P23 = topmost/crown point of the actual rear wheel opening.

P20 = rear-quarter/rear-bumper transition around the rear
      wheel opening.

P21 = lower rear-bumper/wheel-opening transition.

P07 and P09 define the door/rocker side of the rear-quarter
region.

The points should be physically consistent around the same
wheel opening.

Do not independently place these points without considering
their geometric relationship.

============================================================
BUMPER LOWER POINTS
============================================================

The following are required when the bumper is present:

Front bumper:
P14
P15

Rear bumper:
P21
P22

These are meaningful exterior bumper bottom/corner topology
points.

Do not omit them simply because they are not three-way
junctions.

============================================================
NO EXTRA POINTS
============================================================

The ONLY valid point IDs are:

P01L P01R
P02L P02R
P03L P03R
P04L P04R
P05L P05R
P06L P06R
P07L P07R
P08L P08R
P09L P09R
P10L P10R
P11L P11R
P12L P12R
P13L P13R
P14L P14R
P15L P15R
P16L P16R
P17L P17R
P18L P18R
P19L P19R
P20L P20R
P21L P21R
P22L P22R
P23L P23R
P24L P24R
P25L P25R
P26L P26R
P27L P27R
P28L P28R
P29L P29R
P30L P30R

Do not return:

P28

There is no single P28.

============================================================
OUTPUT
============================================================

Return JSON ONLY.

Exact structure:

{
  "vehicle_detected": true,

  "vehicle_body_style": "sedan",

  "door_configuration": "4_door",

  "rear_configuration": "sedan_trunk",

  "coordinate_system": {
    "x": "forward",
    "y": "vehicle_left",
    "z": "up",
    "wheelbase_m": 2.7
  },

  "points": [
    {
      "id": "P01L",
      "x_norm": 0.0,
      "y_norm": 0.0,
      "z_norm": 0.0,
      "visibility": "visible",
      "confidence": 0.95,
      "source_images": [
        "image1.jpg"
      ],
      "reasoning": "Short explanation."
    }
  ]
}

Return the predefined points appropriate to the vehicle.

For normal 4-door vehicles, return P28L and P28R.

Do not return edges.

Do not return loops.

Do not return additional topology.
"""


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(image_paths):

    names = "\n".join(
        f"Image {i + 1}: {p.name}"
        for i, p in enumerate(image_paths)
    )

    return f"""
Analyze the following photographs jointly.

They show the SAME vehicle.

{names}

Localize the predefined AutoCraft topology points.

This is a semantic point-localization task.

Do NOT generate:

- CAD geometry
- edges
- loops
- curves
- meshes

Use the exact point IDs from the system instructions.

============================================================
MOST IMPORTANT REQUIREMENTS
============================================================

1. The B-pillar is NOT an exterior panel.

2. Windows ARE exterior panels.

3. For a 4-door vehicle, P28L and P28R are REQUIRED.

4. P28L/P28R are the visible roof-rail points where the front
   and rear door windows meet.

5. P28 lies between P25 and P27.

6. Do NOT replace P28 with a B-pillar point.

7. P16 and P23 MUST lie directly on the top/crown of the
   actual front/rear wheel opening.

8. P20 and P21 must be physically associated with the rear
   wheel opening and rear bumper transition.

9. P14 and P15 are required front bumper lower/corner points
   when the bumper is visible.

10. P21 and P22 are required rear bumper lower/corner points
    when the bumper is visible.

============================================================
WINDOW CHECK
============================================================

For a normal 4-door vehicle, make sure you explicitly reason
about these four exterior window panels:

front door window left
front door window right
rear door window left
rear door window right

The front and rear door windows meet at:

P03 -> P28

P28 is the upper point of that shared window division.

============================================================
REAR WHEEL CHECK
============================================================

Before placing P20, P21 and P23:

1. Find the rear wheel.
2. Find the actual body wheel opening around it.
3. Find the topmost point of that body opening.
4. Put P23 exactly there.
5. Find where the quarter panel transitions into the bumper
   around the rear wheel opening and place P20 there.
6. Find the lower bumper/wheel-opening transition and place
   P21 there.

These points should describe the same physical rear wheel
opening.

Use all photographs to improve the localization.

Return JSON only.
"""


# ============================================================
# IMAGE DATA URL
# ============================================================

def image_data_url(path: Path) -> str:

    suffix = path.suffix.lower()

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)

    if mime is None:
        raise ValueError(
            f"Unsupported image type: {suffix}"
        )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime};base64,{encoded}"


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(raw):

    raw = raw.strip()

    if raw.startswith("```"):

        lines = raw.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines)

    return json.loads(raw)


# ============================================================
# ALLOWED POINTS
# ============================================================

def allowed_point_ids():

    ids = []

    for number in range(1, 31):

        ids.append(
            f"P{number:02d}L"
        )

        ids.append(
            f"P{number:02d}R"
        )

    return set(ids)


# ============================================================
# CLEAN POINTS
# ============================================================

def clean_points(result):

    allowed = allowed_point_ids()

    cleaned = {}

    for point in result.get(
        "points",
        []
    ):

        point_id = point.get("id")

        if point_id not in allowed:
            continue

        if point_id in cleaned:
            continue

        try:

            point["x_norm"] = float(
                point["x_norm"]
            )

            point["y_norm"] = float(
                point["y_norm"]
            )

            point["z_norm"] = float(
                point["z_norm"]
            )

            point["confidence"] = float(
                point.get(
                    "confidence",
                    0
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        cleaned[point_id] = point

    ordered = []

    for number in range(1, 31):

        for side in ["L", "R"]:

            point_id = (
                f"P{number:02d}{side}"
            )

            if point_id in cleaned:

                ordered.append(
                    cleaned[point_id]
                )

    result["points"] = ordered

    return result


# ============================================================
# MISSING POINT REPORT
# ============================================================

def report_missing_points(result):

    expected = allowed_point_ids()

    actual = {
        point["id"]
        for point in result.get(
            "points",
            []
        )
    }

    missing = sorted(
        expected - actual
    )

    if missing:

        print()
        print(
            "WARNING: MISSING SEMANTIC POINTS"
        )
        print(
            "------------------------------------------"
        )

        for point_id in missing:

            print(
                f"  {point_id}"
            )

        print()

    else:

        print(
            "All predefined points were returned."
        )

    return missing


# ============================================================
# POINT LOOKUP
# ============================================================

def point_map(result):

    return {
        point["id"]: point
        for point in result.get(
            "points",
            []
        )
    }


# ============================================================
# LOOP BUILDING
# ============================================================

def build_loops(result):

    points = point_map(result)

    door_config = result.get(
        "door_configuration",
        "4_door"
    )

    rear_config = result.get(
        "rear_configuration",
        "sedan_trunk"
    )

    loops = []

    def add_loop(
        loop_id,
        loop_type,
        node_ids,
        side=None
    ):

        for node_id in node_ids:

            point = points.get(
                node_id
            )

            if point is None:
                return

            if point.get(
                "visibility"
            ) == "not_visible":

                return

        if len(node_ids) < 3:
            return

        loops.append(
            {
                "id": loop_id,
                "type": loop_type,
                "side": side,
                "outer_loop": node_ids,
            }
        )

    # --------------------------------------------------------
    # Windshield
    # --------------------------------------------------------

    add_loop(
        "windshield",
        "windshield",
        [
            "P01L",
            "P01R",
            "P24R",
            "P24L",
        ],
    )

    # --------------------------------------------------------
    # Roof
    # --------------------------------------------------------

    add_loop(
        "roof",
        "roof",
        [
            "P24L",
            "P24R",
            "P26R",
            "P26L",
        ],
    )

    # --------------------------------------------------------
    # Side topology
    # --------------------------------------------------------

    for side in ["L", "R"]:

        # ----------------------------------------------------
        # Front fender
        # ----------------------------------------------------

        add_loop(
            f"front_fender_{side.lower()}",
            "front_fender",
            [
                f"P02{side}",
                f"P10{side}",
                f"P12{side}",
                f"P13{side}",
                f"P16{side}",
                f"P17{side}",
                f"P08{side}",
            ],
            side,
        )

        # ----------------------------------------------------
        # Rocker
        # ----------------------------------------------------

        add_loop(
            f"rocker_panel_{side.lower()}",
            "rocker_panel",
            [
                f"P05{side}",
                f"P06{side}",
                f"P07{side}",
                f"P09{side}",
                f"P08{side}",
            ],
            side,
        )

        # ----------------------------------------------------
        # Front door
        # ----------------------------------------------------

        if door_config == "4_door":

            add_loop(
                f"front_door_{side.lower()}",
                "front_door",
                [
                    f"P02{side}",
                    f"P03{side}",
                    f"P06{side}",
                    f"P05{side}",
                ],
                side,
            )

        else:

            add_loop(
                f"front_door_{side.lower()}",
                "front_door",
                [
                    f"P02{side}",
                    f"P04{side}",
                    f"P07{side}",
                    f"P05{side}",
                ],
                side,
            )

        # ----------------------------------------------------
        # Rear door
        # ----------------------------------------------------

        if door_config == "4_door":

            add_loop(
                f"rear_door_{side.lower()}",
                "rear_door",
                [
                    f"P03{side}",
                    f"P04{side}",
                    f"P07{side}",
                    f"P06{side}",
                ],
                side,
            )

        # ----------------------------------------------------
        # Front door window
        # ----------------------------------------------------

        if door_config == "4_door":

            add_loop(
                f"front_door_window_{side.lower()}",
                "front_door_window",
                [
                    f"P02{side}",
                    f"P03{side}",
                    f"P28{side}",
                    f"P25{side}",
                ],
                side,
            )

        else:

            add_loop(
                f"front_door_window_{side.lower()}",
                "front_door_window",
                [
                    f"P02{side}",
                    f"P04{side}",
                    f"P27{side}",
                    f"P25{side}",
                ],
                side,
            )

        # ----------------------------------------------------
        # Rear door window
        # ----------------------------------------------------

        if door_config == "4_door":

            add_loop(
                f"rear_door_window_{side.lower()}",
                "rear_door_window",
                [
                    f"P03{side}",
                    f"P04{side}",
                    f"P27{side}",
                    f"P28{side}",
                ],
                side,
            )

        # ----------------------------------------------------
        # Quarter panel
        # ----------------------------------------------------

        quarter_nodes = [
            f"P04{side}",
            f"P07{side}",
            f"P09{side}",
            f"P23{side}",
            f"P20{side}",
            f"P19{side}",
            f"P18{side}",
        ]

        if rear_config == "sedan_trunk":

            quarter_nodes.extend(
                [
                    f"P29{side}",
                    f"P30{side}",
                ]
            )

        add_loop(
            f"quarter_panel_{side.lower()}",
            "rear_quarter_panel",
            quarter_nodes,
            side,
        )

        # ----------------------------------------------------
        # A-pillar exterior boundary
        # ----------------------------------------------------

        add_loop(
            f"a_pillar_{side.lower()}",
            "a_pillar_exterior",
            [
                f"P01{side}",
                f"P02{side}",
                f"P25{side}",
                f"P24{side}",
            ],
            side,
        )

        # ----------------------------------------------------
        # Roof rail
        # ----------------------------------------------------

        add_loop(
            f"roof_rail_{side.lower()}",
            "roof_rail",
            [
                f"P24{side}",
                f"P25{side}",
                f"P28{side}",
                f"P27{side}",
                f"P26{side}",
            ],
            side,
        )

    # --------------------------------------------------------
    # Hood
    # --------------------------------------------------------

    add_loop(
        "hood",
        "hood",
        [
            "P01L",
            "P10L",
            "P11L",
            "P11R",
            "P10R",
            "P01R",
        ],
    )

    # --------------------------------------------------------
    # Front bumper
    # --------------------------------------------------------

    add_loop(
        "front_bumper",
        "front_bumper",
        [
            "P12L",
            "P13L",
            "P14L",
            "P15L",
            "P15R",
            "P14R",
            "P13R",
            "P12R",
        ],
    )

    # --------------------------------------------------------
    # Rear bumper
    # --------------------------------------------------------

    add_loop(
        "rear_bumper",
        "rear_bumper",
        [
            "P19L",
            "P20L",
            "P21L",
            "P22L",
            "P22R",
            "P21R",
            "P20R",
            "P19R",
        ],
    )

    # --------------------------------------------------------
    # Rear window
    # --------------------------------------------------------

    add_loop(
        "rear_window",
        "rear_window",
        [
            "P26L",
            "P26R",
            "P29R",
            "P29L",
        ],
    )

    # --------------------------------------------------------
    # Trunk
    # --------------------------------------------------------

    if rear_config == "sedan_trunk":

        add_loop(
            "trunk_lid",
            "trunk_lid",
            [
                "P30L",
                "P30R",
                "P26R",
                "P26L",
            ],
        )

    result["loops"] = loops

    return result


# ============================================================
# GLOBAL EDGE GRAPH
# ============================================================

def build_edges(result):

    edge_map = {}

    for loop in result.get(
        "loops",
        []
    ):

        node_ids = loop[
            "outer_loop"
        ]

        loop_id = loop[
            "id"
        ]

        for i in range(
            len(node_ids)
        ):

            a = node_ids[i]

            b = node_ids[
                (i + 1) % len(node_ids)
            ]

            if a == b:
                continue

            key = tuple(
                sorted(
                    [a, b]
                )
            )

            if key not in edge_map:

                edge_map[key] = {
                    "id": (
                        f"E"
                        f"{len(edge_map) + 1:04d}"
                    ),
                    "node_a": a,
                    "node_b": b,
                    "loops": [],
                }

            if loop_id not in edge_map[
                key
            ]["loops"]:

                edge_map[key][
                    "loops"
                ].append(
                    loop_id
                )

    edges = list(
        edge_map.values()
    )

    for edge in edges:

        if len(
            edge["loops"]
        ) > 1:

            edge["type"] = (
                "shared_boundary"
            )

        else:

            edge["type"] = (
                "panel_boundary"
            )

    result["edges"] = edges

    return result


# ============================================================
# SCALE TO METERS
# ============================================================

def scale_points(result):

    for point in result.get(
        "points",
        []
    ):

        point["x_m"] = (
            point["x_norm"]
            * WHEELBASE_M
        )

        point["y_m"] = (
            point["y_norm"]
            * WHEELBASE_M
        )

        point["z_m"] = (
            point["z_norm"]
            * WHEELBASE_M
        )

    result["scale"] = {
        "wheelbase_m": WHEELBASE_M
    }

    return result


# ============================================================
# OBJ EXPORT
# ============================================================

def write_obj(
    result,
    path
):

    points = result.get(
        "points",
        []
    )

    edges = result.get(
        "edges",
        []
    )

    index = {
        point["id"]: i + 1
        for i, point in enumerate(points)
    }

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# AutoCraft vehicle topology\n"
        )

        f.write(
            "# +X forward, +Y left, +Z up\n"
        )

        f.write(
            "# Units: meters\n\n"
        )

        for point in points:

            f.write(
                "v "
                f"{point['x_m']:.6f} "
                f"{point['y_m']:.6f} "
                f"{point['z_m']:.6f}\n"
            )

        f.write("\n")

        for edge in edges:

            a = index.get(
                edge["node_a"]
            )

            b = index.get(
                edge["node_b"]
            )

            if a is None or b is None:
                continue

            f.write(
                f"l {a} {b}\n"
            )


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    result,
    path,
    missing
):

    lines = []

    lines.append(
        "AutoCraft Fixed Semantic Topology"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"Model: {MODEL}"
    )

    lines.append(
        f"Body style: "
        f"{result.get('vehicle_body_style')}"
    )

    lines.append(
        f"Doors: "
        f"{result.get('door_configuration')}"
    )

    lines.append(
        f"Rear: "
        f"{result.get('rear_configuration')}"
    )

    lines.append("")

    lines.append(
        f"Points returned: "
        f"{len(result.get('points', []))}"
    )

    lines.append(
        f"Loops: "
        f"{len(result.get('loops', []))}"
    )

    lines.append(
        f"Edges: "
        f"{len(result.get('edges', []))}"
    )

    lines.append("")

    lines.append(
        "MISSING POINTS"
    )

    lines.append(
        "-" * 60
    )

    if missing:

        for point_id in missing:

            lines.append(
                point_id
            )

    else:

        lines.append(
            "None"
        )

    lines.append("")

    lines.append(
        "LOOPS"
    )

    lines.append(
        "-" * 60
    )

    for loop in result.get(
        "loops",
        []
    ):

        nodes = loop[
            "outer_loop"
        ]

        lines.append("")

        lines.append(
            loop["id"]
        )

        lines.append(
            "  "
            + " -> ".join(nodes)
            + " -> "
            + nodes[0]
        )

    lines.append("")

    lines.append(
        "SHARED EDGES"
    )

    lines.append(
        "-" * 60
    )

    for edge in result.get(
        "edges",
        []
    ):

        if edge["type"] == (
            "shared_boundary"
        ):

            lines.append(
                f"{edge['id']}: "
                f"{edge['node_a']} <-> "
                f"{edge['node_b']} "
                f"shared by "
                f"{edge['loops']}"
            )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Vehicle photographs",
    )

    args = parser.parse_args()

    if len(args.images) < 2:

        raise RuntimeError(
            "Use at least two vehicle photographs."
        )

    for path in args.images:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    print()
    print(
        "=========================================="
    )
    print(
        "AutoCraft Fixed Semantic Topology"
    )
    print(
        "=========================================="
    )

    print()

    print(
        f"Model: {MODEL}"
    )

    print(
        "Coordinate system:"
    )

    print(
        "  +X = forward"
    )

    print(
        "  +Y = vehicle left"
    )

    print(
        "  +Z = up"
    )

    print(
        f"Wheelbase = {WHEELBASE_M} m"
    )

    print()

    print(
        "Input images:"
    )

    for path in args.images:

        print(
            f"  {path}"
        )

    client = OpenAI(
        api_key=api_key
    )

    content = [
        {
            "type": "input_text",
            "text": build_user_prompt(
                args.images
            ),
        }
    ]

    for path in args.images:

        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(
                    path
                ),
                "detail": "high",
            }
        )

    print()
    print(
        "Sending photographs to Sol..."
    )

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

    print(
        "Response received."
    )

    result = parse_json(
        response.output_text
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    result = clean_points(
        result
    )

    # --------------------------------------------------------
    # Report missing points BEFORE topology construction
    # --------------------------------------------------------

    missing = report_missing_points(
        result
    )

    # --------------------------------------------------------
    # Build deterministic topology
    # --------------------------------------------------------

    result = build_loops(
        result
    )

    result = build_edges(
        result
    )

    result = scale_points(
        result
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = Path(
        "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = args.images[0].stem

    json_path = (
        output_dir
        / f"{stem}_fixed_topology.json"
    )

    obj_path = (
        output_dir
        / f"{stem}_fixed_topology.obj"
    )

    summary_path = (
        output_dir
        / f"{stem}_fixed_topology.txt"
    )

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
        obj_path
    )

    write_summary(
        result,
        summary_path,
        missing
    )

    print()
    print(
        "=========================================="
    )
    print(
        "RESULT"
    )
    print(
        "=========================================="
    )

    print()

    print(
        f"Points returned: "
        f"{len(result.get('points', []))}"
    )

    print(
        f"Missing points: "
        f"{len(missing)}"
    )

    print(
        f"Loops: "
        f"{len(result.get('loops', []))}"
    )

    print(
        f"Edges: "
        f"{len(result.get('edges', []))}"
    )

    shared = sum(
        1
        for edge in result.get(
            "edges",
            []
        )
        if edge["type"]
        == "shared_boundary"
    )

    print(
        f"Shared edges: {shared}"
    )

    print()

    print(
        f"JSON: {json_path}"
    )

    print(
        f"OBJ: {obj_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print()


if __name__ == "__main__":
    main()