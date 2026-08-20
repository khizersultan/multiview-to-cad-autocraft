import argparse
import json
import struct
from pathlib import Path

import numpy as np


WHEELBASE = 2.7


# ============================================================
# Binary writing helpers
# ============================================================

def write_int(f, value):
    f.write(struct.pack("<i", int(value)))


def write_double(f, value):
    f.write(struct.pack("<d", float(value)))


# ============================================================
# JSON
# ============================================================

def load_json(path):

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Point helpers
# ============================================================

def get_point_coordinates(point):

    if not all(
        key in point
        for key in ("x_m", "y_m", "z_m")
    ):
        raise ValueError(
            f"Point {point.get('id')} does not contain "
            "x_m, y_m and z_m coordinates."
        )

    return np.array(
        [
            float(point["x_m"]),
            float(point["y_m"]),
            float(point["z_m"]),
        ],
        dtype=float,
    )


def find_point(points, point_id):

    for point in points:
        if point["id"] == point_id:
            return point

    return None


# ============================================================
# Node indexing
# ============================================================

def build_node_index(result):

    points = result.get("points", [])

    return {
        point["id"]: index
        for index, point in enumerate(points)
    }


# ============================================================
# Vehicle center
# ============================================================

def calculate_vehicle_center(result):

    points = result.get("points", [])

    coordinates = np.array(
        [
            get_point_coordinates(point)
            for point in points
        ]
    )

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)

    return (minimum + maximum) / 2.0


# ============================================================
# Loop normal
# ============================================================

def calculate_loop_normal(
    node_ids,
    points_by_id
):
    """
    Calculate an approximate loop normal using Newell's method.

    This works better than simply using the first three points,
    especially for large/non-planar vehicle panel loops.
    """

    if len(node_ids) < 3:
        return np.zeros(3)

    coords = np.array(
        [
            get_point_coordinates(
                points_by_id[node_id]
            )
            for node_id in node_ids
        ]
    )

    normal = np.zeros(3)

    for i in range(len(coords)):

        current = coords[i]
        next_point = coords[
            (i + 1) % len(coords)
        ]

        normal[0] += (
            current[1] - next_point[1]
        ) * (
            current[2] + next_point[2]
        )

        normal[1] += (
            current[2] - next_point[2]
        ) * (
            current[0] + next_point[0]
        )

        normal[2] += (
            current[0] - next_point[0]
        ) * (
            current[1] + next_point[1]
        )

    length = np.linalg.norm(normal)

    if length < 1e-12:
        return np.zeros(3)

    return normal / length


# ============================================================
# Loop outward direction
# ============================================================

def calculate_outward_direction(
    node_ids,
    points_by_id,
    vehicle_center
):
    """
    Estimate which direction is outward from the vehicle.

    The vehicle is represented by its bounding box.

    A surface farther from the vehicle center along:
        X -> front/rear
        Y -> left/right
        Z -> roof
    is assumed to face outward in that direction.

    This is intentionally simple because this experiment is
    about producing a correctly oriented initial wire topology.
    """

    coords = np.array(
        [
            get_point_coordinates(
                points_by_id[node_id]
            )
            for node_id in node_ids
        ]
    )

    centroid = coords.mean(axis=0)

    offset = centroid - vehicle_center

    # Ignore very small numerical offsets.
    scale = max(
        np.linalg.norm(offset),
        1e-12
    )

    normalized_offset = offset / scale

    return normalized_offset


# ============================================================
# Orient loop CCW from exterior
# ============================================================

def orient_loop_ccw(
    node_ids,
    points_by_id,
    vehicle_center
):
    """
    Return node order such that the loop is CCW when viewed
    from the estimated exterior/visible side.

    Right-hand rule:

        CCW viewed from outside
        =>
        loop normal points toward viewer.
    """

    normal = calculate_loop_normal(
        node_ids,
        points_by_id
    )

    outward = calculate_outward_direction(
        node_ids,
        points_by_id,
        vehicle_center
    )

    # Degenerate loop.
    if np.linalg.norm(normal) < 1e-12:
        return node_ids

    # If normal points inward, reverse the loop.
    if np.dot(normal, outward) < 0:

        return list(reversed(node_ids))

    return node_ids


# ============================================================
# Dimensions
# ============================================================

def calculate_dimensions(result):

    points = result.get("points", [])

    if not points:
        raise ValueError(
            "No topology points found."
        )

    coordinates = np.array(
        [
            get_point_coordinates(point)
            for point in points
        ]
    )

    min_x = coordinates[:, 0].min()
    max_x = coordinates[:, 0].max()

    min_y = coordinates[:, 1].min()
    max_y = coordinates[:, 1].max()

    min_z = coordinates[:, 2].min()
    max_z = coordinates[:, 2].max()

    # --------------------------------------------------------
    # Body width
    # --------------------------------------------------------

    width = max_y - min_y

    # --------------------------------------------------------
    # Ground clearance
    # --------------------------------------------------------

    rocker_ids = [
        "P08L",
        "P08R",
        "P09L",
        "P09R",
    ]

    rocker_z = []

    for point_id in rocker_ids:

        point = find_point(
            points,
            point_id
        )

        if point is not None:

            _, _, z = get_point_coordinates(
                point
            )

            rocker_z.append(z)

    if not rocker_z:
        raise ValueError(
            "Could not determine ground clearance. "
            "No P08/P09 points found."
        )

    ground_clearance = min(
        rocker_z
    )

    # --------------------------------------------------------
    # Shoulder height
    # --------------------------------------------------------

    shoulder_ids = [
        "P02L",
        "P02R",
        "P03L",
        "P03R",
        "P04L",
        "P04R",
        "P18L",
        "P18R",
    ]

    shoulder_z = []

    for point_id in shoulder_ids:

        point = find_point(
            points,
            point_id
        )

        if point is not None:

            _, _, z = get_point_coordinates(
                point
            )

            shoulder_z.append(z)

    if not shoulder_z:
        raise ValueError(
            "Could not determine shoulder height."
        )

    shoulder_height = (
        sum(shoulder_z)
        / len(shoulder_z)
    )

    # --------------------------------------------------------
    # Height
    # --------------------------------------------------------

    height = max_z

    # --------------------------------------------------------
    # Axles
    #
    # Rear axle = X 0
    # Front axle = X 2.7
    # --------------------------------------------------------

    rear_axle_x = 0.0
    front_axle_x = WHEELBASE

    # --------------------------------------------------------
    # Overhangs
    # --------------------------------------------------------

    front_overhang = max(
        0.0,
        max_x - front_axle_x
    )

    rear_overhang = max(
        0.0,
        rear_axle_x - min_x
    )

    # --------------------------------------------------------
    # Front wheel radius
    # --------------------------------------------------------

    front_arch_point = (
        find_point(points, "P13L")
        or find_point(points, "P13R")
    )

    if front_arch_point is None:

        front_wheel_radius = 0.0

    else:

        x, _, _ = get_point_coordinates(
            front_arch_point
        )

        front_wheel_radius = abs(
            x - front_axle_x
        )

    # --------------------------------------------------------
    # Rear wheel radius
    # --------------------------------------------------------

    rear_arch_point = (
        find_point(points, "P20L")
        or find_point(points, "P20R")
    )

    if rear_arch_point is None:

        rear_wheel_radius = 0.0

    else:

        x, _, _ = get_point_coordinates(
            rear_arch_point
        )

        rear_wheel_radius = abs(
            x - rear_axle_x
        )

    return {
        "wheelbase": WHEELBASE,
        "width": width,
        "ground_clearance": ground_clearance,
        "shoulder_height": shoulder_height,
        "height": height,
        "front_overhang": front_overhang,
        "rear_overhang": rear_overhang,
        "front_wheel_radius": front_wheel_radius,
        "rear_wheel_radius": rear_wheel_radius,
    }


# ============================================================
# Edge lookup
# ============================================================

def build_edge_lookup(result):

    edges = result.get("edges", [])

    lookup = {}

    for index, edge in enumerate(edges):

        a = edge["node_a"]
        b = edge["node_b"]

        lookup[(a, b)] = index
        lookup[(b, a)] = index

    return lookup


# ============================================================
# Convert oriented loop to signed edges
# ============================================================

def convert_loop(
    loop,
    result,
    edge_lookup,
    points_by_id,
    vehicle_center
):

    original_node_ids = loop["outer_loop"]

    if len(original_node_ids) < 3:
        raise ValueError(
            f"Loop {loop['id']} has fewer than 3 nodes."
        )

    # --------------------------------------------------------
    # Orient loop CCW from exterior.
    # --------------------------------------------------------

    node_ids = orient_loop_ccw(
        original_node_ids,
        points_by_id,
        vehicle_center
    )

    edges = result.get("edges", [])

    signed_indices = []

    for i in range(len(node_ids)):

        node1 = node_ids[i]

        node2 = node_ids[
            (i + 1) % len(node_ids)
        ]

        edge_index = edge_lookup.get(
            (node1, node2)
        )

        if edge_index is None:

            raise ValueError(
                f"Loop '{loop['id']}' is not closed. "
                f"Missing edge {node1} -> {node2}."
            )

        edge = edges[edge_index]

        edge_node1 = edge["node_a"]
        edge_node2 = edge["node_b"]

        # Same direction.
        if (
            edge_node1 == node1
            and
            edge_node2 == node2
        ):

            signed_index = edge_index

        # Opposite direction.
        elif (
            edge_node1 == node2
            and
            edge_node2 == node1
        ):

            signed_index = -edge_index - 1

        else:

            raise RuntimeError(
                "Unexpected edge orientation."
            )

        signed_indices.append(
            signed_index
        )

    return signed_indices


# ============================================================
# Topology validation
# ============================================================

def validate_topology(result):

    points = result.get("points", [])
    edges = result.get("edges", [])
    loops = result.get("loops", [])

    point_ids = {
        point["id"]
        for point in points
    }

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    for edge in edges:

        if edge["node_a"] not in point_ids:

            raise ValueError(
                f"Edge {edge.get('id')} references "
                f"missing node {edge['node_a']}."
            )

        if edge["node_b"] not in point_ids:

            raise ValueError(
                f"Edge {edge.get('id')} references "
                f"missing node {edge['node_b']}."
            )

    # --------------------------------------------------------
    # Loops
    # --------------------------------------------------------

    edge_lookup = build_edge_lookup(
        result
    )

    for loop in loops:

        node_ids = loop["outer_loop"]

        if len(node_ids) < 3:

            raise ValueError(
                f"Loop {loop['id']} has fewer than 3 nodes."
            )

        for i in range(len(node_ids)):

            node1 = node_ids[i]

            node2 = node_ids[
                (i + 1) % len(node_ids)
            ]

            if (
                node1,
                node2
            ) not in edge_lookup:

                raise ValueError(
                    f"Loop {loop['id']} is open. "
                    f"Missing edge "
                    f"{node1} -> {node2}."
                )


# ============================================================
# Write BSH
# ============================================================

def write_bsh(
    result,
    dimensions,
    output_path
):

    points = result.get("points", [])
    edges = result.get("edges", [])
    loops = result.get("loops", [])

    node_index = build_node_index(
        result
    )

    points_by_id = {
        point["id"]: point
        for point in points
    }

    edge_lookup = build_edge_lookup(
        result
    )

    vehicle_center = calculate_vehicle_center(
        result
    )

    # --------------------------------------------------------
    # Convert and orient all loops first.
    # --------------------------------------------------------

    converted_loops = []

    for loop in loops:

        signed_edges = convert_loop(
            loop,
            result,
            edge_lookup,
            points_by_id,
            vehicle_center
        )

        converted_loops.append(
            signed_edges
        )

    # --------------------------------------------------------
    # Write binary
    # --------------------------------------------------------

    with output_path.open("wb") as f:

        # ====================================================
        # HEADER
        # ====================================================

        write_int(f, 0)
        write_int(f, 1)

        # ====================================================
        # VEHICLE PARAMETERS
        # ====================================================

        write_double(
            f,
            dimensions["wheelbase"]
        )

        write_double(
            f,
            dimensions["width"]
        )

        write_double(
            f,
            dimensions["ground_clearance"]
        )

        write_double(
            f,
            dimensions["shoulder_height"]
        )

        write_double(
            f,
            dimensions["height"]
        )

        write_double(
            f,
            dimensions["front_overhang"]
        )

        write_double(
            f,
            dimensions["rear_overhang"]
        )

        write_double(
            f,
            dimensions["front_wheel_radius"]
        )

        write_double(
            f,
            dimensions["rear_wheel_radius"]
        )

        # ====================================================
        # NODES
        # ====================================================

        write_int(
            f,
            len(points)
        )

        for point in points:

            x, y, z = get_point_coordinates(
                point
            )

            write_int(f, 0)

            
            write_double(f, x)
            write_double(f, y)
            write_double(f, z)

        # ====================================================
        # EDGES
        # ====================================================

        write_int(
            f,
            len(edges)
        )

        for edge in edges:

            node1_index = node_index[
                edge["node_a"]
            ]

            node2_index = node_index[
                edge["node_b"]
            ]

            write_int(
                f,
                node1_index
            )

            write_int(
                f,
                node2_index
            )

            write_double(f, 0.0)
            write_double(f, 0.0)
            write_double(f, 0.0)

        # ====================================================
        # LOOPS
        # ====================================================

        write_int(
            f,
            len(converted_loops)
        )

        for signed_edges in converted_loops:

            write_int(
                f,
                len(signed_edges)
            )

            for signed_edge in signed_edges:

                write_int(
                    f,
                    signed_edge
                )

        # ====================================================
        # SURFACES
        # ====================================================

        write_int(
            f,
            len(converted_loops)
        )

        for loop_index in range(
            len(converted_loops)
        ):

            write_int(f, 0)
            write_int(f, loop_index)
            write_int(f, 0)
            write_int(f, 0)

        # ====================================================
        # FINAL ZERO
        # ====================================================

        write_int(
            f,
            0
        )


# ============================================================
# Print dimensions
# ============================================================

def print_dimensions(
    dimensions
):

    print()
    print("==========================================")
    print("BODY SHELL PARAMETERS")
    print("==========================================")

    print(
        f"Wheelbase          : "
        f"{dimensions['wheelbase']:.6f} m"
    )

    print(
        f"Body width         : "
        f"{dimensions['width']:.6f} m"
    )

    print(
        f"Ground clearance   : "
        f"{dimensions['ground_clearance']:.6f} m"
    )

    print(
        f"Shoulder height    : "
        f"{dimensions['shoulder_height']:.6f} m"
    )

    print(
        f"Height             : "
        f"{dimensions['height']:.6f} m"
    )

    print(
        f"Front overhang     : "
        f"{dimensions['front_overhang']:.6f} m"
    )

    print(
        f"Rear overhang      : "
        f"{dimensions['rear_overhang']:.6f} m"
    )

    print(
        f"Front wheel radius : "
        f"{dimensions['front_wheel_radius']:.6f} m"
    )

    print(
        f"Rear wheel radius  : "
        f"{dimensions['rear_wheel_radius']:.6f} m"
    )

    print()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert AutoCraft topology JSON "
            "to binary Body Shell (.bsh)."
        )
    )

    parser.add_argument(
        "json_file",
        type=Path
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None
    )

    args = parser.parse_args()

    if not args.json_file.exists():

        raise FileNotFoundError(
            args.json_file
        )

    print()
    print("AutoCraft Topology -> Body Shell")
    print("================================")

    print(
        f"Input: {args.json_file}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    result = load_json(
        args.json_file
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print(
        "Validating topology..."
    )

    validate_topology(
        result
    )

    print(
        "Topology validation passed."
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    dimensions = calculate_dimensions(
        result
    )

    print_dimensions(
        dimensions
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if args.output is None:

        args.output = (
            args.json_file.parent
            /
            f"{args.json_file.stem}.bsh"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    print(
        f"Writing: {args.output}"
    )

    write_bsh(
        result,
        dimensions,
        args.output
    )

    print()
    print("==========================================")
    print("BSH CREATED SUCCESSFULLY")
    print("==========================================")

    print(
        f"File     : {args.output}"
    )

    print(
        f"Nodes    : "
        f"{len(result.get('points', []))}"
    )

    print(
        f"Edges    : "
        f"{len(result.get('edges', []))}"
    )

    print(
        f"Loops    : "
        f"{len(result.get('loops', []))}"
    )

    print(
        f"Surfaces : "
        f"{len(result.get('loops', []))}"
    )

    print(
        "Loop orientation: exterior CCW"
    )

    print(
        "Final int 0: written"
    )

    print()


if __name__ == "__main__":
    main()