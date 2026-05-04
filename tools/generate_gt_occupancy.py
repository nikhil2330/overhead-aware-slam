#!/usr/bin/env python3
"""Generate a ROS occupancy map directly from Gazebo SDF collision geometry."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_pose(text: str | None) -> tuple[float, float, float, float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = [float(v) for v in text.split()]
    while len(values) < 6:
        values.append(0.0)
    return tuple(values[:6])


def rotation_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ]
    ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]
    rz = [
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return matmul3(matmul3(rz, ry), rx)


def matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def matvec3(matrix: list[list[float]], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def compose_transform(
    parent_t: tuple[float, float, float],
    parent_r: list[list[float]],
    local_pose: tuple[float, float, float, float, float, float],
) -> tuple[tuple[float, float, float], list[list[float]]]:
    lx, ly, lz, roll, pitch, yaw = local_pose
    local_t = (lx, ly, lz)
    local_r = rotation_matrix(roll, pitch, yaw)
    rotated_t = matvec3(parent_r, local_t)
    return (
        (
            parent_t[0] + rotated_t[0],
            parent_t[1] + rotated_t[1],
            parent_t[2] + rotated_t[2],
        ),
        matmul3(parent_r, local_r),
    )


def transform_points(
    points: list[tuple[float, float, float]],
    t: tuple[float, float, float],
    r: list[list[float]],
) -> list[tuple[float, float, float]]:
    transformed: list[tuple[float, float, float]] = []
    for point in points:
        rotated = matvec3(r, point)
        transformed.append((rotated[0] + t[0], rotated[1] + t[1], rotated[2] + t[2]))
    return transformed


def box_points(size_text: str) -> list[tuple[float, float, float]]:
    sx, sy, sz = (float(v) / 2.0 for v in size_text.split())
    return [
        (x, y, z)
        for x in (-sx, sx)
        for y in (-sy, sy)
        for z in (-sz, sz)
    ]


def cylinder_points(radius_text: str, length_text: str, samples: int = 32) -> list[tuple[float, float, float]]:
    radius = float(radius_text)
    half_len = float(length_text) / 2.0
    points: list[tuple[float, float, float]] = []
    for z in (-half_len, half_len):
        for i in range(samples):
            angle = 2.0 * math.pi * i / samples
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return points


def sphere_points(radius_text: str, samples: int = 32) -> list[tuple[float, float, float]]:
    radius = float(radius_text)
    points: list[tuple[float, float, float]] = [(0.0, 0.0, radius), (0.0, 0.0, -radius)]
    for i in range(samples):
        angle = 2.0 * math.pi * i / samples
        points.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
    return points


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def draw_segment(
    occupied: list[list[bool]],
    p0: tuple[float, float],
    p1: tuple[float, float],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    distance = max(abs(dx), abs(dy))
    steps = max(1, int(math.ceil(distance / (resolution * 0.5))))
    height = len(occupied)
    width = len(occupied[0])
    for i in range(steps + 1):
        ratio = i / steps
        x = p0[0] + ratio * dx
        y = p0[1] + ratio * dy
        gx = int(math.floor((x - origin_x) / resolution))
        gy = int(math.floor((y - origin_y) / resolution))
        if 0 <= gx < width and 0 <= gy < height:
            occupied[gy][gx] = True


def rasterize_polygon(
    polygon: list[tuple[float, float]],
    occupied: list[list[bool]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> None:
    if len(polygon) < 3:
        return

    height = len(occupied)
    width = len(occupied[0])

    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)

    min_gx = max(0, int(math.floor((min_x - origin_x) / resolution)))
    max_gx = min(width - 1, int(math.floor((max_x - origin_x) / resolution)))
    min_gy = max(0, int(math.floor((min_y - origin_y) / resolution)))
    max_gy = min(height - 1, int(math.floor((max_y - origin_y) / resolution)))

    sample_offsets = (
        (0.5, 0.5),
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    )

    for gy in range(min_gy, max_gy + 1):
        for gx in range(min_gx, max_gx + 1):
            for ox, oy in sample_offsets:
                x = origin_x + (gx + ox) * resolution
                y = origin_y + (gy + oy) * resolution
                if point_in_polygon(x, y, polygon):
                    occupied[gy][gx] = True
                    break

    for i in range(len(polygon)):
        draw_segment(
            occupied,
            polygon[i],
            polygon[(i + 1) % len(polygon)],
            origin_x,
            origin_y,
            resolution,
        )

    for x, y in polygon:
        gx = int(math.floor((x - origin_x) / resolution))
        gy = int(math.floor((y - origin_y) / resolution))
        if 0 <= gx < width and 0 <= gy < height:
            occupied[gy][gx] = True


def load_model_sdf(models_dir: Path, model_name: str) -> ET.Element:
    model_path = models_dir / model_name / "model.sdf"
    if not model_path.exists():
        raise FileNotFoundError(f"Model SDF not found: {model_path}")
    return ET.parse(model_path).getroot()


def collect_polygons(
    world_path: Path,
    models_dir: Path,
    min_z: float,
    max_obstacle_z: float,
) -> tuple[list[list[tuple[float, float]]], list[str]]:
    world_root = ET.parse(world_path).getroot()
    identity_t = (0.0, 0.0, 0.0)
    identity_r = rotation_matrix(0.0, 0.0, 0.0)

    polygons: list[list[tuple[float, float]]] = []
    warnings: list[str] = []

    for include in world_root.findall(".//include"):
        uri = (include.findtext("uri") or "").strip()
        if not uri.startswith("model://"):
            continue

        model_name = uri.replace("model://", "", 1)
        include_pose = parse_pose(include.findtext("pose"))
        include_t, include_r = compose_transform(identity_t, identity_r, include_pose)

        try:
            model_root = load_model_sdf(models_dir, model_name)
        except FileNotFoundError as exc:
            warnings.append(str(exc))
            continue

        model_el = model_root.find("model")
        if model_el is None:
            warnings.append(f"Missing <model> in {model_name}/model.sdf")
            continue

        model_pose = parse_pose(model_el.findtext("pose"))
        model_t, model_r = compose_transform(include_t, include_r, model_pose)

        for link in model_el.findall("link"):
            link_pose = parse_pose(link.findtext("pose"))
            link_t, link_r = compose_transform(model_t, model_r, link_pose)

            for collision in link.findall("collision"):
                collision_pose = parse_pose(collision.findtext("pose"))
                coll_t, coll_r = compose_transform(link_t, link_r, collision_pose)

                geometry = collision.find("geometry")
                if geometry is None:
                    continue

                local_points: list[tuple[float, float, float]] | None = None

                box = geometry.find("box")
                cylinder = geometry.find("cylinder")
                sphere = geometry.find("sphere")

                if box is not None:
                    size_text = box.findtext("size")
                    if size_text:
                        local_points = box_points(size_text)
                elif cylinder is not None:
                    radius_text = cylinder.findtext("radius")
                    length_text = cylinder.findtext("length")
                    if radius_text and length_text:
                        local_points = cylinder_points(radius_text, length_text)
                elif sphere is not None:
                    radius_text = sphere.findtext("radius")
                    if radius_text:
                        local_points = sphere_points(radius_text)
                else:
                    name = collision.attrib.get("name", "<unnamed>")
                    warnings.append(
                        f"Skipping unsupported geometry in model '{model_name}', collision '{name}'"
                    )
                    continue

                if not local_points:
                    continue

                world_points = transform_points(local_points, coll_t, coll_r)
                z_values = [p[2] for p in world_points]
                if max(z_values) < min_z or min(z_values) > max_obstacle_z:
                    continue

                polygon = convex_hull([(p[0], p[1]) for p in world_points])
                if len(polygon) >= 3:
                    polygons.append(polygon)

    return polygons, warnings


def compute_bounds(polygons: list[list[tuple[float, float]]], resolution: float, margin: float) -> tuple[float, float, int, int]:
    min_x = min(point[0] for polygon in polygons for point in polygon)
    max_x = max(point[0] for polygon in polygons for point in polygon)
    min_y = min(point[1] for polygon in polygons for point in polygon)
    max_y = max(point[1] for polygon in polygons for point in polygon)

    origin_x = math.floor((min_x - margin) / resolution) * resolution
    origin_y = math.floor((min_y - margin) / resolution) * resolution
    max_x = math.ceil((max_x + margin) / resolution) * resolution
    max_y = math.ceil((max_y + margin) / resolution) * resolution

    width = int(round((max_x - origin_x) / resolution))
    height = int(round((max_y - origin_y) / resolution))
    return origin_x, origin_y, width, height


def flood_fill_interior(
    occupied: list[list[bool]],
    seed_gx: int,
    seed_gy: int,
) -> list[list[bool]]:
    """Return a boolean grid marking cells reachable from the seed without
    crossing occupied cells.  Used to distinguish interior free space from
    exterior (which becomes black in the final map)."""
    height = len(occupied)
    width = len(occupied[0])

    interior: list[list[bool]] = [[False] * width for _ in range(height)]
    if occupied[seed_gy][seed_gx]:
        return interior  # seed is inside a wall — nothing to fill

    from collections import deque
    queue: deque[tuple[int, int]] = deque()
    queue.append((seed_gy, seed_gx))
    interior[seed_gy][seed_gx] = True

    while queue:
        gy, gx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < height and 0 <= nx < width and not interior[ny][nx] and not occupied[ny][nx]:
                interior[ny][nx] = True
                queue.append((ny, nx))

    return interior


def write_pgm(
    path: Path,
    occupied: list[list[bool]],
    interior: list[list[bool]] | None = None,
    border_px: int = 4,
) -> None:
    height = len(occupied)
    width = len(occupied[0])
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    pixels = bytearray()

    # PGM stores rows top-to-bottom, while the occupancy origin is bottom-left.
    # image_row 0 corresponds to grid row (height-1) (top of image = highest Y).
    for img_row in range(height):
        gy = height - 1 - img_row
        for gx in range(width):
            # thin white border at the very edge of the image
            if (
                border_px > 0
                and (img_row < border_px or img_row >= height - border_px
                     or gx < border_px or gx >= width - border_px)
            ):
                pixels.append(254)  # white border
            elif occupied[gy][gx]:
                pixels.append(0)    # wall / obstacle → black
            elif interior is not None and not interior[gy][gx]:
                pixels.append(0)    # outside the house → black
            else:
                pixels.append(254)  # free interior → white

    path.write_bytes(header + pixels)


def write_yaml(path: Path, pgm_name: str, resolution: float, origin_x: float, origin_y: float) -> None:
    contents = (
        f"image: {pgm_name}\n"
        "mode: trinary\n"
        f"resolution: {resolution:.3f}\n"
        f"origin: [{origin_x:.3f}, {origin_y:.3f}, 0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    path.write_text(contents, encoding="ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a GT occupancy map from Gazebo SDF collisions."
    )
    parser.add_argument(
        "--world",
        type=Path,
        default=Path("src/sensor_fusion_desc/worlds/house3.world"),
        help="Path to the Gazebo world SDF.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("src/sensor_fusion_desc/models"),
        help="Directory containing model folders.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("maps/house3_gt"),
        help="Output prefix, without extension.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Map resolution in meters/cell.",
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=0.01,
        help="Ignore geometry entirely below this world Z.",
    )
    parser.add_argument(
        "--max-obstacle-z",
        type=float,
        default=0.25,
        help=(
            "Project geometry that occupies the low obstacle band up to this Z. "
            "This keeps table legs but drops most tabletops."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.10,
        help="Extra map border in meters.",
    )
    parser.add_argument(
        "--interior-seed-x",
        type=float,
        default=1.2265,
        help="World X of a point known to be inside the house (robot spawn default).",
    )
    parser.add_argument(
        "--interior-seed-y",
        type=float,
        default=-0.9923,
        help="World Y of a point known to be inside the house (robot spawn default).",
    )
    parser.add_argument(
        "--fill-exterior",
        action="store_true",
        default=True,
        help="Mark everything outside the house walls as black (default: on).",
    )
    parser.add_argument(
        "--no-fill-exterior",
        dest="fill_exterior",
        action="store_false",
        help="Leave exterior cells as white instead of black.",
    )
    parser.add_argument(
        "--border-px",
        type=int,
        default=4,
        help="Width in pixels of the white border at the image edge (0 to disable).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    polygons, warnings = collect_polygons(
        world_path=args.world,
        models_dir=args.models_dir,
        min_z=args.min_z,
        max_obstacle_z=args.max_obstacle_z,
    )

    if not polygons:
        raise RuntimeError("No obstacle polygons were generated. Check the world path and Z thresholds.")

    origin_x, origin_y, width, height = compute_bounds(
        polygons,
        resolution=args.resolution,
        margin=args.margin,
    )

    occupied = [[False for _ in range(width)] for _ in range(height)]
    for polygon in polygons:
        rasterize_polygon(
            polygon,
            occupied,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=args.resolution,
        )

    interior = None
    if args.fill_exterior:
        seed_gx = int(math.floor((args.interior_seed_x - origin_x) / args.resolution))
        seed_gy = int(math.floor((args.interior_seed_y - origin_y) / args.resolution))
        seed_gx = max(0, min(width - 1, seed_gx))
        seed_gy = max(0, min(height - 1, seed_gy))
        interior = flood_fill_interior(occupied, seed_gx, seed_gy)
        if not any(interior[gy][gx] for gy in range(height) for gx in range(width)):
            print("Warning: interior seed landed on an occupied cell — exterior fill skipped.")
            interior = None

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pgm_path = output_prefix.with_suffix(".pgm")
    yaml_path = output_prefix.with_suffix(".yaml")

    write_pgm(pgm_path, occupied, interior=interior, border_px=args.border_px)
    write_yaml(yaml_path, pgm_path.name, args.resolution, origin_x, origin_y)

    occupied_cells = sum(1 for row in occupied for cell in row if cell)
    print(f"Wrote {pgm_path}")
    print(f"Wrote {yaml_path}")
    print(
        f"Map info: width={width}, height={height}, resolution={args.resolution:.3f}, "
        f"origin=({origin_x:.3f}, {origin_y:.3f}), occupied_cells={occupied_cells}"
    )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
