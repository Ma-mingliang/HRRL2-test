"""Generate OBJ meshes for the three single-turn path variants.

The meshes are visual-only ribbons used by PyBullet path rendering.
They follow the same geometry used by env.py:
1. Entry straight segment along +Y from x=0.
2. Quarter-circle left turn centered at (radius, straight_len).
3. Exit straight segment along +X at y=straight_len + radius.
"""

from __future__ import annotations

import math
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "3D"
PATH_SPECS = {
    "single_turn_90": {"straight_len": 20.0, "radius": 10.0, "exit_len": 25.0},
    "single_turn_wide": {"straight_len": 25.0, "radius": 20.0, "exit_len": 35.0},
    "single_turn_exit": {"straight_len": 20.0, "radius": 8.0, "exit_len": 18.0},
}

PATH_WIDTH = 1.4
PATH_HEIGHT = 0.10
ARC_SEGMENTS = 32


class ObjMesh:
    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []

    def add_vertex(self, xyz: tuple[float, float, float]) -> int:
        self.vertices.append(xyz)
        return len(self.vertices)

    def add_face(self, a: int, b: int, c: int) -> None:
        self.faces.append((a, b, c))

    def add_quad(self, a: int, b: int, c: int, d: int) -> None:
        self.add_face(a, b, c)
        self.add_face(a, c, d)

    def add_box(self, min_x: float, max_x: float, min_y: float, max_y: float, z0: float, z1: float) -> None:
        v1 = self.add_vertex((min_x, min_y, z0))
        v2 = self.add_vertex((max_x, min_y, z0))
        v3 = self.add_vertex((max_x, max_y, z0))
        v4 = self.add_vertex((min_x, max_y, z0))
        v5 = self.add_vertex((min_x, min_y, z1))
        v6 = self.add_vertex((max_x, min_y, z1))
        v7 = self.add_vertex((max_x, max_y, z1))
        v8 = self.add_vertex((min_x, max_y, z1))

        self.add_quad(v1, v2, v3, v4)
        self.add_quad(v5, v8, v7, v6)
        self.add_quad(v1, v5, v6, v2)
        self.add_quad(v2, v6, v7, v3)
        self.add_quad(v3, v7, v8, v4)
        self.add_quad(v4, v8, v5, v1)

    def add_arc_strip(
        self,
        center_x: float,
        center_y: float,
        inner_r: float,
        outer_r: float,
        angle_start: float,
        angle_end: float,
        segments: int,
        z0: float,
        z1: float,
    ) -> None:
        rings: list[tuple[int, int, int, int]] = []
        for i in range(segments + 1):
            t = i / segments
            angle = angle_start + (angle_end - angle_start) * t
            ca = math.cos(angle)
            sa = math.sin(angle)

            outer_b = self.add_vertex((center_x + outer_r * ca, center_y + outer_r * sa, z0))
            outer_t = self.add_vertex((center_x + outer_r * ca, center_y + outer_r * sa, z1))
            inner_b = self.add_vertex((center_x + inner_r * ca, center_y + inner_r * sa, z0))
            inner_t = self.add_vertex((center_x + inner_r * ca, center_y + inner_r * sa, z1))
            rings.append((outer_b, outer_t, inner_b, inner_t))

        for current_ring, next_ring in zip(rings[:-1], rings[1:]):
            cob, cot, cib, cit = current_ring
            nob, not_, nib, nit = next_ring

            self.add_quad(cot, not_, nit, cit)
            self.add_quad(cob, cib, nib, nob)
            self.add_quad(cob, nob, not_, cot)
            self.add_quad(cib, cit, nit, nib)

        sob, sot, sib, sit = rings[0]
        eob, eot, eib, eit = rings[-1]
        self.add_quad(sob, sot, sit, sib)
        self.add_quad(eob, eib, eit, eot)

    def to_obj(self, name: str) -> str:
        lines = [
            f"# {name}",
            f"# Generated with {len(self.vertices)} vertices and {len(self.faces)} faces",
            "",
        ]
        for x, y, z in self.vertices:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        lines.append("")
        for a, b, c in self.faces:
            lines.append(f"f {a} {b} {c}")
        lines.append("")
        return "\n".join(lines)


def build_single_turn_mesh(name: str, straight_len: float, radius: float, exit_len: float) -> str:
    mesh = ObjMesh()
    half_width = PATH_WIDTH / 2.0
    exit_y = straight_len + radius

    mesh.add_box(-half_width, half_width, 0.0, straight_len, 0.0, PATH_HEIGHT)
    mesh.add_arc_strip(
        center_x=radius,
        center_y=straight_len,
        inner_r=radius - half_width,
        outer_r=radius + half_width,
        angle_start=math.pi,
        angle_end=math.pi / 2.0,
        segments=ARC_SEGMENTS,
        z0=0.0,
        z1=PATH_HEIGHT,
    )
    mesh.add_box(radius, radius + exit_len, exit_y - half_width, exit_y + half_width, 0.0, PATH_HEIGHT)
    return mesh.to_obj(name)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path_name, spec in PATH_SPECS.items():
        obj_text = build_single_turn_mesh(path_name, **spec)
        output_path = OUTPUT_DIR / f"{path_name}_path.obj"
        output_path.write_text(obj_text, encoding="utf-8")
        print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
