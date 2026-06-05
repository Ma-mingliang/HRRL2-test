"""Runtime verification for path assets."""

from __future__ import annotations

import os

import pybullet as p
import pybullet_data

import env


def main() -> None:
    required_files = [
        env._asset_path("s_line_path.obj"),
        env._asset_path("complex_path.obj"),
        env._asset_path("terrain.urdf"),
        env._asset_path("generated_terrain_lab.urdf"),
        env._asset_path("single_turn_90_path.obj"),
        env._asset_path("single_turn_wide_path.obj"),
        env._asset_path("single_turn_exit_path.obj"),
    ]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for name in ("s_line", "complex", "single_turn_90", "single_turn_wide", "single_turn_exit"):
            p.resetSimulation()
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            start_pos = env._load_path_scene(name)
            print("LOAD_OK", name, start_pos)
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    main()
