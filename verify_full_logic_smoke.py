"""Full smoke verification for the main environments and scripts."""

from __future__ import annotations

import numpy as np

import env


class DummyModel:
    def __init__(self, action):
        self.action = np.array(action, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.action.copy(), None


def check_stage1():
    instance = env.Attitude_control_stage1(render=False, action_repeat=1)
    try:
        obs, info = instance.reset()
        assert instance.action_space.shape == (1,)
        action = np.zeros(1, dtype=np.float32)
        result = instance.step(action)
        print("STAGE1_OK", obs.shape, len(result))
    finally:
        instance.close()


def check_stage3(path_type: str):
    instance = env.Path_tracking_stage3(
        render=False,
        agent_lqr_path="dummy_stage1_model.zip",
        path_type=path_type,
        action_repeat=1,
        heading_offset_reset_mode="legacy",
    )
    instance.agent_lqr = DummyModel([0.0, 0.0, 0.0])
    try:
        obs, info = instance.reset()
        action = np.zeros(2, dtype=np.float32)
        result = instance.step(action)
        print("STAGE3_OK", path_type, obs.shape, len(result))
    finally:
        instance.close()


def main():
    check_stage1()
    for path_type in ("s_line", "complex", "single_turn_90", "single_turn_wide", "single_turn_exit"):
        check_stage3(path_type)


if __name__ == "__main__":
    main()
