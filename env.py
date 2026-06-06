"""
自行车自平衡与路径跟踪控制系统

包含：
- 第一阶段：纯强化学习平衡控制器
- 第三阶段：自适应Stanley控制器
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import random
import numpy as np
from stable_baselines3 import TD3
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(PROJECT_ROOT, "3D")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
BIKE_URDF_PATH = os.path.join(ASSET_DIR, "bike", "urdf", "bike.urdf")


def _asset_path(*parts):
    return os.path.join(ASSET_DIR, *parts)


def _model_path(*parts):
    return os.path.join(MODEL_DIR, *parts)


def _configure_console_output():
    """在 Windows 控制台下尽量切到 UTF-8，避免 emoji 日志触发编码错误。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_console_output()

# ==================== 运动学模型辅助函数 ====================

def newton_iteration(theta, v):
    """使用牛顿迭代法求解车把角"""
    l = 1.389
    l1 = 0.7
    g = 9.8
    
    x = 0
    
    for i in range(20):
        f = math.tan(x) * (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)) + 0.4407) - \
            ((l / v)**2) * g * math.tan(theta)
        
        diff_f = (1 + (math.tan(x)**2)) * (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)) + 0.4407) + \
                 math.tan(x) * ((l1**2) * math.tan(x) * (1 + (math.tan(x)**2))) / \
                 (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)))
        
        x = x - f / diff_f

    x = math.atan((math.tan(x) * math.cos(theta)) / 
                  (math.cos(0.12222) + math.tan(x) * math.sin(0.12222) * math.sin(theta)))

    return x


def steady_state_calculation(x, v):
    """计算稳态倾斜角"""
    l = 1.489
    l1 = 0.7
    g = 9.8
    
    theta = (math.tan(x) * (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)) + 0.4407)) / \
            (((l / v)**2) * g)
    theta = math.atan(theta)
    
    return theta




# ==================== 路径跟踪环境辅助函数 ====================

SINGLE_TURN_PATH_SPECS = {
    "single_turn_90": {
        "display_name": "单弯90°路径",
        "straight_len": 20.0,
        "radius": 10.0,
        "exit_len": 25.0,
        "baseline_lateral": 0.40,
        "baseline_course": 0.09,
        "max_step_actions": 840,
        "early_failure_threshold": 25,
    },
    "single_turn_wide": {
        "display_name": "单弯大半径路径",
        "straight_len": 25.0,
        "radius": 20.0,
        "exit_len": 35.0,
        "baseline_lateral": 0.35,
        "baseline_course": 0.08,
        "max_step_actions": 1050,
        "early_failure_threshold": 35,
    },
    "single_turn_exit": {
        "display_name": "单弯短出口路径",
        "straight_len": 20.0,
        "radius": 8.0,
        "exit_len": 18.0,
        "baseline_lateral": 0.48,
        "baseline_course": 0.11,
        "max_step_actions": 760,
        "early_failure_threshold": 28,
    },
}

SINGLE_TURN_PATH_TYPES = tuple(SINGLE_TURN_PATH_SPECS.keys())
SINGLE_TURN_PATH_MESH_FILES = {
    "single_turn_90": _asset_path("single_turn_90_path.obj"),
    "single_turn_wide": _asset_path("single_turn_wide_path.obj"),
    "single_turn_exit": _asset_path("single_turn_exit_path.obj"),
}

COMPLEX_SEGMENT_1_LENGTH = 55.0
COMPLEX_SEGMENT_2_RADIUS = 15.0
COMPLEX_SEGMENT_2_LENGTH = COMPLEX_SEGMENT_2_RADIUS * math.pi / 2.0
COMPLEX_SEGMENT_3_LENGTH = 20.0
COMPLEX_SEGMENT_4_RADIUS = 15.0
COMPLEX_SEGMENT_4_LENGTH = COMPLEX_SEGMENT_4_RADIUS * math.pi
COMPLEX_SEGMENT_5_RADIUS = 12.0
COMPLEX_SEGMENT_5_LENGTH = COMPLEX_SEGMENT_5_RADIUS * math.pi
COMPLEX_SEGMENT_6_RADIUS = 8.0
COMPLEX_SEGMENT_6_GOAL_ANGLE = math.pi / 2.0
COMPLEX_SEGMENT_6_GOAL_LENGTH = COMPLEX_SEGMENT_6_RADIUS * COMPLEX_SEGMENT_6_GOAL_ANGLE
COMPLEX_GOAL_PATH_LENGTH = (
    COMPLEX_SEGMENT_1_LENGTH +
    COMPLEX_SEGMENT_2_LENGTH +
    COMPLEX_SEGMENT_3_LENGTH +
    COMPLEX_SEGMENT_4_LENGTH +
    COMPLEX_SEGMENT_5_LENGTH +
    COMPLEX_SEGMENT_6_GOAL_LENGTH
)
COMPLEX_COMPLETION_Y = 42.0
COMPLEX_COMPLETION_X_MIN = 0.0
COMPLEX_COMPLETION_X_MAX = 28.0


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _get_complex_reset_profile(epoch_num):
    """复杂路径使用温和课程，不再随 episode 指数爆炸。"""
    if epoch_num < 40:
        return 0.15, 0.40
    if epoch_num < 120:
        return 0.35, 0.90
    if epoch_num < 240:
        return 0.65, 1.80
    return 0.90, 2.80


def _get_path_tracking_config(path_type, action_repeat):
    if path_type == "s_line":
        return {
            "display_name": "S型路径",
            "max_step_num": 760 * action_repeat,
            "early_failure_threshold": 20,
            "completion_y": 61.5,
            "completion_x_min": -30.0,
            "completion_x_max": 30.0,
            "baseline_lateral": 0.50,
            "baseline_course": 0.11,
            "completion_desc": "y>=61.5 and -30<x<30",
        }
    if path_type == "complex":
        return {
            "display_name": "复杂路径",
            "max_step_num": 2000 * action_repeat,
            "early_failure_threshold": 50,
            "completion_y": COMPLEX_COMPLETION_Y,
            "completion_x_min": COMPLEX_COMPLETION_X_MIN,
            "completion_x_max": COMPLEX_COMPLETION_X_MAX,
            "baseline_lateral": 0.50,
            "baseline_course": 0.11,
            "completion_desc": "y>=42 and 0<=x<=28 (复杂路径上方目标区)",
        }
    if path_type in SINGLE_TURN_PATH_SPECS:
        spec = SINGLE_TURN_PATH_SPECS[path_type]
        end_y = spec["straight_len"] + spec["radius"]
        end_x = spec["radius"] + spec["exit_len"]
        return {
            "display_name": spec["display_name"],
            "max_step_num": spec["max_step_actions"] * action_repeat,
            "early_failure_threshold": spec["early_failure_threshold"],
            "completion_y": end_y,
            "completion_x_min": end_x - 4.0,
            "completion_x_max": end_x + 8.0,
            "completion_y_min": end_y - 4.0,
            "completion_y_max": end_y + 4.0,
            "baseline_lateral": spec["baseline_lateral"],
            "baseline_course": spec["baseline_course"],
            "completion_desc": (
                f"{end_x - 4.0:.1f}<=x<={end_x + 8.0:.1f} and "
                f"{end_y - 4.0:.1f}<=y<={end_y + 4.0:.1f}"
            ),
        }
    raise ValueError(f"未知路径类型: {path_type}")


def _apply_path_tracking_config(holder, path_cfg):
    holder.max_step_num = path_cfg["max_step_num"]
    holder.early_failure_threshold = path_cfg["early_failure_threshold"]
    holder.completion_y = path_cfg["completion_y"]
    holder.completion_x_min = path_cfg["completion_x_min"]
    holder.completion_x_max = path_cfg["completion_x_max"]
    holder.completion_y_min = path_cfg.get("completion_y_min", -float("inf"))
    holder.completion_y_max = path_cfg.get("completion_y_max", float("inf"))
    holder.baseline_lateral = path_cfg["baseline_lateral"]
    holder.baseline_course = path_cfg["baseline_course"]


def _draw_single_turn_path(path_type):
    spec = SINGLE_TURN_PATH_SPECS[path_type]
    straight_len = spec["straight_len"]
    radius = spec["radius"]
    exit_len = spec["exit_len"]
    center = np.array([radius, straight_len])

    points = []
    for y in np.linspace(0.0, straight_len, 60):
        points.append([0.0, float(y), 0.03])
    for angle in np.linspace(math.pi, math.pi / 2, 80):
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append([float(x), float(y), 0.03])
    end_y = straight_len + radius
    for x in np.linspace(radius, radius + exit_len, 60):
        points.append([float(x), float(end_y), 0.03])

    for p0, p1 in zip(points[:-1], points[1:]):
        p.addUserDebugLine(p0, p1, [1.0, 0.2, 0.0], lineWidth=4, lifeTime=0)


def _load_path_scene(path_type):
    """加载路径场景资源。"""
    if path_type == "s_line":
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=_asset_path("s_line_path.obj"),
            rgbaColor=[255/255, 50/255, 0/255, 0.5],
            specularColor=[0.4, 0.4, 0.8],
            visualFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1]
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id)
        p.loadURDF(_asset_path("terrain.urdf"), [0, 0, 0], globalScaling=5)
        start_pos = [0, 0, 0.7]
    elif path_type == "complex":
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=_asset_path("complex_path.obj"),
            rgbaColor=[255/255, 50/255, 0/255, 0.5],
            specularColor=[0.4, 0.4, 0.8],
            visualFramePosition=[-1, -1, 0],
            meshScale=[0.001, 0.001, 0.001]
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id)
        p.loadURDF(_asset_path("generated_terrain_lab.urdf"), globalScaling=2)
        start_pos = [0, 0, 0.78]
    elif path_type in SINGLE_TURN_PATH_TYPES:
        mesh_path = SINGLE_TURN_PATH_MESH_FILES[path_type]
        if os.path.exists(mesh_path):
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=mesh_path,
                rgbaColor=[255/255, 50/255, 0/255, 0.55],
                specularColor=[0.4, 0.4, 0.8],
                visualFramePosition=[0, 0, 0],
                meshScale=[1, 1, 1]
            )
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id)
        p.loadURDF(_asset_path("terrain.urdf"), [0, 0, 0], globalScaling=5)
        if not os.path.exists(mesh_path):
            _draw_single_turn_path(path_type)
        start_pos = [0, 0, 0.78]
    else:
        raise ValueError(f"未知路径类型: {path_type}")
    return start_pos

def _get_reset_start_pos(path_type, epoch_num, init_lateral_offset=0.0, use_random_start=True):
    if path_type == "s_line":
        return [init_lateral_offset, 0, 0.7]
    if path_type == "complex":
        if use_random_start and abs(init_lateral_offset) < 1e-12:
            sigma, clip_limit = _get_complex_reset_profile(epoch_num)
            y = np.clip(np.random.normal(loc=0.0, scale=sigma, size=None), -clip_limit, clip_limit)
        else:
            y = init_lateral_offset
        return [0, y, 0.78]
    if path_type in SINGLE_TURN_PATH_TYPES:
        return [init_lateral_offset, 0, 0.78]
    raise ValueError(f"未知路径类型: {path_type}")


def _get_reset_start_yaw(path_type, heading_offset_reset_mode="legacy", init_heading_offset=0.0):
    """统一reset阶段初始航向逻辑。

    - legacy: 原方法。保持原来的固定初始朝向，不额外引入航向偏置。
    - offset: 新方法。在路径默认起始朝向基础上叠加 init_heading_offset。

    当前这几类路径的默认起始朝向都可视为 yaw_base = 0.0，因此新方法等价于
    yaw = yaw_base + init_heading_offset。保留独立函数后，后续如果某条路径需要非零
    的默认起始朝向，也只需要在这里统一修改。
    """
    yaw_base = 0.0
    if heading_offset_reset_mode == "legacy":
        return yaw_base
    if heading_offset_reset_mode == "offset":
        return yaw_base + init_heading_offset
    raise ValueError(
        'heading_offset_reset_mode 只能是 "legacy" 或 "offset"。'
        '其中 "legacy"=原方法，"offset"=新方法。'
    )


def _compute_heading_error(bike_direction_vector, ref_tangent):
    bike_heading = math.atan2(bike_direction_vector[1], bike_direction_vector[0])
    ref_heading = math.atan2(ref_tangent[1], ref_tangent[0])
    return _wrap_angle(bike_heading - ref_heading)


def _compute_single_turn_tracking_errors(path_type, forward_wheel_point_x, forward_wheel_point_y, bike_direction_vector):
    spec = SINGLE_TURN_PATH_SPECS[path_type]
    straight_len = spec["straight_len"]
    radius = spec["radius"]
    exit_len = spec["exit_len"]
    exit_y = straight_len + radius
    center = np.array([radius, straight_len], dtype=float)
    pos = np.array([forward_wheel_point_x, forward_wheel_point_y], dtype=float)

    candidates = []

    # segment 1: 垂直直线 x=0, y:[0, straight_len]
    line_y = np.clip(pos[1], 0.0, straight_len)
    ref_point_1 = np.array([0.0, line_y])
    tangent_1 = np.array([0.0, 1.0])
    left_normal_1 = np.array([-1.0, 0.0])
    lateral_1 = float(np.dot(pos - ref_point_1, left_normal_1))
    dist_1 = abs(lateral_1) + 0.05 * max(0.0, pos[1] - straight_len) + 0.05 * max(0.0, -pos[1])
    course_1 = _compute_heading_error(bike_direction_vector, tangent_1)
    candidates.append((dist_1, lateral_1, course_1, 0.0, 1))

    # segment 2: 1/4 圆弧，中心 (r, straight_len), 从 pi -> pi/2
    rel = pos - center
    theta = math.atan2(rel[1], rel[0])
    theta = float(np.clip(theta, math.pi / 2, math.pi))
    ref_point_2 = center + radius * np.array([math.cos(theta), math.sin(theta)])
    tangent_2 = np.array([math.sin(theta), -math.cos(theta)])  # 保持起点向 +y，终点向 +x
    tangent_norm = np.linalg.norm(tangent_2)
    if tangent_norm < 1e-9:
        tangent_2 = np.array([1.0, 0.0])
    else:
        tangent_2 = tangent_2 / tangent_norm
    left_normal_2 = np.array([-tangent_2[1], tangent_2[0]])
    lateral_2 = float(np.dot(pos - ref_point_2, left_normal_2))
    dist_2 = abs(np.linalg.norm(rel) - radius)
    course_2 = _compute_heading_error(bike_direction_vector, tangent_2)
    candidates.append((dist_2, lateral_2, course_2, -1.0 / max(radius, 1e-6), 2))

    # segment 3: 水平直线 y=exit_y, x:[radius, radius+exit_len]
    line_x = np.clip(pos[0], radius, radius + exit_len)
    ref_point_3 = np.array([line_x, exit_y])
    tangent_3 = np.array([1.0, 0.0])
    left_normal_3 = np.array([0.0, 1.0])
    lateral_3 = float(np.dot(pos - ref_point_3, left_normal_3))
    dist_3 = abs(lateral_3) + 0.05 * max(0.0, radius - pos[0]) + 0.05 * max(0.0, pos[0] - (radius + exit_len))
    course_3 = _compute_heading_error(bike_direction_vector, tangent_3)
    candidates.append((dist_3, lateral_3, course_3, 0.0, 3))

    _, lateral_error, course_error_angle, curvature, segment_id = min(candidates, key=lambda item: item[0])
    return lateral_error, course_error_angle, curvature, segment_id


def _compute_single_turn_progress(path_type, x_s, y_s):
    progress, total_len = _compute_single_turn_progress_distance(path_type, x_s, y_s)
    spec = SINGLE_TURN_PATH_SPECS[path_type]
    exit_y = spec["straight_len"] + spec["radius"]
    end_x = spec["radius"] + spec["exit_len"]
    progress_percent = float(np.clip(progress / total_len, 0.0, 1.0) * 100.0)
    path_completed = (
        (end_x - 4.0) <= x_s <= (end_x + 8.0) and
        (exit_y - 4.0) <= y_s <= (exit_y + 4.0)
    )
    return progress_percent, path_completed


def _infer_complex_segment_id(x_s, y_s, segment_id):
    if segment_id:
        return segment_id
    if y_s <= 6.0 and x_s <= 55.0:
        return 1
    if x_s >= 62.0 and y_s <= 18.0:
        return 2
    if x_s >= 62.0 and y_s <= 38.0:
        return 3
    if x_s >= 36.0 and y_s >= 28.0:
        return 4
    if -2.0 <= x_s <= 20.0 and y_s >= 30.0:
        return 6
    if x_s >= 14.0 and y_s >= 20.0:
        return 5
    return 0


def _compute_complex_progress_distance(x_s, y_s, segment_id=0):
    segment_id = _infer_complex_segment_id(x_s, y_s, segment_id)

    if segment_id == 1:
        progress = np.clip(x_s, 0.0, COMPLEX_SEGMENT_1_LENGTH)
    elif segment_id == 2:
        theta = math.atan2(y_s - 15.0, x_s - 55.0)
        theta = float(np.clip(theta, -math.pi / 2.0, 0.0))
        progress = COMPLEX_SEGMENT_1_LENGTH + COMPLEX_SEGMENT_2_RADIUS * (theta + math.pi / 2.0)
    elif segment_id == 3:
        progress = (
            COMPLEX_SEGMENT_1_LENGTH +
            COMPLEX_SEGMENT_2_LENGTH +
            np.clip(y_s - 15.0, 0.0, COMPLEX_SEGMENT_3_LENGTH)
        )
    elif segment_id == 4:
        theta = math.atan2(y_s - 35.0, x_s - 55.0)
        theta = float(np.clip(theta, 0.0, math.pi))
        progress = (
            COMPLEX_SEGMENT_1_LENGTH +
            COMPLEX_SEGMENT_2_LENGTH +
            COMPLEX_SEGMENT_3_LENGTH +
            COMPLEX_SEGMENT_4_RADIUS * theta
        )
    elif segment_id == 5:
        theta = math.atan2(y_s - 35.0, x_s - 28.0)
        theta = float(np.clip(theta, -math.pi, 0.0))
        progress = (
            COMPLEX_SEGMENT_1_LENGTH +
            COMPLEX_SEGMENT_2_LENGTH +
            COMPLEX_SEGMENT_3_LENGTH +
            COMPLEX_SEGMENT_4_LENGTH +
            COMPLEX_SEGMENT_5_RADIUS * abs(theta)
        )
    elif segment_id == 6:
        theta = math.atan2(y_s - 35.0, x_s - 8.0)
        theta = float(np.clip(theta, 0.0, COMPLEX_SEGMENT_6_GOAL_ANGLE))
        progress = (
            COMPLEX_SEGMENT_1_LENGTH +
            COMPLEX_SEGMENT_2_LENGTH +
            COMPLEX_SEGMENT_3_LENGTH +
            COMPLEX_SEGMENT_4_LENGTH +
            COMPLEX_SEGMENT_5_LENGTH +
            COMPLEX_SEGMENT_6_RADIUS * theta
        )
    elif segment_id >= 7:
        progress = COMPLEX_GOAL_PATH_LENGTH
    else:
        progress = 0.0

    return float(np.clip(progress, 0.0, COMPLEX_GOAL_PATH_LENGTH)), COMPLEX_GOAL_PATH_LENGTH


def _update_complex_progress_tracker(holder, x_s, y_s):
    progress, total_len = _compute_complex_progress_distance(
        x_s,
        y_s,
        getattr(holder, "current_path_segment", 0),
    )
    best_progress = float(getattr(holder, "_complex_best_progress", 0.0))
    progress_delta = max(0.0, progress - best_progress)
    holder._complex_last_progress = progress
    holder._complex_best_progress = max(best_progress, progress)
    holder._complex_total_length = total_len
    return progress_delta, progress, total_len


def _compute_path_progress_and_completion(
    path_type,
    x_s,
    y_s,
    completion_y,
    completion_x_min,
    completion_x_max,
    completion_y_min=-float("inf"),
    completion_y_max=float("inf"),
    segment_id=0,
):
    if path_type == "s_line":
        progress_percent = min(100.0, (y_s / 60.0) * 100)
        path_completed = (
            y_s >= completion_y and
            completion_x_min <= x_s <= completion_x_max
        )
        return progress_percent, path_completed

    if path_type in SINGLE_TURN_PATH_TYPES:
        return _compute_single_turn_progress(path_type, x_s, y_s)

    progress_distance, total_len = _compute_complex_progress_distance(x_s, y_s, segment_id)
    progress_percent = float(np.clip(progress_distance / max(total_len, 1e-6), 0.0, 1.0) * 100.0)

    path_completed = (
        y_s >= completion_y and
        completion_x_min <= x_s <= completion_x_max
    )
    return progress_percent, path_completed


def _compute_single_turn_path_length(path_type):
    spec = SINGLE_TURN_PATH_SPECS[path_type]
    return float(spec["straight_len"] + spec["radius"] * math.pi / 2 + spec["exit_len"])


def _compute_single_turn_progress_distance(path_type, x_s, y_s):
    spec = SINGLE_TURN_PATH_SPECS[path_type]
    straight_len = spec["straight_len"]
    radius = spec["radius"]
    exit_len = spec["exit_len"]
    exit_y = straight_len + radius
    center = np.array([radius, straight_len], dtype=float)
    arc_len = radius * math.pi / 2
    total_len = _compute_single_turn_path_length(path_type)

    if y_s <= straight_len and x_s <= radius * 0.5:
        progress = np.clip(y_s, 0.0, straight_len)
    elif y_s < exit_y + 1.0 and x_s < radius + 1.0:
        theta = math.atan2(y_s - center[1], x_s - center[0])
        theta = float(np.clip(theta, math.pi / 2, math.pi))
        progress = straight_len + radius * (math.pi - theta)
    else:
        progress = straight_len + arc_len + np.clip(x_s - radius, 0.0, exit_len)

    return float(np.clip(progress, 0.0, total_len)), total_len


def _update_single_turn_progress_tracker(holder, path_type, x_s, y_s):
    progress, total_len = _compute_single_turn_progress_distance(path_type, x_s, y_s)
    last_progress = float(getattr(holder, "_single_turn_last_progress", 0.0))
    progress_delta = progress - last_progress
    holder._single_turn_last_progress = progress
    holder._single_turn_total_length = total_len
    return progress_delta, progress, total_len


def _reset_path_progress_trackers(holder, path_type):
    holder.current_path_segment = 0
    holder.last_x_position = 0.0
    holder.last_y_position = 0.0
    holder.total_forward_distance = 0.0
    holder._complex_last_progress = 0.0
    holder._complex_best_progress = 0.0
    holder._complex_total_length = COMPLEX_GOAL_PATH_LENGTH if path_type == "complex" else 0.0
    holder._single_turn_last_progress = 0.0
    holder._single_turn_total_length = (
        _compute_single_turn_path_length(path_type)
        if path_type in SINGLE_TURN_PATH_TYPES else 0.0
    )


def _calculate_complex_reward_core(
    state_last_raw,
    state_raw,
    baseline_lateral,
    baseline_course,
    progress_delta,
    progress_percent,
    segment_id,
):
    lateral_error = abs(state_raw[0])
    lateral_error_last = abs(state_last_raw[0])
    course_error = abs(state_raw[1])
    angular_velocity = abs(state_raw[4])
    velocity = abs(state_raw[2])

    if lateral_error < baseline_lateral * 0.5:
        precision_reward = 3.0
    elif lateral_error < baseline_lateral * 0.8:
        precision_reward = 1.8
    elif lateral_error < baseline_lateral:
        precision_reward = 0.8
    elif lateral_error < baseline_lateral * 1.25:
        precision_reward = -1.2
    elif lateral_error < baseline_lateral * 1.6:
        precision_reward = -3.0
    else:
        precision_reward = -6.0

    error_reduction = lateral_error_last - lateral_error
    improvement_reward = 8.0 * float(np.clip(error_reduction, -0.12, 0.12))

    if lateral_error < baseline_lateral * 1.15:
        if course_error < baseline_course * 0.8:
            heading_reward = 1.5
        elif course_error < baseline_course * 1.1:
            heading_reward = 0.5
        elif course_error < baseline_course * 1.5:
            heading_reward = -0.8
        else:
            heading_reward = -2.0
    else:
        heading_reward = -min(1.5, 1.8 * max(0.0, course_error - baseline_course))

    if angular_velocity < 0.25:
        smoothness_reward = 0.4
    elif angular_velocity < 0.45:
        smoothness_reward = 0.0
    elif angular_velocity < 0.75:
        smoothness_reward = -0.5
    else:
        smoothness_reward = -1.4

    progress_reward = 10.0 * float(np.clip(progress_delta, 0.0, 1.2))
    route_reward = 0.8 if segment_id > 0 else -2.5

    goal_lane_bonus = 0.0
    if progress_percent >= 80.0 and lateral_error < baseline_lateral:
        goal_lane_bonus += 1.0
    if progress_percent >= 92.0 and course_error < baseline_course * 1.2:
        goal_lane_bonus += 1.5

    if velocity < 2.0:
        efficiency_term = -0.8
    elif velocity > 4.2 and lateral_error > baseline_lateral * 0.9:
        efficiency_term = -1.0
    else:
        efficiency_term = 0.0

    time_penalty = -0.15

    return (
        precision_reward +
        improvement_reward +
        heading_reward +
        smoothness_reward +
        progress_reward +
        route_reward +
        goal_lane_bonus +
        efficiency_term +
        time_penalty
    )


def _calculate_single_turn_reward_core(
    state_last_raw,
    state_raw,
    baseline_lateral,
    baseline_course,
    progress_delta,
    segment_id,
    path_type,
):
    lateral_error = abs(state_raw[0])
    lateral_error_last = abs(state_last_raw[0])
    course_error = abs(state_raw[1])
    angular_velocity = abs(state_raw[4])
    velocity = abs(state_raw[2])

    profiles = {
        "single_turn_90": {
            "progress_scale": 5.0,
            "turn_speed_target": 3.0,
            "turn_speed_limit": 3.8,
            "turn_bonus": 2.0,
            "exit_bonus": 1.0,
        },
        "single_turn_wide": {
            "progress_scale": 5.5,
            "turn_speed_target": 3.4,
            "turn_speed_limit": 4.2,
            "turn_bonus": 1.5,
            "exit_bonus": 1.0,
        },
        "single_turn_exit": {
            "progress_scale": 4.5,
            "turn_speed_target": 2.7,
            "turn_speed_limit": 3.5,
            "turn_bonus": 2.5,
            "exit_bonus": 1.5,
        },
    }
    profile = profiles[path_type]

    if lateral_error < baseline_lateral * 0.6:
        precision_reward = 12.0
    elif lateral_error < baseline_lateral * 0.85:
        precision_reward = 6.0
    elif lateral_error < baseline_lateral:
        precision_reward = 2.0
    elif lateral_error < baseline_lateral * 1.25:
        precision_reward = -2.0
    elif lateral_error < baseline_lateral * 1.6:
        precision_reward = -6.0
    else:
        precision_reward = -12.0

    error_reduction = lateral_error_last - lateral_error
    if error_reduction >= 0:
        improvement_reward = 18.0 * min(error_reduction, 0.12)
    else:
        improvement_reward = 24.0 * max(error_reduction, -0.15)

    if lateral_error < baseline_lateral * 1.1:
        if course_error < baseline_course * 0.7:
            heading_reward = 3.0
        elif course_error < baseline_course:
            heading_reward = 1.5
        elif course_error < baseline_course * 1.35:
            heading_reward = -1.0
        else:
            heading_reward = -4.0
    else:
        heading_reward = -min(2.0, 1.5 * max(0.0, course_error - baseline_course))

    if angular_velocity < 0.25:
        smoothness_reward = 0.8
    elif angular_velocity < 0.45:
        smoothness_reward = 0.0
    elif angular_velocity < 0.75:
        smoothness_reward = -0.6
    else:
        smoothness_reward = -1.4

    if lateral_error < baseline_lateral * 0.7:
        tracking_reward = 2.0
    elif lateral_error < baseline_lateral:
        tracking_reward = 1.0
    elif lateral_error < baseline_lateral * 1.4:
        tracking_reward = -1.0
    else:
        tracking_reward = -3.0

    progress_reward = (
        profile["progress_scale"] * max(progress_delta, 0.0) -
        3.0 * max(-progress_delta, 0.0)
    )
    progress_reward = float(np.clip(progress_reward, -2.5, 2.5))

    speed_reward = 0.0
    segment_reward = 0.0
    turn_speed_target = profile["turn_speed_target"]
    turn_speed_limit = profile["turn_speed_limit"]

    if segment_id == 1:
        if course_error < baseline_course and lateral_error < baseline_lateral:
            segment_reward += 0.5
        if velocity < 1.8:
            speed_reward -= 0.8
        elif velocity > 3.0 and lateral_error < baseline_lateral:
            speed_reward += 0.5
    elif segment_id == 2:
        if lateral_error < baseline_lateral and course_error < baseline_course:
            segment_reward += profile["turn_bonus"]
        elif lateral_error > baseline_lateral * 1.6 or course_error > baseline_course * 1.5:
            segment_reward -= 2.0

        if (turn_speed_target - 0.7) <= velocity <= turn_speed_limit and lateral_error < baseline_lateral * 1.1:
            speed_reward += 1.0
        elif velocity > turn_speed_limit and (
            lateral_error > baseline_lateral * 0.8 or course_error > baseline_course
        ):
            speed_reward -= 2.0
    elif segment_id == 3:
        if course_error < baseline_course * 0.8 and lateral_error < baseline_lateral * 1.1:
            segment_reward += profile["exit_bonus"]
        elif course_error > baseline_course * 1.5:
            segment_reward -= 1.5

        if velocity < 1.8:
            speed_reward -= 0.6
        elif velocity > 3.2 and lateral_error < baseline_lateral:
            speed_reward += 0.4

    time_penalty = -0.05

    reward = (
        precision_reward +
        improvement_reward +
        heading_reward +
        smoothness_reward +
        tracking_reward +
        progress_reward +
        speed_reward +
        segment_reward +
        time_penalty
    )
    return reward

# ==================== 第一阶段：纯强化学习平衡控制器训练环境 ====================

class Attitude_control_stage1(gym.Env):
    """第一阶段：训练纯强化学习车把控制器。"""

    def __init__(self, render: bool = False, 
                 failure_penalty: float = -10.0,
                 early_termination_penalty: float = -20.0,
                 action_repeat: int = 1):
        self._render = render
        
        # 惩罚参数
        self.FAILURE_PENALTY = failure_penalty
        self.EARLY_TERMINATION_PENALTY = early_termination_penalty
        self.action_repeat = action_repeat
        
        print(f"[Stage 1 - Pure RL] 初始化环境:")
        print(f"  - 失败惩罚: {self.FAILURE_PENALTY}")
        print(f"  - 早期失败额外惩罚: {self.EARLY_TERMINATION_PENALTY}")
        print(f"  - 动作重复: {self.action_repeat}")
        
        # 动作空间和状态空间
        self.action_space = spaces.Box(
            low=np.array([-1.]),
            high=np.array([1.]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1.]),
            high=np.array([1., 1., 1., 1.]),
            dtype=np.float32
        )

        # 物理引擎初始化
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.cycle = 1 / 30
        p.setTimeStep(self.cycle)
        
        self.max_step_num = 1000
        self.min_success_steps = 1000
        self.angle_error = 0
        self.angle_error_csv = []
        
        # 状态变量
        self.target_theta = 0
        self.theta0_old = 0
        self.theta1_old = 0
        
        self.step_num = 0
        self.epoch_num = 0
        self.r = 0
        self.epoch_r_list = []
        
        p.setGravity(0, 0, -9.8)

        # 数据记录
        self.record_flag = 0
        self.target_theta_csv = []
        self.theta0_csv = []
        self.handle_angle_csv = []
        self.target_handle_angle_csv = []

        # 加载环境
        p.loadURDF(r"plane.urdf", globalScaling=15)

        #p.loadURDF(_asset_path("generated_terrain_lab.urdf"), globalScaling=2)
        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = BIKE_URDF_PATH
        self.bike = p.loadURDF(path, startPos, startOrientation)

        # 设置物理参数
        for link_idx in [0, 1, 2, -1]:
            p.changeDynamics(
                bodyUniqueId=self.bike, 
                linkIndex=link_idx,
                restitution=0.5,
                contactStiffness=10**8,
                contactDamping=10**5
            )

    def _map_action_to_handle_angle(self, action):
        """将归一化动作映射为最终车把目标角；不使用LQR基准项。"""
        return float(np.clip(action[0], -1.0, 1.0) * (math.pi / 4))

    def __get_observation(self, target_theta=0, recoder=0):
        """获取当前环境的观测状态"""
        _, cubeOrn = p.getBasePositionAndOrientation(self.bike)
        t0 = p.getEulerFromQuaternion(cubeOrn)
        
        theta0 = t0[0]
        dis_angle = target_theta - theta0
        w0 = (theta0 - self.theta0_old) / self.cycle

        if recoder == 1:
            self.theta0_old = theta0

        vx = p.getBaseVelocity(self.bike)[0][0]
        vy = p.getBaseVelocity(self.bike)[0][1]
        v = math.sqrt(vx**2 + vy**2)

        if recoder:
            self.angle_error += abs(dis_angle)
            self.theta0_csv.append(theta0)
            self.target_theta_csv.append(target_theta)
            self.handle_angle_csv.append(p.getJointState(bodyUniqueId=self.bike, jointIndex=1)[0])

        # 状态归一化
        dis_angle = np.clip(dis_angle, -1.57, 1.57) / 1.57
        theta0 = np.clip(theta0, -1.57, 1.57) / 1.57
        w0 = np.clip(w0, -10., 10.) / 10
        v = np.clip(v, -5., 5.) / 5

        return np.array([dis_angle, theta0, w0, v], dtype=np.float32)

    def __observation_reduction(self, state):
        """状态反归一化"""
        dis_angle = state[0] * 1.57
        theta0 = state[1] * 1.57
        w0 = state[2] * 10
        v = state[3] * 5
        
        return np.array([dis_angle, theta0, w0, v])

    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
                    """Potential-based reward shaping with conservative potential function"""
                    state_last_raw = self.__observation_reduction(state_last)
                    state_raw = self.__observation_reduction(state)

                    current_error = abs(state_raw[0])  # dis_angle (tilt from vertical)
                    angular_velocity = state_raw[2]  # w0 (can be negative)
                    last_error = abs(state_last_raw[0])
                    last_angular_velocity = state_last_raw[2]

                    # 1. Core tracking reward (based on current error)
                    max_penalty = 2.0
                    tracking_reward = -min(current_error**2, max_penalty)

                    # 2. High precision bonus
                    bonus_reward = 0.0
                    if current_error < 0.005:
                        bonus_reward = 1.0
                    elif current_error < 0.01:
                        bonus_reward = 0.5
                    elif current_error < 0.02:
                        bonus_reward = 0.2

                    # 3. Potential-Based Reward Shaping (PBRS) with discount factor
                    # Use conservative potential function: Phi(s) = -alpha * error (safety-aware)
                    # Research formula: F(s, a, s') = gamma * Phi(s') - Phi(s)
                    gamma = 0.99  # Discount factor from research

                    # Conservative potential: higher alpha when near unsafe boundaries
                    # This creates risk-aware potentials that penalize unsafe states more heavily
                    alpha = 2.0  # Base scaling
                    if current_error > 0.5:  # Near tipping point - increase penalty weight
                        alpha = 4.0 * (1.0 + current_error)  # Escalating penalty

                    # Calculate potentials with safety-aware scaling
                    potential_current = -alpha * current_error
                    potential_last = -alpha * last_error  # Use same alpha for consistency

                    # PBRS shaped reward: gamma * Phi(s') - Phi(s)
                    # This preserves optimal policy while providing learning signal
                    shaped_reward = gamma * potential_current - potential_last

                    # Risk-gating: reduce shaping when error is dangerously high
                    risk_gate = 1.0
                    if current_error > 0.5:  # Approaching unsafe region
                        risk_gate = max(0.2, 1.0 - 1.5 * (current_error - 0.5))

                    improvement_reward = shaped_reward * risk_gate

                    # 4. Action penalty for smooth control
                    action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)

                    # 5. Safety boundary reward (gated against unsafe behavior)
                    safety_bonus = 0.0
                    if current_error > 0.8:  # Near tipping point
                        safety_bonus = -2.0 * (current_error - 0.8)
                    elif current_error < 0.05:  # Very balanced
                        safety_bonus = 0.5

                    # 6. Oscillation penalty (excessive angular velocity changes)
                    oscillation_penalty = 0.0
                    if abs(angular_velocity) > 5.0:  # Large oscillations
                        oscillation_penalty = -0.1 * (abs(angular_velocity) - 5.0)

                    # 7. Learning progress reward (decays over time)
                    learning_progress = 0.0
                    if self.step_num > 0:
                        progress_factor = 1.0 - min(self.step_num / self.max_step_num, 1.0)
                        learning_progress = 0.1 * progress_factor

                    # 8. Terminal condition bonus for long survival
                    survival_bonus = 0.0
                    if self.step_num > 800:  # Survived most of episode
                        survival_bonus = 0.5

                    # Compute total reward
                    reward = (tracking_reward + bonus_reward + improvement_reward + 
                             action_penalty + safety_bonus + oscillation_penalty + 
                             learning_progress + survival_bonus)

                    # Ensure finite and bounded reward
                    if math.isnan(reward) or math.isinf(reward):
                        reward = 0.0
                    reward = max(-100.0, min(100.0, reward))

                    return reward

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.epoch_r_list.append(self.r)
        
        print(f"Episode:{self.epoch_num} | Steps:{self.step_num} | "
              f"Reward:{self.r:.2f} | Mean_reward:{np.mean(self.epoch_r_list[-20:]):.2f} | "
              f"Angle_error:{self.angle_error/(1+self.step_num):.4f}")

        self.target_theta = 0
        self.step_num = 0
        self.r = 0
        self.epoch_num += 1
        
        self.theta0_old = 0
        self.theta1_old = 0
        
        self.angle_error = 0

        self.target_theta_csv = []
        self.theta0_csv = []
        self.handle_angle_csv = []
        self.target_handle_angle_csv = []

        startPos = [0, 0, 0.75]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        
        p.resetBasePositionAndOrientation(self.bike, startPos, startOrientation)
        p.resetJointState(self.bike, 0, 0, 0)
        p.resetJointState(self.bike, 1, 0, 0)
        p.resetJointState(self.bike, 2, 0, 0)
        
        p.stepSimulation()

        return self.__get_observation(self.target_theta), {}

    def step(self, action):
        """执行纯RL车把角控制动作。"""
        cumulative_reward = 0.0
        terminated = False
        truncated = False
        
        target_handle_angle = self._map_action_to_handle_angle(action)
        
        # 动作重复循环
        for repeat_idx in range(self.action_repeat):
            # 更新相机
            location, _ = p.getBasePositionAndOrientation(self.bike)
            p.resetDebugVisualizerCamera(
                cameraDistance=2,
                cameraYaw=-70,
                cameraPitch=-10.0001,
                cameraTargetPosition=location
            )
            
            state_last = self.__get_observation(self.target_theta, recoder=1)
            
            if self.record_flag:
                self.target_handle_angle_csv.append(target_handle_angle)
            
            # 目标倾斜角的动态变化，与旧HRRL纯RL控制入口保持一致
            if (self.step_num + repeat_idx) % 100 == 0:
                sigma = min(0.3, (1.08**(self.epoch_num)) * 0.01)
                self.target_theta = np.random.normal(loc=0.0, scale=sigma, size=None)
                if abs(self.target_theta) >= math.pi/12:
                    self.target_theta = random.uniform(-math.pi/12, math.pi/12)
            
            # 执行电机控制
            p.setJointMotorControl2(self.bike, 0, p.VELOCITY_CONTROL, 
                                    targetVelocity=-10, force=20)
            p.setJointMotorControl2(self.bike, 2, p.VELOCITY_CONTROL, 
                                    targetVelocity=-10, force=20)
            p.setJointMotorControl2(self.bike, 1, p.POSITION_CONTROL,
                                    targetPosition=target_handle_angle, force=200, positionGain=0.3)
            
            p.stepSimulation()
            
            state = self.__get_observation(self.target_theta)
            reward = self.__calculate_reward(state_last, state, target_handle_angle)
            cumulative_reward += reward
            
            # 检查终止条件
            if abs(1.57 * state[1]) > (math.pi/3):
                if self.step_num < self.min_success_steps:
                    early_penalty = self.EARLY_TERMINATION_PENALTY * \
                                  (1 - self.step_num / self.min_success_steps)
                    cumulative_reward += self.FAILURE_PENALTY + early_penalty
                else:
                    cumulative_reward += self.FAILURE_PENALTY
                terminated = True
                break
        
        self.step_num += self.action_repeat
        
        if self.step_num >= self.max_step_num and not terminated:
            truncated = True
        
        if terminated or truncated:
            if self.step_num > 0:
                self.angle_error_csv.append(self.angle_error / self.step_num)
                np.savetxt(_model_path("stage1_angle_error.csv"), 
                          np.array([self.angle_error_csv]).T, delimiter=',')
            
            if self.record_flag == 1:
                min_len = min(
                    len(self.target_theta_csv),
                    len(self.theta0_csv),
                    len(self.handle_angle_csv),
                    len(self.target_handle_angle_csv)
                )
                
                print(f"[Stage 1] 保存数据: {min_len} 个时间步")
                
                np.savetxt(_model_path("stage1_data.csv"), np.array([
                    self.target_theta_csv[:min_len],
                    self.theta0_csv[:min_len],
                    self.handle_angle_csv[:min_len],
                    self.target_handle_angle_csv[:min_len]
                ]).T, delimiter=',')
        
        info = {}
        self.r += cumulative_reward
        
        return state, cumulative_reward, terminated, truncated, info

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

# ==================== 第三阶段：自适应Stanley控制器训练环境 ====================

class Path_tracking_stage3(gym.Env):
    """第三阶段：训练自适应Stanley控制器（完整版）"""

    def __init__(self, render: bool = False, 
                agent_lqr_path: str = None,
                path_type: str = "s_line",
                action_repeat: int = 1,
                heading_offset_reset_mode: str = "legacy"):
        """
        初始化环境
        
        参数:
            render: 是否可视化
            agent_lqr_path: LQR模型路径
            path_type: 路径类型 ("s_line"、"complex"、"single_turn_90"、"single_turn_wide"、"single_turn_exit")
            action_repeat: 动作重复次数
            heading_offset_reset_mode: reset阶段航向初始化模式
                - "legacy": 原方法。保持原来的固定初始朝向，不额外引入航向偏置。
                - "offset": 新方法。在路径默认起始朝向基础上叠加一个初始航向偏置。
                  说明：Stage3 当前没有独立的 init_heading_offset 参数，这里的 "offset" 主要用于
                  让第三阶段在不同 reset 语义下保持行为一致，便于实验组织和结果解释。
        """
        self._render = render
        self.path_type = path_type
        self.action_repeat = action_repeat
        self.heading_offset_reset_mode = heading_offset_reset_mode
        if self.heading_offset_reset_mode not in ("legacy", "offset"):
            raise ValueError(
                'heading_offset_reset_mode 只能是 "legacy" 或 "offset"。'
                '其中 "legacy"=原方法，"offset"=新方法。'
            )
        
        # ✅ 路径特定参数（以 env.py 为唯一真相来源）
        path_cfg = _get_path_tracking_config(path_type, action_repeat)
        _apply_path_tracking_config(self, path_cfg)
        print(f"[Stage 3] {path_cfg['display_name']}参数:")
        print(f"  完成条件: {path_cfg['completion_desc']}")
        print(f"  最大步数: {self.max_step_num} (={self.max_step_num//action_repeat}个动作)")
        print(f"  早期失败阈值: {self.early_failure_threshold}个动作")
        print(f"  基线: Lateral={self.baseline_lateral}m, Course={self.baseline_course}rad")
        print(
            f"  reset航向模式: {self.heading_offset_reset_mode} "
            f"({'原方法' if self.heading_offset_reset_mode == 'legacy' else '新方法'})"
        )

        # 验证模型
        if agent_lqr_path is None :    
            raise ValueError("必须提供模型路径")
        
        if os.path.exists(agent_lqr_path):
            self.agent_lqr = TD3.load(agent_lqr_path)
            print(f"  ✅ Agent_LQR 已加载")
        else:
            self.agent_lqr = None
            print(f"  ⏭️  Agent_LQR 外部提供")
        # 动作和状态空间
        self.action_space = spaces.Box(
            low=np.array([-1., -1.]),
            high=np.array([1., 1.]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., -1.]),
            high=np.array([1., 1., 1., 1., 1., 1.]),
            dtype=np.float32
        )

        # 物理引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.cycle = 1 / 30
        p.setTimeStep(self.cycle)

        # 状态变量
        self.target_theta = 0
        self.theta0_old = 0
        self.theta1_old = 0
        self.step_num = 0
        self.action_num = 0
        self.epoch_num = 0
        self.r = 0
        self.epoch_r_list = []
        
        p.setGravity(0, 0, -9.8)
        
        # 数据记录
        self.record_flag = 0
        self.target_theta_csv = []
        self.theta0_csv = []
        self.target_handle_angle_csv = []
        self.handle_angle_csv = []
        self.later_error_csv = []
        self.course_error_angle_csv = []
        self.X = []
        self.Y = []
        self.k_lat_csv = []
        self.k_course_csv = []
        self.position_error_csv = []

        # ✅ 加载环境（根据路径类型）
        startPos = _load_path_scene(path_type)
        print(f"  ✅ {path_cfg['display_name']}已加载")
        
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = BIKE_URDF_PATH
        self.bike = p.loadURDF(path, startPos, startOrientation)

        for link_idx in [0, 1, 2, -1]:
            p.changeDynamics(
                bodyUniqueId=self.bike, 
                linkIndex=link_idx,
                restitution=0.5,
                contactStiffness=10**8,
                contactDamping=10**5
            )
        
        print(f"[Stage 3] 初始化完成\n")

    def _map_action_to_stanley_gains(self, action):
        """映射Stanley增益"""
        k_lat = (action[0] + 1) / 2 * (1.0 - 0.2) + 0.2
        k_course = (action[1] + 1) / 2 * (0.8 - 0.2) + 0.2
        return k_lat, k_course

    def _map_lqr_action_to_gains(self, action):
        """映射LQR增益"""
        kp = (action[0] + 1) / 2 * (25 - 5) + 5
        kd = (action[1] + 1) / 2 * (5 - 1) + 1
        k = (action[2] + 1) / 2 * (20 - 0) + 0
        return kp, kd, k

    def get_state_attitude_control(self, target_theta=0, recoder=0):
        """获取姿态控制状态"""
        _, cubeOrn = p.getBasePositionAndOrientation(self.bike)
        t0 = p.getEulerFromQuaternion(cubeOrn)
        
        theta0 = t0[0]
        dis_angle = target_theta - theta0
        w0 = (theta0 - self.theta0_old) / self.cycle

        if recoder == 1:
            self.theta0_old = theta0

        vx = p.getBaseVelocity(self.bike)[0][0]
        vy = p.getBaseVelocity(self.bike)[0][1]
        v = math.sqrt(vx**2 + vy**2)

        if recoder:
            self.theta0_csv.append(theta0)
            self.target_theta_csv.append(target_theta)
            self.handle_angle_csv.append(p.getJointState(bodyUniqueId=self.bike, jointIndex=1)[0])

        dis_angle = np.clip(dis_angle, -1.57, 1.57) / 1.57
        theta0 = np.clip(theta0, -1.57, 1.57) / 1.57
        w0 = np.clip(w0, -10., 10.) / 10
        v = np.clip(v, -5., 5.) / 5

        return np.array([dis_angle, theta0, w0, v], dtype=np.float32)

    def __get_observation(self, recoder=0):
                """
                ✅✅✅ 终极修复版 V3 - 添加segment7
                
                核心策略：
                1. 优先匹配圆弧段（segment2/4/5/6）- 用精确的几何判断
                2. 直线段作为兜底（segment1/3/7）- 只有圆弧不匹配时才用
                3. 过渡区用"距离最近"原则而非"范围重叠"
                
                ✅ Segment 6: 从segment5下方接入，角度范围-90°~180°
                ✅ Segment 7: 垂直向下直线段，从(0,35)到(0,0)
                """
                # 🆕 初始化步数计数器（如果不存在）
                if not hasattr(self, 'step_counter'):
                    self.step_counter = 0
                
                self.step_counter += 1
                
                _, cubeOrn = p.getBasePositionAndOrientation(self.bike)
                t0 = p.getEulerFromQuaternion(cubeOrn)

                theta0 = t0[0]
                w0 = (theta0 - self.theta0_old) / self.cycle

                vx = p.getBaseVelocity(self.bike)[0][0]
                vy = p.getBaseVelocity(self.bike)[0][1]
                v = math.sqrt(vx**2 + vy**2)

                back_wheel_point_x, back_wheel_point_y, _ = p.getLinkState(self.bike, 0)[0]
                forward_wheel_point_x, forward_wheel_point_y, _ = p.getLinkState(self.bike, 2)[0]

                bike_direction_vector = [forward_wheel_point_x - back_wheel_point_x,
                                        forward_wheel_point_y - back_wheel_point_y]
                
                # ========== S型路径（使用修复版v3的完整逻辑）==========
                if self.path_type == "s_line":
                    """
                    ✅ 完全修复版 v3：精确匹配S型路径定义
                    
                    路径定义（来自生成器）：
                    - 第一段圆弧：圆心(0, 15)，半径15
                    轨迹：(0,0) → (15,15) → (0,30)
                    判定条件：x >= 0 且 0 <= y <= 30
                    - 第二段圆弧：圆心(0, 45)，半径15
                    轨迹：(0,30) → (-15,45) → (0,60)
                    判定条件：x <= 0 且 30 <= y <= 60
                    """
                    
                    # ========== ✅ 修复v3：正确的路径判定 ==========
                    
                    # 第一段圆弧：圆心(0, 15)，半径15
                    # 适用于：x >= 0 且 0 <= y <= 30
                    if (forward_wheel_point_x >= 0) and (0 <= forward_wheel_point_y <= 30):
                        auxiliary_vector = [0 - forward_wheel_point_x, 15 - forward_wheel_point_y]
                        modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                        
                        if modulus_bike < 0.001 or modulus_auxiliary < 0.001:
                            course_error_angle = 0
                        else:
                            var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                bike_direction_vector[1] * auxiliary_vector[1]
                            var = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                            course_error_angle = math.acos(var)
                            course_error_angle = math.pi/2 - course_error_angle
                        
                        lateral_error = 15 - modulus_auxiliary
                        k = 1/15  # 正曲率（右弯）
                    
                    # 第二段圆弧：圆心(0, 45)，半径15
                    # 适用于：x <= 0 且 30 <= y <= 60
                    elif (forward_wheel_point_x <= 0) and (30 <= forward_wheel_point_y <= 60):
                        auxiliary_vector = [0 - forward_wheel_point_x, 45 - forward_wheel_point_y]
                        modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                        
                        if modulus_bike < 0.001 or modulus_auxiliary < 0.001:
                            course_error_angle = 0
                        else:
                            var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                bike_direction_vector[1] * auxiliary_vector[1]
                            var = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                            course_error_angle = math.acos(var)
                            course_error_angle = course_error_angle - math.pi/2
                        
                        lateral_error = modulus_auxiliary - 15
                        k = -1/15  # 负曲率（左弯）
                    
                    # ========== ✅ 修复v3：处理交界处和超出范围 ==========
                    else:
                        # 根据y坐标判断使用哪个圆心
                        if forward_wheel_point_y < 15:
                            # 接近起点，使用第一段圆心
                            auxiliary_vector = [0 - forward_wheel_point_x, 15 - forward_wheel_point_y]
                            modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                            modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                            
                            if modulus_bike < 0.001 or modulus_auxiliary < 0.001:
                                course_error_angle = 0
                            else:
                                var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                    bike_direction_vector[1] * auxiliary_vector[1]
                                var = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                                course_error_angle = math.acos(var)
                                course_error_angle = math.pi/2 - course_error_angle
                            
                            lateral_error = 15 - modulus_auxiliary
                            k = 1/15
                            
                        elif 15 <= forward_wheel_point_y < 45:
                            # 在两段之间，根据x判断
                            if forward_wheel_point_x >= 0:
                                # 靠近第一段
                                auxiliary_vector = [0 - forward_wheel_point_x, 15 - forward_wheel_point_y]
                                modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                                modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                                
                                if modulus_bike < 0.001 or modulus_auxiliary < 0.001:
                                    course_error_angle = 0
                                else:
                                    var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                        bike_direction_vector[1] * auxiliary_vector[1]
                                    var = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                                    course_error_angle = math.acos(var)
                                    course_error_angle = math.pi/2 - course_error_angle
                                
                                lateral_error = 15 - modulus_auxiliary
                                k = 1/15
                            else:
                                # 靠近第二段
                                auxiliary_vector = [0 - forward_wheel_point_x, 45 - forward_wheel_point_y]
                                modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                                modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                                
                                if modulus_bike < 0.001 or modulus_auxiliary < 0.001:
                                    course_error_angle = 0
                                else:
                                    var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                        bike_direction_vector[1] * auxiliary_vector[1]
                                    var = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                                    course_error_angle = math.acos(var)
                                    course_error_angle = course_error_angle - math.pi/2
                                
                                lateral_error = modulus_auxiliary - 15
                                k = -1/15
                                
                        else:  # y >= 45
                            # 接近终点，使用第二段圆心
                            auxiliary_vector = [0 - forward_wheel_point_x, 45 - forward_wheel_point_y]
                            modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                            modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                            
                            if modulus_bike < 0.001 or modulus_auxiliary < 0.001:
                                course_error_angle = 0
                            else:
                                var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                    bike_direction_vector[1] * auxiliary_vector[1]
                                var = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                                course_error_angle = math.acos(var)
                                course_error_angle = course_error_angle - math.pi/2
                            
                            lateral_error = modulus_auxiliary - 15
                            k = -1/15
                            
                            # ✅ 关键：超过终点后增加误差惩罚
                            if forward_wheel_point_y > 60:
                                lateral_error += (forward_wheel_point_y - 60) * 1.0
                
                elif self.path_type in SINGLE_TURN_PATH_TYPES:
                    lateral_error, course_error_angle, k, self.current_path_segment = _compute_single_turn_tracking_errors(
                        self.path_type,
                        forward_wheel_point_x,
                        forward_wheel_point_y,
                        bike_direction_vector
                    )

                else:
                    # ========== 复杂路径（优先级判断版）==========
                    segment_matched = False
                    
                    # ✅ 策略：先计算到所有段的距离，然后按优先级选择
                    segment_distances = {}
                    
                    # ===== 计算到各段的距离 =====
                    
                    # 段1：起始直线（y=0线，x<55）
                    if forward_wheel_point_x < 55:
                        segment_distances['segment1'] = abs(forward_wheel_point_y)
                    
                    # 段2：右下圆弧（圆心55,15，半径15，角度-110°~10°）
                    dx2 = forward_wheel_point_x - 55
                    dy2 = forward_wheel_point_y - 15
                    angle2_deg = math.degrees(math.atan2(dy2, dx2))
                    dist2_to_center = math.sqrt(dx2**2 + dy2**2)
                    
                    if -120 <= angle2_deg <= 20:
                        segment_distances['segment2'] = abs(dist2_to_center - 15)
                    
                    # 段3：右侧直线（x=70线，y>15）
                    if forward_wheel_point_y > 15:
                        segment_distances['segment3'] = abs(forward_wheel_point_x - 70)
                    
                    # 段4：右上圆弧（圆心55,35，半径15，角度-10°~190°）
                    dx4 = forward_wheel_point_x - 55
                    dy4 = forward_wheel_point_y - 35
                    angle4_deg = math.degrees(math.atan2(dy4, dx4))
                    dist4_to_center = math.sqrt(dx4**2 + dy4**2)
                    
                    if -20 <= angle4_deg <= 200:
                        segment_distances['segment4'] = abs(dist4_to_center - 15)
                    
                    # 段5：左上圆弧（圆心28,35，半径12，角度-160°~50°）
                    dx5 = forward_wheel_point_x - 28
                    dy5 = forward_wheel_point_y - 35
                    angle5_deg = math.degrees(math.atan2(dy5, dx5))
                    dist5_to_center = math.sqrt(dx5**2 + dy5**2)
                    
                    if -180 <= angle5_deg <= -10:
                        if 8 < dist5_to_center < 16:
                            segment_distances['segment5'] = abs(dist5_to_center - 12)
                    
                    # ✅✅✅ 段6：上半圆弧（圆心8,35，半径8）
                    # 定义：从右下(16,27)经右侧(16,35)到顶部(8,43)再到左侧(0,35)
                    dx6 = forward_wheel_point_x - 8
                    dy6 = forward_wheel_point_y - 35
                    angle6_deg = math.degrees(math.atan2(dy6, dx6))
                    dist6_to_center = math.sqrt(dx6**2 + dy6**2)
                    
                    # ✅✅✅ 修复：角度范围应该包括负角度！
                    # 从segment5下方接入：-90° (下方) → 0° (右侧) → 90° (顶部) → 180° (左侧)
                    if -10 <= angle6_deg <= 180:
                        # 距离验证
                        if 4 < dist6_to_center < 13:
                            segment_distances['segment6'] = abs(dist6_to_center - 8)
                    
                    # 🆕🆕🆕 段7：垂直向下直线段（x=0线，y从35到0）
                    # 定义：从segment6左侧终点(0,35)垂直向下回到起点(0,0)
                    if forward_wheel_point_y < 35 and forward_wheel_point_y > 2:
                        x_distance = abs(forward_wheel_point_x)
                        if x_distance < 3.0:
                            segment_distances['segment7'] = x_distance
                    # ===== 按优先级选择最佳段 =====
                    # 优先级1：圆弧段（如果距离<5米，优先选择）
                    best_arc_segment = None
                    best_arc_distance = 5.0
                    
                    for seg in ['segment2', 'segment4', 'segment5', 'segment6']:
                        if seg in segment_distances and segment_distances[seg] < best_arc_distance:
                            best_arc_distance = segment_distances[seg]
                            best_arc_segment = seg
                    
                    # 优先级2：直线段（如果没有合适的圆弧，选择最近的直线）
                    best_line_segment = None
                    best_line_distance = float('inf')
                    
                    for seg in ['segment1', 'segment3', 'segment7']:
                        if seg in segment_distances and segment_distances[seg] < best_line_distance:
                            best_line_distance = segment_distances[seg]
                            best_line_segment = seg
                    
                    # ✅ 决策逻辑：圆弧优先，直线兜底
                    if best_arc_segment:
                        chosen_segment = best_arc_segment
                        if self.step_counter % 10 == 0:
                            #print(f"[步数 {self.step_counter:4d}] 位置:({forward_wheel_point_x:6.2f}, {forward_wheel_point_y:6.2f}) | 匹配段:{chosen_segment} | 距离:{best_arc_distance:.3f}m")
                            None
                    elif best_line_segment and best_line_distance < 5.0:
                        chosen_segment = best_line_segment
                        if self.step_counter % 10 == 0:
                            None
                            #print(f"[步数 {self.step_counter:4d}] 位置:({forward_wheel_point_x:6.2f}, {forward_wheel_point_y:6.2f}) | 匹配段:{chosen_segment} | 距离:{best_line_distance:.3f}m")
                    else:
                        chosen_segment = None
                        if self.step_counter % 10 == 0:
                            #print(f"[步数 {self.step_counter:4d}] 位置:({forward_wheel_point_x:6.2f}, {forward_wheel_point_y:6.2f}) | 匹配段:None (离开路径)")
                            None
                    # ===== 根据选定的段计算控制误差 =====
                    if chosen_segment == 'segment1':
                        self.current_path_segment = 1
                        modulus = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        if modulus > 0.001:
                            course_error_angle = math.acos(np.clip(bike_direction_vector[0] / modulus, -1, 1))
                            if bike_direction_vector[1] < 0:
                                course_error_angle = -course_error_angle
                        else:
                            course_error_angle = 0
                        lateral_error = forward_wheel_point_y
                        k = 0
                        segment_matched = True
                        
                    elif chosen_segment == 'segment2':
                        self.current_path_segment = 2
                        auxiliary_vector = [55 - forward_wheel_point_x, 15 - forward_wheel_point_y]
                        modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                        
                        if modulus_bike > 0.001 and modulus_auxiliary > 0.001:
                            var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                bike_direction_vector[1] * auxiliary_vector[1]
                            var_clamped = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                            course_error_angle = math.acos(var_clamped)
                            course_error_angle = math.pi/2 - course_error_angle
                        else:
                            course_error_angle = 0
                        lateral_error = 15 - modulus_auxiliary
                        k = 1/15
                        segment_matched = True
                        
                    elif chosen_segment == 'segment3':
                        self.current_path_segment = 3
                        modulus = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        if modulus > 0.001:
                            course_error_angle = math.acos(np.clip(bike_direction_vector[1] / modulus, -1, 1))
                            if bike_direction_vector[0] > 0:
                                course_error_angle = -course_error_angle
                        else:
                            course_error_angle = 0
                        lateral_error = -(forward_wheel_point_x - 70)
                        k = 0
                        segment_matched = True
                        
                    elif chosen_segment == 'segment4':
                        self.current_path_segment = 4
                        auxiliary_vector = [55 - forward_wheel_point_x, 35 - forward_wheel_point_y]
                        modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        modulus_auxiliary = math.sqrt(auxiliary_vector[0]**2 + auxiliary_vector[1]**2)
                        
                        if modulus_bike > 0.001 and modulus_auxiliary > 0.001:
                            var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                bike_direction_vector[1] * auxiliary_vector[1]
                            var_clamped = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                            course_error_angle = math.acos(var_clamped)
                            course_error_angle = math.pi/2 - course_error_angle
                        else:
                            course_error_angle = 0
                        lateral_error = 15 - modulus_auxiliary
                        k = 1/15
                        segment_matched = True
                        
                    elif chosen_segment == 'segment5':
                        self.current_path_segment = 5
                        auxiliary_vector = [28 - forward_wheel_point_x, 35 - forward_wheel_point_y]
                        modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        modulus_auxiliary = dist5_to_center
                        
                        if modulus_bike > 0.001 and modulus_auxiliary > 0.001:
                            var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                bike_direction_vector[1] * auxiliary_vector[1]
                            var_clamped = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                            course_error_angle = math.acos(var_clamped)
                            course_error_angle = course_error_angle - math.pi/2
                        else:
                            course_error_angle = 0
                        lateral_error = modulus_auxiliary - 12
                        k = -1/12
                        segment_matched = True
                        

                    elif chosen_segment == 'segment6':
                        self.current_path_segment = 6
                        # ✅ 使用与 S2/S4 相同的方法（逆时针圆弧）
                        
                        auxiliary_vector = [8 - forward_wheel_point_x, 35 - forward_wheel_point_y]
                        modulus_bike = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                        modulus_auxiliary = dist6_to_center
                        
                        if modulus_bike > 0.001 and modulus_auxiliary > 0.001:
                            var = bike_direction_vector[0] * auxiliary_vector[0] + \
                                bike_direction_vector[1] * auxiliary_vector[1]
                            var_clamped = np.clip(var / (modulus_bike * modulus_auxiliary), -1.0, 1.0)
                            course_error_angle = math.acos(var_clamped)
                            
                            # ✅ 与 S2/S4 相同（逆时针公式）
                            course_error_angle = math.pi/2 - course_error_angle
                        else:
                            course_error_angle = 0
                        
                        # ✅ 与 S2/S4 相同的横向误差符号
                        lateral_error = 8 - modulus_auxiliary
                        
                        # ✅ 正曲率（逆时针）
                        k = 1/8
                        segment_matched = True

                    elif chosen_segment == 'segment7':
                            self.current_path_segment = 7
                            # 🆕🆕🆕 垂直向下直线段：x=0，y从35到0
                            # 期望方向：沿y轴负向（向下）
                            
                            modulus = math.sqrt(bike_direction_vector[0]**2 + bike_direction_vector[1]**2)
                            
                            if modulus > 0.001:
                                # 航向误差计算：期望方向是纯-y方向 (0, -1)
                                # 计算自行车方向与期望方向的夹角
                                cos_value = -bike_direction_vector[1] / modulus
                                cos_value = np.clip(cos_value, -1.0, 1.0)
                                course_error_angle = math.acos(cos_value)
                                
                                # 根据x方向分量的符号调整误差正负
                                # 如果向左偏（x分量<0），误差为负；向右偏（x分量>0），误差为正
                                if bike_direction_vector[0] < 0:
                                    course_error_angle = -course_error_angle
                            else:
                                course_error_angle = 0
                    
                    # ✅ 在这里补充这3行（您原代码缺少）
                            lateral_error = forward_wheel_point_x
                            k = 0
                            segment_matched = True
                    else:
                        # 兜底：没有匹配到任何路径段
                        self.current_path_segment = 0
                        course_error_angle = 0
                        lateral_error = 0
                        k = 0

                # 记录数据
                if recoder:
                    self.later_error_csv.append(lateral_error)
                    self.course_error_angle_csv.append(course_error_angle)
                    self.X.append(forward_wheel_point_x)
                    self.Y.append(forward_wheel_point_y)

                # ========== 状态归一化 ==========
                
                lateral_error = np.clip(lateral_error, -10., 10.) / 10
                course_error_angle = np.clip(course_error_angle, -1.57, 1.57) / 1.57
                theta0 = np.clip(theta0, -1.57, 1.57) / 1.57
                w0 = np.clip(w0, -10., 10.) / 10
                v = np.clip(v, -5., 5.) / 5
                k = k * 8  # 曲率归一化

                return np.array([lateral_error, course_error_angle, v, theta0, w0, k], dtype=np.float32) 
    
    def __observation_reduction(self, state):
        """反归一化"""
        lateral_error = state[0] * 10
        course_error_angle = state[1] * 1.57
        v = state[2] * 5
        theta0 = state[3] * 1.57
        w0 = state[4] * 10
        k = state[5] / 8
        return np.array([lateral_error, course_error_angle, v, theta0, w0, k])

    def __calculate_reward(self, state_last, state):
            """
            【完整修改后的奖励函数】
            计算路径跟踪的奖励函数
            
            设计思路:
            1. 基础奖励：精度奖励、改进速度奖励、航向对齐奖励
            2. 平滑度奖励：惩罚过大的角速度（避免抖动）
            3. 效率奖励：鼓励在保持精度的同时提高速度
            4. 🆕 进度奖励：鼓励沿路径前进
            5. 🆕 路径段奖励：鼓励保持在路径上
            6. 🆕 针对第六段的特殊处理：增强稳定性
            """
            # 反归一化状态
            state_last_raw = self.__observation_reduction(state_last)
            state_raw = self.__observation_reduction(state)

            if self.path_type in SINGLE_TURN_PATH_TYPES:
                forward_wheel_point_x, forward_wheel_point_y, _ = p.getLinkState(self.bike, 2)[0]
                progress_delta, _, _ = _update_single_turn_progress_tracker(
                    self,
                    self.path_type,
                    forward_wheel_point_x,
                    forward_wheel_point_y,
                )
                return _calculate_single_turn_reward_core(
                    state_last_raw=state_last_raw,
                    state_raw=state_raw,
                    baseline_lateral=self.baseline_lateral,
                    baseline_course=self.baseline_course,
                    progress_delta=progress_delta,
                    segment_id=getattr(self, "current_path_segment", 0),
                    path_type=self.path_type,
                )

            if self.path_type == "complex":
                forward_wheel_point_x, forward_wheel_point_y, _ = p.getLinkState(self.bike, 2)[0]
                progress_delta, progress, total_len = _update_complex_progress_tracker(
                    self,
                    forward_wheel_point_x,
                    forward_wheel_point_y,
                )
                progress_percent = float(np.clip(progress / max(total_len, 1e-6), 0.0, 1.0) * 100.0)
                return _calculate_complex_reward_core(
                    state_last_raw=state_last_raw,
                    state_raw=state_raw,
                    baseline_lateral=self.baseline_lateral,
                    baseline_course=self.baseline_course,
                    progress_delta=progress_delta,
                    progress_percent=progress_percent,
                    segment_id=getattr(self, "current_path_segment", 0),
                )

            lateral_error = abs(state_raw[0])
            lateral_error_last = abs(state_last_raw[0])
            course_error = abs(state_raw[1])
            angular_velocity = abs(state_raw[4])
            velocity = abs(state_raw[2])
            
            # ========== 1. 精度奖励（分段，基于基线）==========
            baseline = self.baseline_lateral
            
            if lateral_error < baseline * 0.4:  # <0.2m
                precision_reward = 15.0
            elif lateral_error < baseline * 0.6:  # 0.2-0.3m
                precision_reward = 10.0
            elif lateral_error < baseline * 0.8:  # 0.3-0.4m
                precision_reward = 5.0
            elif lateral_error < baseline:  # 0.4-0.5m（基线）
                precision_reward = 1.0
            elif lateral_error < baseline * 1.2:  # 0.5-0.6m
                precision_reward = -2.0
            elif lateral_error < baseline * 1.6:  # 0.6-0.8m
                precision_reward = -5.0
            elif lateral_error < baseline * 2.0:  # 0.8-1.0m
                precision_reward = -10.0
            else:  # >1.0m
                precision_reward = -20.0
            
            # ========== 2. 改进速度奖励 ==========
            error_reduction = lateral_error_last - lateral_error
            
            if error_reduction > 0.01:
                improvement_reward = 20.0 * error_reduction
            elif error_reduction > 0:
                improvement_reward = 10.0 * error_reduction
            elif error_reduction > -0.01:
                improvement_reward = 15.0 * error_reduction
            else:
                improvement_reward = 30.0 * error_reduction
            
            # ========== 3. 航向对齐奖励 ==========
            if lateral_error < baseline:
                if course_error < self.baseline_course * 0.73:  # <0.08rad
                    heading_reward = 3.0
                elif course_error < self.baseline_course:  # <0.11rad
                    heading_reward = 1.0
                elif course_error < self.baseline_course * 1.36:  # <0.15rad
                    heading_reward = -1.0
                else:
                    heading_reward = -3.0
            else:
                heading_reward = 0.0
            
            # ========== 4. 平滑度 ==========
            if angular_velocity < 0.3:
                smoothness_reward = 0.5
            elif angular_velocity < 0.5:
                smoothness_reward = 0.0
            elif angular_velocity < 0.8:
                smoothness_reward = -0.5
            else:
                smoothness_reward = -1.5
            
            # ========== 5. 效率奖励 ==========
            if lateral_error < baseline * 0.8 and velocity > 3.5:
                efficiency_bonus = 3.0
            elif lateral_error < baseline and velocity > 3.0:
                efficiency_bonus = 1.0
            elif lateral_error > baseline * 1.6 or velocity < 2.0:
                efficiency_bonus = -2.0
            else:
                efficiency_bonus = 0.0
            
            # ========== 🆕 6. 进度奖励：鼓励沿路径前进 ==========
            # 需要在__init__中初始化：self.last_x_position = 0, self.total_forward_distance = 0
            if not hasattr(self, 'last_x_position'):
                self.last_x_position = 0
                self.total_forward_distance = 0
            
            # 获取当前位置
            _, _, _ = p.getLinkState(self.bike, 0)[0]
            forward_wheel_point_x, forward_wheel_point_y, _ = p.getLinkState(self.bike, 2)[0]
            
            # S型路径：主要沿y方向前进
            forward_progress = forward_wheel_point_y - getattr(self, 'last_y_position', 0)
            self.last_y_position = forward_wheel_point_y
            
            self.last_x_position = forward_wheel_point_x
            
            # 前进奖励（每前进1米奖励0.5分）
            r_progress = 0.5 * max(0, forward_progress)
            
            if forward_progress > 0:
                if not hasattr(self, 'total_forward_distance'):
                    self.total_forward_distance = 0
                self.total_forward_distance += forward_progress
            
            # ========== 🆕 7. 路径段奖励：保持在路径上 ==========
            if hasattr(self, 'current_path_segment'):
                # 在路径段上（segment > 0）持续奖励
                r_on_path = 2.0 if self.current_path_segment > 0 else -5.0
            else:
                r_on_path = 0.0
            
            # ========== 🆕 8. 第六段特殊处理：增强稳定性 ==========
            r_segment_6 = 0.0
            
            if hasattr(self, 'current_path_segment') and self.current_path_segment == 6:
                # 8.1 稳定奖励：误差都很小时给予高奖励
                if lateral_error < 0.5 and course_error < 0.2:
                    r_segment_6 += 3.0
                
                # 8.2 惩罚误差增加
                lateral_error_increase = lateral_error - lateral_error_last
                if lateral_error_increase > 0:
                    r_segment_6 -= 2.0 * lateral_error_increase
                
                # 8.3 额外惩罚过大的角速度（第6段需要更平滑）
                if angular_velocity > 5.0:
                    r_segment_6 -= 1.0
                elif angular_velocity > 3.0:
                    r_segment_6 -= 0.5
                
                # 8.4 鼓励低速通过（第6段半径小，需要谨慎）
                if velocity < 3.0 and lateral_error < baseline:
                    r_segment_6 += 1.0
                elif velocity > 4.0 and lateral_error > baseline * 0.8:
                    r_segment_6 -= 1.0
            
            # ========== 🆕 9. 时间惩罚：鼓励快速完成 ==========
            r_time = -0.05
            
            # ========== 10. 组合所有奖励 ==========
            reward = (
                precision_reward +
                improvement_reward +
                heading_reward +
                smoothness_reward +
                efficiency_bonus +
                r_progress +
                r_on_path +
                r_segment_6 +
                r_time
            )
            
            return reward

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.epoch_r_list.append(self.r)
        
        mean_lateral = sum(map(abs, self.later_error_csv)) / (len(self.later_error_csv) + 0.00001)
        mean_course = sum(map(abs, self.course_error_angle_csv)) / (len(self.course_error_angle_csv) + 0.00001)
        
        print(f"Episode:{self.epoch_num} | Steps:{self.step_num} | Actions:{self.action_num} | "
              f"Reward:{self.r:.2f} | Mean_reward:{np.mean(self.epoch_r_list[-20:]):.2f} | "
              f"Lateral:{mean_lateral:.4f}m | Course:{mean_course:.4f}rad")

        self.target_theta = 0
        self.step_num = 0
        self.action_num = 0
        self.r = 0
        self.epoch_num += 1
        self.theta0_old = 0
        self.theta1_old = 0

        self.target_theta_csv = []
        self.theta0_csv = []
        self.handle_angle_csv = []
        self.target_handle_angle_csv = []
        self.later_error_csv = []
        self.course_error_angle_csv = []
        self.X = []
        self.Y = []
        self.k_lat_csv = []
        self.k_course_csv = []
        _reset_path_progress_trackers(self, self.path_type)

        # ✅ 根据路径类型设置初始位置
        startPos = _get_reset_start_pos(self.path_type, self.epoch_num, init_lateral_offset=0.0, use_random_start=True)

        # Stage3 reset航向模式说明：
        # - legacy: 原方法。保持原来的固定初始朝向。
        # - offset: 新方法。在路径默认起始朝向基础上叠加一个轻量的初始航向偏置，
        #           使第三阶段在不同 reset 语义下保持一致，便于实验组织与结果解释。
        stage3_heading_offset = 0.0 if self.heading_offset_reset_mode == "legacy" else math.radians(5.0)
        yaw0 = _get_reset_start_yaw(
            self.path_type,
            heading_offset_reset_mode=self.heading_offset_reset_mode,
            init_heading_offset=stage3_heading_offset,
        )
        startOrientation = p.getQuaternionFromEuler([0, 0, yaw0])
        p.resetBasePositionAndOrientation(self.bike, startPos, startOrientation)
        p.resetJointState(self.bike, 0, 0, 0)
        p.resetJointState(self.bike, 1, 0, 0)
        p.resetJointState(self.bike, 2, 0, 0)
        p.stepSimulation()
        
        return self.__get_observation(self.target_theta), {}

    def step(self, action):
        """
        ✅ 修复完成判定逻辑
        
        关键修改：
        1. 先判断位置是否完成
        2. 再判断是否失败
        3. 完成优先级 > 失败
        """
        cumulative_reward = 0.0
        terminated = False
        truncated = False
        termination_reason = "running"
        
        k_lat, k_course = self._map_action_to_stanley_gains(action)
        
        if self.record_flag:
            self.k_lat_csv.append(k_lat)
            self.k_course_csv.append(k_course)
        
        # ========== 动作重复循环 ==========
        for repeat_idx in range(self.action_repeat):
            location, _ = p.getBasePositionAndOrientation(self.bike)
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=-0,
                cameraPitch=-89.9,
                cameraTargetPosition=location
            )
            
            state_last = self.__get_observation(recoder=(repeat_idx == 0))
            
            lateral_error = state_last[0] * 10
            course_error_angle = state_last[1] * 1.57
            
            x = math.atan(k_lat * lateral_error / (state_last[2] * 5 + 0.001)) + \
                course_error_angle * k_course
            
            tar_attitude = steady_state_calculation(x, state_last[2] * 5)
            tar_attitude = np.clip(tar_attitude, -math.pi/6, math.pi/6)
            
            attitude_state = self.get_state_attitude_control(tar_attitude, recoder=1)
            
            lqr_action = self.agent_lqr.predict(attitude_state, deterministic=True)[0]
            kp, kd, k = self._map_lqr_action_to_gains(lqr_action)
            
            pid_control = kp * attitude_state[1] * 1.57 + \
                        kd * attitude_state[2] * 10 - \
                        tar_attitude * k
            pid_control = math.atan((pid_control * 2.25 / (4**2)))
            pid_control = np.clip(pid_control, -0.785, 0.785)

            final_handle_angle = pid_control

            p.setJointMotorControl2(self.bike, 0, p.VELOCITY_CONTROL, 
                                    targetVelocity=-10, force=20)
            p.setJointMotorControl2(self.bike, 2, p.VELOCITY_CONTROL, 
                                    targetVelocity=-10, force=20)
            p.setJointMotorControl2(self.bike, 1, p.POSITION_CONTROL,
                                    targetPosition=final_handle_angle, force=200, positionGain=0.3)
            
            p.stepSimulation()
            
            state = self.__get_observation()
            reward = self.__calculate_reward(state_last, state)
            cumulative_reward += reward
            
            # ========== ❌ 删除这里的终止判断！先不判断失败 ==========
            # 让它跑完整个动作重复循环
        
        self.step_num += self.action_repeat
        self.action_num += 1
        
        # ========== ✅ 1. 先获取位置并判断是否完成（最高优先级）==========
        back_wheel_point_x, back_wheel_point_y, _ = p.getLinkState(self.bike, 0)[0]
        forward_wheel_point_x, forward_wheel_point_y, _ = p.getLinkState(self.bike, 2)[0]
        
        x_s = (back_wheel_point_x + forward_wheel_point_x) / 2
        y_s = (back_wheel_point_y + forward_wheel_point_y) / 2
        
        # 计算进度
        progress_percent, path_completed = _compute_path_progress_and_completion(
            self.path_type,
            x_s,
            y_s,
            self.completion_y,
            self.completion_x_min,
            self.completion_x_max,
            self.completion_y_min,
            self.completion_y_max,
            getattr(self, "current_path_segment", 0),
        )
        
        # ========== ✅ 2. 完成判定（最高优先级）==========
        if path_completed:
            termination_reason = "task_complete"
            truncated = True  # 完成用truncated
            
            # 计算平均误差
            if len(self.later_error_csv) > 0:
                mean_lateral = sum(map(abs, self.later_error_csv)) / len(self.later_error_csv)
            else:
                mean_lateral = 1.0
            
            baseline = self.baseline_lateral
            expected_actions = self.max_step_num // self.action_repeat

            if self.path_type == "complex":
                base_completion_reward = 2200.0

                if mean_lateral < baseline * 0.5:
                    quality_bonus = 800.0
                    quality_level = "🏆 卓越"
                elif mean_lateral < baseline * 0.7:
                    quality_bonus = 500.0
                    quality_level = "🌟 优秀"
                elif mean_lateral < baseline * 0.9:
                    quality_bonus = 250.0
                    quality_level = "✅ 良好"
                elif mean_lateral < baseline * 1.1:
                    quality_bonus = 0.0
                    quality_level = "☑️  及格"
                elif mean_lateral < baseline * 1.4:
                    quality_bonus = -150.0
                    quality_level = "⚠️  一般"
                else:
                    quality_bonus = -300.0
                    quality_level = "❌ 偏大"

                if self.action_num < expected_actions * 0.55:
                    speed_bonus = 250.0
                elif self.action_num < expected_actions * 0.75:
                    speed_bonus = 100.0
                elif self.action_num < expected_actions:
                    speed_bonus = 0.0
                else:
                    speed_bonus = -120.0
            else:
                base_completion_reward = 500.0

                if mean_lateral < baseline * 0.5:
                    quality_bonus = 300.0
                    quality_level = "🏆 卓越"
                elif mean_lateral < baseline * 0.7:
                    quality_bonus = 200.0
                    quality_level = "🌟 优秀"
                elif mean_lateral < baseline * 0.9:
                    quality_bonus = 100.0
                    quality_level = "✅ 良好"
                elif mean_lateral < baseline * 1.1:
                    quality_bonus = 50.0
                    quality_level = "☑️  及格"
                elif mean_lateral < baseline * 1.4:
                    quality_bonus = 0.0
                    quality_level = "⚠️  一般"
                elif mean_lateral < baseline * 1.7:
                    quality_bonus = -50.0
                    quality_level = "❌ 较差"
                else:
                    quality_bonus = -100.0
                    quality_level = "💀 很差"

                if self.action_num < expected_actions * 0.7:
                    speed_bonus = 100.0
                elif self.action_num < expected_actions * 0.85:
                    speed_bonus = 50.0
                elif self.action_num < expected_actions:
                    speed_bonus = 0.0
                else:
                    speed_bonus = -50.0
            
            total_completion_reward = base_completion_reward + quality_bonus + speed_bonus
            cumulative_reward += total_completion_reward
            
            print(f"\n🎉 完成！[{self.path_type}] y={y_s:.2f}, x={x_s:.2f}, 动作={self.action_num}")
            print(f"   质量: {quality_level} (误差={mean_lateral:.3f}m, 基线={baseline}m)")
            print(f"   奖励: 基础+{base_completion_reward:.0f} + 质量+{quality_bonus:.0f} + 速度+{speed_bonus:.0f} = {total_completion_reward:.0f}")
        
        # ========== ✅ 3. 失败判定（低优先级，只有未完成时才判断）==========
        elif not path_completed:
            # ✅ 现在才检查失败条件
            if abs(10 * state[0]) >= 6:
                terminated = True
                termination_reason = "lateral_error"
            elif abs(attitude_state[1]) * 1.57 > math.pi/3:
                terminated = True
                termination_reason = "tilt_error"
            elif abs(state[1]) * 1.57 > math.pi/3:
                terminated = True
                termination_reason = "heading_error"
            elif self.step_num >= self.max_step_num:
                terminated = True
                termination_reason = "time_out"
            # 失败惩罚
            if terminated:
                if len(self.later_error_csv) > 0:
                    mean_lateral = sum(map(abs, self.later_error_csv)) / len(self.later_error_csv)
                else:
                    mean_lateral = 1.0
                
                # 早期失败
                if self.action_num < self.early_failure_threshold:
                    early_ratio = self.action_num / self.early_failure_threshold
                    base_early_penalty = -900.0 if self.path_type == "complex" else -400.0
                    
                    if self.action_num < 5:
                        severity_multiplier = 1.6 if self.path_type == "complex" else 1.5
                    elif self.action_num < 10:
                        severity_multiplier = 1.25 if self.path_type == "complex" else 1.2
                    else:
                        severity_multiplier = 1.0
                    
                    early_penalty = base_early_penalty * (1 - early_ratio) * severity_multiplier
                    cumulative_reward += early_penalty
                    
                    print(f"\n❌ 早期失败！[{self.path_type}] 动作={self.action_num}/{self.early_failure_threshold}, "
                        f"惩罚={early_penalty:.1f}, 原因={termination_reason}")
                
                # 中期失败
                elif self.action_num < 150:
                    if self.path_type == "complex":
                        base_penalty = -650.0 * (1 - progress_percent / 100.0)
                        if mean_lateral > 1.0:
                            quality_penalty = -180.0
                        elif mean_lateral > 0.7:
                            quality_penalty = -90.0
                        else:
                            quality_penalty = 0.0
                    else:
                        base_penalty = -200.0 * (1 - progress_percent / 100.0)
                        if mean_lateral > 1.0:
                            quality_penalty = -100.0
                        elif mean_lateral > 0.7:
                            quality_penalty = -50.0
                        else:
                            quality_penalty = 0.0
                    
                    mid_penalty = base_penalty + quality_penalty
                    cumulative_reward += mid_penalty
                    
                    print(f"\n❌ 中期失败！[{self.path_type}] 动作={self.action_num}, 进度={progress_percent:.1f}%, "
                        f"误差={mean_lateral:.3f}m, 惩罚={mid_penalty:.1f}, 原因={termination_reason}")
                
                # 后期失败
                else:
                    if self.path_type == "complex":
                        if progress_percent >= 95:
                            base_penalty = -120.0
                        elif progress_percent >= 90:
                            base_penalty = -220.0
                        elif progress_percent >= 80:
                            base_penalty = -350.0
                        else:
                            base_penalty = -500.0

                        if mean_lateral < self.baseline_lateral:
                            quality_adjustment = 40.0
                        elif mean_lateral > 1.0:
                            quality_adjustment = -60.0
                        else:
                            quality_adjustment = 0.0
                    else:
                        if progress_percent >= 95:
                            base_penalty = -30.0
                        elif progress_percent >= 90:
                            base_penalty = -60.0
                        elif progress_percent >= 80:
                            base_penalty = -100.0
                        else:
                            base_penalty = -150.0

                        if mean_lateral < self.baseline_lateral:
                            quality_adjustment = 20.0
                        elif mean_lateral > 1.0:
                            quality_adjustment = -30.0
                        else:
                            quality_adjustment = 0.0
                    
                    late_penalty = base_penalty + quality_adjustment
                    cumulative_reward += late_penalty
                    
                    print(f"\n❌ 后期失败！[{self.path_type}] 动作={self.action_num}, 进度={progress_percent:.1f}%, "
                        f"误差={mean_lateral:.3f}m, 惩罚={late_penalty:.1f}, 原因={termination_reason}")
        
        # ========== 数据保存 ==========
        if terminated or truncated:
            if len(self.later_error_csv) > 0:
                self.position_error_csv.append(sum(map(abs, self.later_error_csv)) / 
                                            len(self.later_error_csv))
                save_path = _model_path(f"stage3_{self.path_type}_position_error.csv")
                np.savetxt(save_path, np.array([self.position_error_csv]).T, delimiter=',')
            
            if self.record_flag == 1:
                save_path = _model_path(f"stage3_{self.path_type}_data.csv")
                
                path_data_len = len(self.later_error_csv)
                path_data = np.array([
                    self.later_error_csv[:path_data_len],
                    self.course_error_angle_csv[:path_data_len],
                    self.X[:path_data_len],
                    self.Y[:path_data_len],
                    self.k_lat_csv[:path_data_len],
                    self.k_course_csv[:path_data_len]
                ]).T
                
                attitude_data_len = len(self.theta0_csv)
                attitude_data = np.array([
                    self.target_theta_csv[:attitude_data_len],
                    self.theta0_csv[:attitude_data_len],
                    self.handle_angle_csv[:attitude_data_len]
                ]).T
                
                np.savetxt(save_path.replace('.csv', '_path.csv'), path_data, delimiter=',')
                np.savetxt(save_path.replace('.csv', '_attitude.csv'), attitude_data, delimiter=',')
                
                print(f"[Stage 3] 保存数据: {path_data_len} 个动作, {attitude_data_len} 个时间步")
        
        # ========== 返回info ==========
        info = {
            'x': x_s,
            'y': y_s,
            'forward_wheel_x': forward_wheel_point_x,
            'forward_wheel_y': forward_wheel_point_y,
            'termination_reason': termination_reason,
            'path_completed': path_completed,
            'progress_percent': progress_percent,
            'action_num': self.action_num,
            'path_type': self.path_type,
        }
        
        self.r += cumulative_reward
        
        return state, cumulative_reward, terminated, truncated, info
    
    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1


# ==================== 主训练程序 ====================
