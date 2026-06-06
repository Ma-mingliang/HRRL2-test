"""
HRRL2 Stage 1 Reward Function Optimizer
========================================
Iterative reward optimization using paper pool methods.
Runs for 5 hours, committing each improved version to git.

Usage:
    conda run -n RL2 python reward_optimizer.py
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(r"D:\research-agent\HRRL2")
ENV_FILE = PROJECT_ROOT / "env.py"
PYTHON = r"E:\Anaconda\envs\RL2\python.exe"
TOTAL_RUNTIME_HOURS = 5
TIMESTEPS_PER_ITER = 50000
IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement to accept
CATEGORIES_PRIORITY = [
    "F_residual_aware_reward",
    "A_potential_based_reward",
    "B_safety_constraint_reward",
    "E_hierarchical_reward",
    "C_curriculum_subgoal_reward",
    "D_adaptive_dynamic_reward",
]

# ──────────────────────────────────────────────────────────────────────────────
# Reward Modifications — concrete, tested changes
# ──────────────────────────────────────────────────────────────────────────────

REWARD_MODIFICATIONS = [
    # ── Category F: Residual-Aware ──
    {
        "id": "F1_residual_action_penalty",
        "category": "F_residual_aware_reward",
        "name": "Residual action magnitude penalty",
        "description": "Penalize large steering angles proportional to squared magnitude (lambda=0.05)",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -0.05 * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        # F1: residual action squared penalty (lambda=0.05)
        action_penalty = -0.05 * (target_handle_angle / (math.pi / 4)) ** 2
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty
        return reward""",
    },
    {
        "id": "F2_residual_smoothness",
        "category": "F_residual_aware_reward",
        "name": "Residual action smoothness penalty",
        "description": "Penalize change in steering angle between steps (smoothness) + squared action penalty",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -0.05 * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        # F2: squared action penalty + smoothness (track prev action)
        action_penalty = -0.04 * (target_handle_angle / (math.pi / 4)) ** 2
        if not hasattr(self, '_prev_handle_angle'):
            self._prev_handle_angle = 0.0
        delta_action = abs(target_handle_angle - self._prev_handle_angle)
        smooth_action_penalty = -0.03 * delta_action / (math.pi / 4)
        self._prev_handle_angle = target_handle_angle
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty + smooth_action_penalty
        return reward""",
    },
    {
        "id": "F3_adaptive_residual_lambda",
        "category": "F_residual_aware_reward",
        "name": "Adaptive residual lambda (decreases as error shrinks)",
        "description": "Residual penalty lambda decreases as tracking improves, encouraging exploration early",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -0.05 * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        # F3: adaptive lambda - less penalty when error is small
        adaptive_lambda = 0.08 * min(current_error / 0.1, 1.0)
        action_penalty = -adaptive_lambda * (target_handle_angle / (math.pi / 4)) ** 2
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty
        return reward""",
    },

    # ── Category A: Potential-Based ──
    {
        "id": "A1_potential_shaping",
        "category": "A_potential_based_reward",
        "name": "Potential-based reward shaping",
        "description": "Add gamma*Phi(s') - Phi(s) shaping where Phi = -error^2 (policy invariant)",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -0.05 * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)
        # A1: potential-based shaping: gamma * Phi(s') - Phi(s)
        # Phi(s) = -error^2, gamma=0.99
        gamma = 0.99
        phi_last = -(abs(state_last_raw[0])) ** 2
        phi_current = -(current_error) ** 2
        potential_shaping = gamma * phi_current - phi_last
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty + 0.1 * potential_shaping
        return reward""",
    },
    {
        "id": "A2_potential_velocity",
        "category": "A_potential_based_reward",
        "name": "Potential shaping with velocity component",
        "description": "Potential function includes both error and velocity: Phi = -(error^2 + 0.1*omega^2)",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -0.05 * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)
        # A2: potential shaping with velocity: Phi = -(error^2 + 0.1*omega^2)
        gamma = 0.99
        phi_last = -(abs(state_last_raw[0])) ** 2 - 0.1 * (abs(state_last_raw[2])) ** 2
        phi_current = -(current_error) ** 2 - 0.1 * (angular_velocity) ** 2
        potential_shaping = gamma * phi_current - phi_last
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty + 0.1 * potential_shaping
        return reward""",
    },

    # ── Category B: Safety/Constraint ──
    {
        "id": "B1_angular_velocity_penalty",
        "category": "B_safety_constraint_reward",
        "name": "Enhanced angular velocity safety penalty",
        "description": "Quadratic angular velocity penalty (instead of linear) to strongly discourage oscillation",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        # B1: quadratic angular velocity penalty (stronger discouragement of oscillation)
        smoothness_penalty = -0.1 * angular_velocity ** 2
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty
        return reward""",
    },
    {
        "id": "B2_tilt_safety_barrier",
        "category": "B_safety_constraint_reward",
        "name": "Tilt angle safety barrier",
        "description": "Exponential penalty that increases sharply as tilt approaches failure threshold (pi/3)",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])
        tilt_angle = abs(state_raw[1])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -0.05 * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)
        # B2: safety barrier - exponential penalty as tilt approaches failure
        failure_threshold = math.pi / 3  # 60 degrees
        safety_ratio = tilt_angle / failure_threshold
        if safety_ratio > 0.5:
            safety_penalty = -2.0 * (safety_ratio - 0.5) ** 2
        else:
            safety_penalty = 0.0
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty + safety_penalty
        return reward""",
    },
    {
        "id": "B3_combined_safety",
        "category": "B_safety_constraint_reward",
        "name": "Combined safety: barrier + quadratic velocity",
        "description": "Tilt barrier + quadratic angular velocity penalty + action smoothness",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])
        tilt_angle = abs(state_raw[1])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        # B3: quadratic velocity penalty
        smoothness_penalty = -0.08 * angular_velocity ** 2
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.03 * abs(target_handle_angle) / (math.pi / 4)
        # Safety barrier
        failure_threshold = math.pi / 3
        safety_ratio = tilt_angle / failure_threshold
        safety_penalty = -1.5 * max(0, safety_ratio - 0.5) ** 2
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty + safety_penalty
        return reward""",
    },

    # ── Category E: Hierarchical ──
    {
        "id": "E1_hierarchical_error_stages",
        "category": "E_hierarchical_reward",
        "name": "Hierarchical error-stage reward",
        "description": "Different reward scales for coarse (>0.05), medium (0.02-0.05), fine (<0.02) error stages",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        # E1: hierarchical reward based on error stage
        if current_error > 0.05:
            # Coarse stage: focus on fast convergence
            tracking_reward = -3.0 * current_error ** 2
            smoothness_penalty = -0.02 * angular_velocity
            bonus_reward = 0.0
        elif current_error > 0.02:
            # Medium stage: balanced
            tracking_reward = -2.0 * current_error ** 2
            smoothness_penalty = -0.05 * angular_velocity
            bonus_reward = 0.3
        else:
            # Fine stage: focus on precision and stability
            tracking_reward = -5.0 * current_error ** 2
            smoothness_penalty = -0.1 * angular_velocity
            bonus_reward = 1.0 if current_error < 0.005 else 0.5

        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty
        return reward""",
    },
    {
        "id": "E2_progressive_difficulty",
        "category": "E_hierarchical_reward",
        "name": "Progressive difficulty curriculum",
        "description": "Reward scales increase with episode count (curriculum), early episodes are more forgiving",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        # E2: progressive difficulty based on episode count
        progress = min(1.0, self.epoch_num / 50.0)  # ramps up over 50 episodes

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005:
            bonus_reward = 1.0 + 0.5 * progress  # bonus increases with progress
        elif current_error < 0.01:
            bonus_reward = 0.5 + 0.3 * progress
        elif current_error < 0.02:
            bonus_reward = 0.2 + 0.1 * progress
        smoothness_penalty = -(0.02 + 0.06 * progress) * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -(0.01 + 0.02 * progress) * abs(target_handle_angle) / (math.pi / 4)
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty
        return reward""",
    },

    # ── Category C: Curriculum/Subgoal ──
    {
        "id": "C1_subgoal_milestones",
        "category": "C_curriculum_subgoal_reward",
        "name": "Subgoal milestone rewards",
        "description": "Extra bonus for sustained precision (consecutive steps under threshold)",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        tracking_reward = -min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -0.05 * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)
        # C1: sustained precision milestone bonus
        if not hasattr(self, '_precise_steps'):
            self._precise_steps = 0
        if current_error < 0.01:
            self._precise_steps += 1
        else:
            self._precise_steps = 0
        milestone_bonus = 0.0
        if self._precise_steps >= 50: milestone_bonus = 2.0
        elif self._precise_steps >= 20: milestone_bonus = 1.0
        elif self._precise_steps >= 10: milestone_bonus = 0.3
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty + milestone_bonus
        return reward""",
    },

    # ── Category D: Adaptive/Dynamic ──
    {
        "id": "D1_adaptive_weight_combination",
        "category": "D_adaptive_dynamic_reward",
        "name": "Adaptive component weights",
        "description": "Tracking weight increases, smoothness weight decreases as training progresses",
        "code": """
    def __calculate_reward(self, state_last, state, target_handle_angle=0.0):
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])

        # D1: adaptive weights based on episode progress
        progress = min(1.0, self.epoch_num / 80.0)
        tracking_weight = 1.0 + 0.5 * progress  # 1.0 -> 1.5
        smoothness_weight = 0.1 * (1.0 - 0.5 * progress)  # 0.1 -> 0.05

        tracking_reward = -tracking_weight * min(current_error**2, 2.0)
        bonus_reward = 0.0
        if current_error < 0.005: bonus_reward = 1.0
        elif current_error < 0.01: bonus_reward = 0.5
        elif current_error < 0.02: bonus_reward = 0.2
        smoothness_penalty = -smoothness_weight * angular_velocity
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        action_penalty = -0.02 * abs(target_handle_angle) / (math.pi / 4)
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward + action_penalty
        return reward""",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_git(args: list[str], cwd: str = None) -> tuple[int, str]:
    cmd = ["git"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(PROJECT_ROOT))
    return r.returncode, r.stdout.strip() + r.stderr.strip()


def backup_env():
    """Backup env.py before modification."""
    shutil.copy2(ENV_FILE, ENV_FILE.with_suffix(".py.bak"))


def restore_env():
    """Restore env.py from backup."""
    bak = ENV_FILE.with_suffix(".py.bak")
    if bak.exists():
        shutil.copy2(bak, ENV_FILE)


def apply_reward_modification(mod: dict) -> bool:
    """Replace __calculate_reward method in env.py with the modification code."""
    content = ENV_FILE.read_text(encoding="utf-8")
    # Find the __calculate_reward method
    pattern = r'(    def __calculate_reward\(self, state_last, state, target_handle_angle=0\.0\):.*?)(?=\n    def |\nclass |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        log("ERROR: Could not find __calculate_reward method")
        return False
    new_content = content[:match.start()] + mod["code"].strip() + "\n" + content[match.end():]
    ENV_FILE.write_text(new_content, encoding="utf-8")
    return True


def train_and_evaluate(timesteps: int) -> dict | None:
    """Train TD3 and return metrics from last 20 episodes."""
    try:
        # Import fresh env module
        import importlib
        import env as env_module
        importlib.reload(env_module)

        env_instance = env_module.Attitude_control_stage1(render=False)
        env_instance.record_flag = 1
        log_dir = str(PROJECT_ROOT / "model" / "optimizer_logs")
        os.makedirs(log_dir, exist_ok=True)
        env_monitored = Monitor(env_instance, log_dir)

        n_actions = env_monitored.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TD3(
            "MlpPolicy", env=env_monitored, device=device,
            gamma=0.99, learning_rate=0.001, batch_size=128,
            buffer_size=100000, action_noise=action_noise,
            learning_starts=128, train_freq=(1, "step"),
            gradient_steps=-1, policy_delay=2, seed=42, verbose=0
        )

        t0 = time.time()
        model.learn(total_timesteps=timesteps)
        elapsed = time.time() - t0

        # Extract metrics from Monitor logs
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(y) < 5:
            log(f"WARNING: Only {len(y)} episodes completed")
            return None

        last_20 = y[-20:] if len(y) >= 20 else y
        metrics = {
            "mean_reward": float(np.mean(last_20)),
            "std_reward": float(np.std(last_20)),
            "total_episodes": len(y),
            "training_time_s": round(elapsed, 1),
            "timesteps": timesteps,
        }

        # Clean up
        env_monitored.close()

        # Remove log files to avoid confusion
        for f in Path(log_dir).glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

        return metrics

    except Exception as e:
        log(f"ERROR in train_and_evaluate: {e}")
        traceback.print_exc()
        return None


def commit_and_push(version: int, mod: dict, metrics: dict, baseline_metrics: dict):
    """Commit the improved reward function and push to GitHub."""
    # Calculate improvement
    improvement = (baseline_metrics["mean_reward"] - metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100
    # Note: for reward, higher is better (less negative), so improvement > 0 means better

    run_git(["add", "env.py"])
    commit_msg = (
        f"v{version}: {mod['name']}\n\n"
        f"Category: {mod['category']}\n"
        f"Modification ID: {mod['id']}\n"
        f"Description: {mod['description']}\n\n"
        f"Metrics (last 20 episodes):\n"
        f"  Mean reward: {metrics['mean_reward']:.2f} (baseline: {baseline_metrics['mean_reward']:.2f})\n"
        f"  Improvement: {improvement:+.1f}%\n"
        f"  Episodes: {metrics['total_episodes']}\n"
        f"  Training time: {metrics['training_time_s']:.0f}s"
    )
    run_git(["commit", "-m", commit_msg])
    run_git(["tag", f"v{version}"])
    run_git(["push", "origin", "main", "--tags"])
    log(f"Committed and pushed v{version}")


def rollback(version: int, mod: dict, metrics: dict, baseline_metrics: dict):
    """Rollback to previous env.py."""
    restore_env()
    run_git(["checkout", "env.py"])
    log(f"Rolled back v{version} ({mod['id']}): no improvement")


def update_changelog(version: int, mod: dict, metrics: dict, baseline_metrics: dict, accepted: bool):
    """Append to CHANGELOG.md."""
    changelog = PROJECT_ROOT / "CHANGELOG.md"
    if not changelog.exists():
        changelog.write_text("# HRRL2 Stage 1 Reward Optimization Changelog\n\n", encoding="utf-8")

    improvement = (baseline_metrics["mean_reward"] - metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100
    status = "ACCEPTED" if accepted else "REJECTED"

    entry = (
        f"## v{version} — {mod['name']} [{status}]\n\n"
        f"- **Category**: {mod['category']}\n"
        f"- **ID**: {mod['id']}\n"
        f"- **Description**: {mod['description']}\n"
        f"- **Mean Reward**: {metrics['mean_reward']:.2f} "
        f"(baseline: {baseline_metrics['mean_reward']:.2f}, change: {improvement:+.1f}%)\n"
        f"- **Episodes**: {metrics['total_episodes']}\n"
        f"- **Training Time**: {metrics['training_time_s']:.0f}s\n"
        f"- **Timestamp**: {datetime.now(timezone.utc).isoformat()}\n\n"
    )

    with open(changelog, "a", encoding="utf-8") as f:
        f.write(entry)


def run_baseline(timesteps: int) -> dict | None:
    """Run baseline training with original reward function."""
    log("Running baseline training...")
    metrics = train_and_evaluate(timesteps)
    if metrics:
        log(f"Baseline: mean_reward={metrics['mean_reward']:.2f}, "
            f"episodes={metrics['total_episodes']}, time={metrics['training_time_s']:.0f}s")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    end_time = start_time + TOTAL_RUNTIME_HOURS * 3600

    log("=" * 70)
    log("HRRL2 Stage 1 Reward Optimizer")
    log(f"Runtime: {TOTAL_RUNTIME_HOURS} hours")
    log(f"Timesteps per iteration: {TIMESTEPS_PER_ITER}")
    log(f"Improvement threshold: {IMPROVEMENT_THRESHOLD*100:.0f}%")
    log(f"Categories: {', '.join(CATEGORIES_PRIORITY)}")
    log("=" * 70)

    # Backup original env.py
    backup_env()

    # Run baseline
    baseline = run_baseline(TIMESTEPS_PER_ITER)
    if not baseline:
        log("FATAL: Baseline training failed")
        return 1

    best_metrics = baseline.copy()
    best_version = 0
    version = 0

    # Initialize CHANGELOG
    changelog_path = PROJECT_ROOT / "CHANGELOG.md"
    changelog_path.write_text(
        "# HRRL2 Stage 1 Reward Optimization Changelog\n\n"
        f"Started: {datetime.now(timezone.utc).isoformat()}\n"
        f"Runtime: {TOTAL_RUNTIME_HOURS} hours\n"
        f"Timesteps per iteration: {TIMESTEPS_PER_ITER}\n\n"
        f"## v0 — Baseline [ACCEPTED]\n\n"
        f"- Mean Reward: {baseline['mean_reward']:.2f}\n"
        f"- Episodes: {baseline['total_episodes']}\n"
        f"- Training Time: {baseline['training_time_s']:.0f}s\n\n",
        encoding="utf-8"
    )

    # Main optimization loop
    for mod in REWARD_MODIFICATIONS:
        # Check time
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = TOTAL_RUNTIME_HOURS - elapsed_hours
        if remaining_hours < 0.15:  # ~9 minutes left
            log(f"Time's up! Elapsed: {elapsed_hours:.1f}h")
            break

        version += 1
        log(f"\n{'='*70}")
        log(f"v{version}: {mod['id']} — {mod['name']}")
        log(f"Category: {mod['category']} | Remaining: {remaining_hours:.1f}h")
        log(f"{'='*70}")

        # Apply modification
        if not apply_reward_modification(mod):
            log(f"SKIP: Failed to apply modification")
            version -= 1
            continue

        # Train and evaluate
        metrics = train_and_evaluate(TIMESTEPS_PER_ITER)
        if not metrics:
            log(f"SKIP: Training failed")
            restore_env()
            version -= 1
            continue

        # Compare with best
        # Reward: higher is better (less negative)
        improvement = (best_metrics["mean_reward"] - metrics["mean_reward"]) / abs(best_metrics["mean_reward"])
        # Note: since rewards are negative, "improvement" means less negative (higher)
        # So we check if metrics["mean_reward"] > best_metrics["mean_reward"]
        actual_improvement = (metrics["mean_reward"] - best_metrics["mean_reward"]) / abs(best_metrics["mean_reward"])

        log(f"Result: reward={metrics['mean_reward']:.2f} vs best={best_metrics['mean_reward']:.2f} "
            f"(change: {actual_improvement*100:+.1f}%)")

        if actual_improvement > IMPROVEMENT_THRESHOLD:
            log(f"ACCEPTED: {actual_improvement*100:.1f}% improvement > {IMPROVEMENT_THRESHOLD*100:.0f}% threshold")
            commit_and_push(version, mod, metrics, best_metrics)
            update_changelog(version, mod, metrics, best_metrics, accepted=True)
            best_metrics = metrics.copy()
            best_version = version
        else:
            log(f"REJECTED: {actual_improvement*100:.1f}% < {IMPROVEMENT_THRESHOLD*100:.0f}% threshold")
            rollback(version, mod, metrics, best_metrics)
            update_changelog(version, mod, metrics, best_metrics, accepted=False)

    # Final summary
    elapsed_total = (time.time() - start_time) / 3600
    log(f"\n{'='*70}")
    log("OPTIMIZATION COMPLETE")
    log(f"Total time: {elapsed_total:.1f}h")
    log(f"Versions tried: {version}")
    log(f"Best version: v{best_version}")
    log(f"Best reward: {best_metrics['mean_reward']:.2f} (baseline: {baseline['mean_reward']:.2f})")
    overall_improvement = (best_metrics["mean_reward"] - baseline["mean_reward"]) / abs(baseline["mean_reward"]) * 100
    log(f"Overall improvement: {overall_improvement:+.1f}%")
    log(f"{'='*70}")

    # Push final changelog
    run_git(["add", "CHANGELOG.md"])
    run_git(["commit", "-m", f"changelog: optimization complete ({version} versions, {elapsed_total:.1f}h)"])
    run_git(["push", "origin", "main"])

    # Tag best version
    if best_version > 0:
        log(f"Best version is v{best_version} (tag already set)")

    # Restore best version if it's not the last one
    if best_version > 0 and best_version < version:
        log(f"Restoring best version v{best_version}...")
        run_git(["checkout", f"v{best_version}", "--", "env.py"])
        run_git(["add", "env.py"])
        run_git(["commit", "-m", f"restore: best version v{best_version}"])
        run_git(["push", "origin", "main"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
