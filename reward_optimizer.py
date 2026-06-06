"""
HRRL2 Stage 1 Reward Function Optimizer (LLM-Driven)
=====================================================
Universal reward optimization using paper pool + LLM generation.
Based on Eureka (NVIDIA) best practices.

Usage:
    set MIMO_API_KEY=xxx && python reward_optimizer.py
    or: conda run -n RL2 python reward_optimizer.py
"""

import csv
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
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(r"D:\research-agent\HRRL2")
ENV_FILE = PROJECT_ROOT / "env.py"
POOL_DIR = Path(r"D:\research-agent\research_agent\reward_paper_pool")
WORK_DIR = PROJECT_ROOT / "optimizer_work"
RESULTS_DIR = PROJECT_ROOT / "optimizer_results"

TOTAL_RUNTIME_HOURS = 5
TIMESTEPS_SCREEN = 30000   # Screening: quick train
TIMESTEPS_FULL = 50000     # Full: standard train
EVAL_EPISODES = 50         # Evaluation episodes
IMPROVEMENT_THRESHOLD = 0.001  # 0.1% improvement to accept

# MIMO API config (from Hermes)
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"

# Category priority for phase-based flow
CATEGORY_PRIORITY = [
    ("F_residual_aware_reward", "S"),
    ("E_hierarchical_reward", "S"),
    ("A_potential_based_reward", "S"),
    ("B_safety_constraint_reward", "S"),
    ("C_curriculum_subgoal_reward", "A"),
    ("D_adaptive_dynamic_reward", "A"),
    ("G_llm_reward_generation", "A"),
    ("H_learned_preference_reward", "B"),
]

ENV_CONTEXT = """Environment: Attitude_control_stage1 (PyBullet bicycle self-balancing)
- Observation space: [dis_angle, theta0, w0, v] (normalized to [-1, 1])
- Action space: [handle_angle] (normalized to [-1, 1])
- __observation_reduction(state): [dis_angle*1.57, theta0*1.57, w0*10, v*5]
- self.epoch_num: current episode count
- self.step_num: current step in episode
- self.max_step_num: 1000 (max steps per episode)
- self.FAILURE_PENALTY: -10.0
- self.EARLY_TERMINATION_PENALTY: -20.0
- Goal: minimize dis_angle (tilt from vertical), keep bicycle balanced"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_git(args: list, cwd: str = None) -> tuple:
    cmd = ["git"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(PROJECT_ROOT))
    return r.returncode, r.stdout.strip() + r.stderr.strip()


def backup_env():
    shutil.copy2(ENV_FILE, ENV_FILE.with_suffix(".py.bak"))


def restore_env():
    bak = ENV_FILE.with_suffix(".py.bak")
    if bak.exists():
        shutil.copy2(bak, ENV_FILE)


def get_current_reward_code() -> str:
    """Extract current __calculate_reward method from env.py."""
    content = ENV_FILE.read_text(encoding="utf-8")
    pattern = r'(    def __calculate_reward\(self, state_last, state, target_handle_angle=0\.0\):.*?)(?=\n    def |\nclass |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(0).strip("\n") if match else ""


def apply_reward_code(code: str) -> bool:
    """Replace __calculate_reward method in env.py."""
    content = ENV_FILE.read_text(encoding="utf-8")
    pattern = r'(    def __calculate_reward\(self, state_last, state, target_handle_angle=0\.0\):.*?)(?=\n    def |\nclass |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        log("ERROR: Could not find __calculate_reward method")
        return False

    # Normalize indentation: ensure method starts with 4 spaces (class method level)
    lines = code.strip("\n").split("\n")
    if lines:
        # Find the def line
        def_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def __calculate_reward"):
                def_idx = i
                break

        if def_idx is not None:
            # Get the indentation of the def line
            def_line = lines[def_idx]
            current_indent = len(def_line) - len(def_line.lstrip())
            target_indent = 4  # Class method indentation

            # Calculate difference
            indent_diff = target_indent - current_indent

            # Apply indentation correction
            if indent_diff != 0:
                new_lines = []
                for line in lines:
                    if line.strip():  # Non-empty line
                        current = len(line) - len(line.lstrip())
                        new_indent = max(0, current + indent_diff)
                        new_lines.append(" " * new_indent + line.lstrip())
                    else:
                        new_lines.append("")
                code = "\n".join(new_lines)

    new_code = code.strip("\n")
    new_content = content[:match.start()] + new_code + "\n" + content[match.end():]
    ENV_FILE.write_text(new_content, encoding="utf-8")
    return True


def check_syntax(code: str) -> Optional[str]:
    """Check Python syntax of code. Returns error message or None."""
    try:
        compile(code, "<reward>", "exec")
        return None
    except SyntaxError as e:
        return str(e)


def commit_and_push(version: int, mod: dict, metrics: dict, baseline_metrics: dict):
    """Commit and push to GitHub."""
    improvement = (metrics["mean_reward"] - baseline_metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100

    run_git(["add", "env.py"])
    commit_msg = (
        f"v{version}: {mod.get('name', 'LLM Generated')}\n\n"
        f"Category: {mod.get('category', 'N/A')}\n"
        f"Method ID: {mod.get('method_id', 'N/A')}\n"
        f"Description: {mod.get('description', 'N/A')}\n\n"
        f"Metrics:\n"
        f"  Mean reward: {metrics['mean_reward']:.2f} (baseline: {baseline_metrics['mean_reward']:.2f})\n"
        f"  Improvement: {improvement:+.1f}%\n"
        f"  Completion rate: {metrics.get('completion_rate', 'N/A')}\n"
        f"  Comprehensive score: {metrics.get('comprehensive_score', 'N/A')}\n"
        f"  Episodes: {metrics.get('total_episodes', 'N/A')}\n"
        f"  Training time: {metrics.get('training_time_s', 0):.0f}s"
    )
    run_git(["commit", "-m", commit_msg])
    run_git(["tag", "-f", f"v{version}"])
    run_git(["push", "origin", "main", "--tags"])
    log(f"Committed and pushed v{version}")


def save_results_csv(version: int, mod: dict, metrics: dict, baseline_metrics: dict, accepted: bool):
    """Append results to CSV file."""
    csv_path = RESULTS_DIR / "optimization_results.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    improvement = (metrics["mean_reward"] - baseline_metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100

    row = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": mod.get("name", ""),
        "category": mod.get("category", ""),
        "method_id": mod.get("method_id", ""),
        "description": mod.get("description", ""),
        "mean_reward": round(metrics["mean_reward"], 2),
        "std_reward": round(metrics.get("std_reward", 0), 2),
        "improvement_pct": round(improvement, 1),
        "completion_rate": round(metrics.get("completion_rate", 0), 3),
        "comprehensive_score": round(metrics.get("comprehensive_score", 0), 4),
        "total_episodes": metrics.get("total_episodes", 0),
        "training_time_s": metrics.get("training_time_s", 0),
        "accepted": accepted,
    }

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_results_json(version: int, mod: dict, metrics: dict, baseline_metrics: dict, accepted: bool):
    """Save detailed results to JSON file."""
    json_path = RESULTS_DIR / "optimization_results.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    improvement = (metrics["mean_reward"] - baseline_metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100

    record = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": mod.get("name", ""),
        "category": mod.get("category", ""),
        "method_id": mod.get("method_id", ""),
        "description": mod.get("description", ""),
        "mean_reward": round(metrics["mean_reward"], 2),
        "std_reward": round(metrics.get("std_reward", 0), 2),
        "improvement_pct": round(improvement, 1),
        "completion_rate": round(metrics.get("completion_rate", 0), 3),
        "comprehensive_score": round(metrics.get("comprehensive_score", 0), 4),
        "total_episodes": metrics.get("total_episodes", 0),
        "training_time_s": metrics.get("training_time_s", 0),
        "accepted": accepted,
        "metrics": metrics,
    }

    with open(json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_changelog(version: int, mod: dict, metrics: dict, baseline_metrics: dict, accepted: bool):
    """Append to CHANGELOG.md."""
    changelog = PROJECT_ROOT / "CHANGELOG.md"
    improvement = (metrics["mean_reward"] - baseline_metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100
    status = "ACCEPTED" if accepted else "REJECTED"

    entry = (
        f"## v{version} — {mod.get('name', 'LLM Generated')} [{status}]\n\n"
        f"- **Category**: {mod.get('category', 'N/A')}\n"
        f"- **Method ID**: {mod.get('method_id', 'N/A')}\n"
        f"- **Description**: {mod.get('description', 'N/A')}\n"
        f"- **Mean Reward**: {metrics['mean_reward']:.2f} "
        f"(baseline: {baseline_metrics['mean_reward']:.2f}, change: {improvement:+.1f}%)\n"
        f"- **Completion Rate**: {metrics.get('completion_rate', 0):.1%}\n"
        f"- **Comprehensive Score**: {metrics.get('comprehensive_score', 0):.4f}\n"
        f"- **Episodes**: {metrics.get('total_episodes', 'N/A')}\n"
        f"- **Training Time**: {metrics.get('training_time_s', 0):.0f}s\n"
        f"- **Timestamp**: {datetime.now(timezone.utc).isoformat()}\n\n"
    )

    with open(changelog, "a", encoding="utf-8") as f:
        f.write(entry)


# ──────────────────────────────────────────────────────────────────────────────
# Main Optimizer Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    end_time = start_time + TOTAL_RUNTIME_HOURS * 3600

    # Validate MIMO API key
    if not MIMO_API_KEY:
        # Try to load from Hermes
        try:
            hermes_path = Path(r"C:\Users\lenovo_mml\.hermes\auth.json")
            if hermes_path.exists():
                with open(hermes_path, encoding="utf-8") as f:
                    auth = json.load(f)
                xiaomi = auth.get("credential_pool", {}).get("xiaomi", [{}])[0]
                api_key = xiaomi.get("access_token", "")
                base_url = xiaomi.get("base_url", MIMO_BASE_URL)
                if api_key:
                    os.environ["MIMO_API_KEY"] = api_key
                    log(f"Loaded MIMO API key from Hermes")
                else:
                    log("FATAL: No MIMO API key found")
                    return 1
            else:
                log("FATAL: MIMO_API_KEY not set and Hermes auth.json not found")
                return 1
        except Exception as e:
            log(f"FATAL: Failed to load API key: {e}")
            return 1

    # Initialize components
    from paper_sampler import PaperSampler
    from llm_reward_generator import LLMRewardGenerator
    from model_evaluator import ModelEvaluator

    api_key = os.environ.get("MIMO_API_KEY", MIMO_API_KEY)
    sampler = PaperSampler(POOL_DIR, WORK_DIR)
    generator = LLMRewardGenerator(api_key, MIMO_BASE_URL, MIMO_MODEL)
    evaluator = ModelEvaluator(PROJECT_ROOT, EVAL_EPISODES)

    log("=" * 70)
    log("HRRL2 Stage 1 Reward Optimizer (LLM-Driven)")
    log(f"Runtime: {TOTAL_RUNTIME_HOURS} hours")
    log(f"Screening timesteps: {TIMESTEPS_SCREEN}")
    log(f"Full training timesteps: {TIMESTEPS_FULL}")
    log(f"Evaluation episodes: {EVAL_EPISODES}")
    log(f"Improvement threshold: {IMPROVEMENT_THRESHOLD*100:.1f}%")
    log(f"API: {MIMO_BASE_URL}")
    log(f"Paper pool: {sampler.summary()['total']} methods, {len(sampler.remaining_categories())} categories")
    log("=" * 70)

    # Backup original env.py
    backup_env()

    # Initialize CHANGELOG
    changelog_path = PROJECT_ROOT / "CHANGELOG.md"
    changelog_path.write_text(
        "# HRRL2 Stage 1 Reward Optimization Changelog (LLM-Driven)\n\n"
        f"Started: {datetime.now(timezone.utc).isoformat()}\n"
        f"Runtime: {TOTAL_RUNTIME_HOURS} hours\n"
        f"Screening timesteps: {TIMESTEPS_SCREEN}\n"
        f"Full training timesteps: {TIMESTEPS_FULL}\n"
        f"Evaluation episodes: {EVAL_EPISODES}\n\n",
        encoding="utf-8"
    )

    # Run baseline — train once, save as best_model_v0.zip
    log("Running baseline training...")
    baseline_model_path = RESULTS_DIR / "best_model_v0.zip"
    baseline = evaluator.train_model(TIMESTEPS_FULL, save_path=baseline_model_path)
    if not baseline:
        log("FATAL: Baseline training failed")
        return 1

    # Evaluate baseline model for comprehensive metrics
    baseline_eval = evaluator.evaluate_model(baseline_model_path)
    if baseline_eval:
        baseline["completion_rate"] = baseline_eval.get("completion_rate", 0)
        baseline["comprehensive_score"] = baseline_eval.get("comprehensive_score", 0)
        baseline["accuracy"] = baseline_eval.get("accuracy", 0)
        baseline["stability"] = baseline_eval.get("stability", 0)
        baseline["efficiency"] = baseline_eval.get("efficiency", 0)
    else:
        baseline["completion_rate"] = 0
        baseline["comprehensive_score"] = 0

    log(f"Baseline: reward={baseline['mean_reward']:.2f}, "
        f"episodes={baseline['total_episodes']}, time={baseline['training_time_s']:.0f}s")

    # Save baseline model
    baseline_model_path = RESULTS_DIR / "best_model_v0.zip"
    evaluator.train_model(TIMESTEPS_FULL, baseline_model_path)

    # Write baseline to CHANGELOG
    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write(
            f"## v0 — Baseline [ACCEPTED]\n\n"
            f"- Mean Reward: {baseline['mean_reward']:.2f}\n"
            f"- Episodes: {baseline['total_episodes']}\n"
            f"- Training Time: {baseline['training_time_s']:.0f}s\n\n"
        )

    best_metrics = baseline.copy()
    best_version = 0
    version = 0
    history = []

    # Phase-based flow
    phases = [
        ("discovery", "Screening all S-priority categories", 2),
        ("deep_dive", "Deep dive best category", 4),
        ("expand", "Expand to A/B categories", 3),
    ]

    current_phase = 0
    consecutive_failures = 0
    best_category = None

    # Main loop
    while True:
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = TOTAL_RUNTIME_HOURS - elapsed_hours
        if remaining_hours < 0.15:
            log(f"Time's up! Elapsed: {elapsed_hours:.1f}h")
            break

        # Determine phase
        if current_phase >= len(phases):
            current_phase = len(phases) - 1  # Stay in last phase

        phase_name, phase_desc, batch_per_category = phases[current_phase]
        log(f"\n{'='*70}")
        log(f"Phase: {phase_name} — {phase_desc}")
        log(f"Remaining: {remaining_hours:.1f}h | Version: {version}")
        log(f"Best so far: v{best_version} ({best_metrics['mean_reward']:.2f})")
        log(f"{'='*70}")

        # Get next method batch
        if phase_name == "deep_dive" and best_category:
            batch = sampler.get_next_batch(1, preferred_category=best_category)
        else:
            batch = sampler.get_next_batch(1)

        if not batch:
            log("No more methods available, moving to next phase")
            current_phase += 1
            consecutive_failures = 0
            continue

        method = batch[0]
        version += 1

        log(f"v{version}: {method.get('method_name', 'N/A')}")
        log(f"  Category: {method.get('category', 'N/A')}")
        log(f"  Core idea: {method.get('core_idea', 'N/A')[:100]}")
        log(f"  Formula: {method.get('reward_formula', 'N/A')}")
        if method.get("_paper_md"):
            log(f"  Paper: loaded ({len(method['_paper_md'])} chars)")
        else:
            log(f"  Paper: not available")

        # Generate modification
        current_code = get_current_reward_code()
        error_feedback = None

        for attempt in range(3):  # Max 3 attempts to fix syntax
            mod = generator.generate_modification(
                current_code, method, ENV_CONTEXT, baseline, history, error_feedback
            )

            if not mod:
                log(f"  LLM generation failed (attempt {attempt+1})")
                continue

            # Check syntax
            syntax_error = check_syntax(mod["code"])
            if syntax_error:
                log(f"  Syntax error (attempt {attempt+1}): {syntax_error[:80]}")
                error_feedback = syntax_error
                continue

            break
        else:
            log(f"  SKIP: Failed to generate valid code after 3 attempts")
            sampler.mark_used(method["method_id"], method.get("category", ""), "syntax_fail")
            consecutive_failures += 1
            if consecutive_failures >= 5:
                current_phase += 1
                consecutive_failures = 0
            continue

        # Apply modification
        if not apply_reward_code(mod["code"]):
            log(f"  SKIP: Failed to apply code")
            sampler.mark_used(method["method_id"], method.get("category", ""), "apply_fail")
            continue

        # Screening: quick train
        log(f"  Screening ({TIMESTEPS_SCREEN} steps)...")
        screen_metrics = evaluator.quick_evaluate(TIMESTEPS_SCREEN)
        if not screen_metrics:
            log(f"  SKIP: Screening failed")
            restore_env()
            sampler.mark_used(method["method_id"], method.get("category", ""), "screen_fail")
            continue

        screen_improvement = (screen_metrics["mean_reward"] - baseline["mean_reward"]) / abs(baseline["mean_reward"])
        log(f"  Screen: reward={screen_metrics['mean_reward']:.2f} ({screen_improvement*100:+.1f}%), "
            f"completion={screen_metrics.get('completion_rate', 0):.0%}")

        # Reject if screening shows significant regression
        if screen_improvement < -0.10:  # More than 10% regression
            log(f"  REJECTED: Screen regression ({screen_improvement*100:+.1f}%)")
            restore_env()
            sampler.mark_used(method["method_id"], method.get("category", ""), "screen_reject")
            history.append({
                "version": version,
                "name": mod.get("name", ""),
                "mean_reward": screen_metrics["mean_reward"],
                "change_pct": screen_improvement * 100,
                "accepted": False,
            })
            consecutive_failures += 1
            if consecutive_failures >= 5:
                current_phase += 1
                consecutive_failures = 0
            continue

        # Full training
        log(f"  Full training ({TIMESTEPS_FULL} steps)...")
        model_path = RESULTS_DIR / f"model_v{version}.zip"
        full_metrics = evaluator.train_model(TIMESTEPS_FULL, save_path=model_path)
        if not full_metrics:
            log(f"  SKIP: Full training failed")
            restore_env()
            sampler.mark_used(method["method_id"], method.get("category", ""), "train_fail")
            continue

        # Full evaluation
        full_eval = evaluator.evaluate_model(model_path)
        if full_eval:
            full_metrics.update({
                "completion_rate": full_eval["completion_rate"],
                "comprehensive_score": full_eval["comprehensive_score"],
                "accuracy": full_eval["accuracy"],
                "stability": full_eval["stability"],
                "efficiency": full_eval["efficiency"],
            })

        full_improvement = (full_metrics["mean_reward"] - baseline["mean_reward"]) / abs(baseline["mean_reward"])

        log(f"  Full: reward={full_metrics['mean_reward']:.2f} ({full_improvement*100:+.1f}%), "
            f"completion={full_metrics.get('completion_rate', 0):.1%}, "
            f"score={full_metrics.get('comprehensive_score', 0):.4f}")

        # Accept/reject
        if full_improvement > IMPROVEMENT_THRESHOLD:
            log(f"  ACCEPTED: {full_improvement*100:+.1f}% improvement")
            accepted = True
            sampler.mark_used(method["method_id"], method.get("category", ""), "accepted")

            # Save as best model
            best_model_path = RESULTS_DIR / f"best_model_v{version}.zip"
            if model_path.exists():
                shutil.copy2(model_path, best_model_path)

            # Update best
            if full_metrics["mean_reward"] > best_metrics["mean_reward"]:
                best_metrics = full_metrics.copy()
                best_version = version
                best_category = method.get("category")

            # Commit and push
            commit_and_push(version, mod, full_metrics, baseline)
            consecutive_failures = 0

            # Phase transition: if we found a good result in discovery, deep dive it
            if current_phase == 0 and full_improvement > 0.05:
                current_phase = 1  # Move to deep_dive
                best_category = method.get("category")
                log(f"  Phase transition: discovery → deep_dive ({best_category})")
        else:
            log(f"  REJECTED: {full_improvement*100:+.1f}% improvement (threshold: {IMPROVEMENT_THRESHOLD*100:.1f}%)")
            accepted = False
            restore_env()
            sampler.mark_used(method["method_id"], method.get("category", ""), "rejected")
            consecutive_failures += 1

            # Phase transition: too many failures, expand to other categories
            if consecutive_failures >= 3 and current_phase == 1:
                current_phase = 2  # Move to expand
                log(f"  Phase transition: deep_dive → expand")

        # Record history
        history.append({
            "version": version,
            "name": mod.get("name", ""),
            "mean_reward": full_metrics["mean_reward"],
            "change_pct": full_improvement * 100,
            "accepted": accepted,
        })

        # Save results
        save_results_csv(version, mod, full_metrics, baseline, accepted)
        save_results_json(version, mod, full_metrics, baseline, accepted)
        update_changelog(version, mod, full_metrics, baseline, accepted)

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
    log(f"Best completion rate: {best_metrics.get('completion_rate', 0):.1%}")
    log(f"Best comprehensive score: {best_metrics.get('comprehensive_score', 0):.4f}")
    log(f"Results saved to: {RESULTS_DIR}")
    log(f"{'='*70}")

    # Push final changelog
    run_git(["add", "CHANGELOG.md", "optimizer_results/"])
    run_git(["commit", "-m", f"changelog: optimization complete ({version} versions, {elapsed_total:.1f}h)"])
    run_git(["push", "origin", "main"])

    # Print sampler summary
    summary = sampler.summary()
    log(f"\nMethod pool usage:")
    for cat, stats in summary["by_category"].items():
        log(f"  {cat}: {stats['used']}/{stats['total']} used")

    return 0


if __name__ == "__main__":
    sys.exit(main())
