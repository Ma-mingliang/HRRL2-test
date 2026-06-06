"""ModelEvaluator: Consistent evaluation protocol for reward optimization."""

import os
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


class BestModelCallback(BaseCallback):
    """Callback that evaluates every N steps and saves the best model by 10-episode average reward."""

    def __init__(self, eval_freq: int = 5000, eval_episodes: int = 10, save_path: Optional[Path] = None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_path = save_path
        self.best_mean_reward = -float("inf")
        self.best_step = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate()
            if self.verbose:
                print(f"  [BestModel] step={self.n_calls} mean_reward={mean_reward:.2f} (best={self.best_mean_reward:.2f})")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_step = self.n_calls
                if self.save_path:
                    self.save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(self.save_path))
                    if self.verbose:
                        print(f"  [BestModel] Saved new best model at step {self.n_calls}")
        return True

    def _evaluate(self) -> float:
        import importlib
        import env as env_module
        importlib.reload(env_module)

        rewards = []
        for _ in range(self.eval_episodes):
            env = env_module.Attitude_control_stage1(render=False)
            obs, _ = env.reset()
            done = False
            total_r = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_r += reward
                done = terminated or truncated
            rewards.append(total_r)
            env.close()
        return float(np.mean(rewards))


class ModelEvaluator:
    """Evaluates trained models with consistent protocol."""

    def __init__(self, project_root: Path, eval_episodes: int = 50):
        self.project_root = project_root
        self.eval_episodes = eval_episodes

    def train_model(self, timesteps: int, save_path: Optional[Path] = None) -> Optional[dict]:
        """Train a TD3 model and return training metrics + save best model.

        Uses BestModelCallback to save the model with highest 10-episode average reward
        during training, not just the final model.

        Args:
            timesteps: Number of training timesteps.
            save_path: Path to save the best model (optional).

        Returns:
            Dict with training metrics or None if failed.
        """
        try:
            import importlib
            import env as env_module
            importlib.reload(env_module)

            env_instance = env_module.Attitude_control_stage1(render=False)
            env_instance.record_flag = 1
            log_dir = str(self.project_root / "model" / "eval_logs")
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

            # Setup best model callback (eval every 5000 steps, 10 episodes)
            callback = None
            if save_path:
                callback = BestModelCallback(
                    eval_freq=5000, eval_episodes=10,
                    save_path=save_path, verbose=1
                )

            t0 = time.time()
            model.learn(total_timesteps=timesteps, callback=callback)
            elapsed = time.time() - t0

            # If no save_path, save final model
            if save_path and not save_path.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(save_path))

            # Extract metrics from Monitor logs
            x, y = ts2xy(load_results(log_dir), "timesteps")
            if len(y) < 5:
                print(f"[Eval] WARNING: Only {len(y)} episodes completed")
                env_monitored.close()
                self._clean_logs(log_dir)
                return None

            last_20 = y[-20:] if len(y) >= 20 else y
            metrics = {
                "mean_reward": float(np.mean(last_20)),
                "std_reward": float(np.std(last_20)),
                "total_episodes": len(y),
                "training_time_s": round(elapsed, 1),
                "timesteps": timesteps,
            }

            if callback:
                metrics["best_eval_reward"] = callback.best_mean_reward
                metrics["best_eval_step"] = callback.best_step

            env_monitored.close()
            self._clean_logs(log_dir)
            return metrics

        except Exception as e:
            print(f"[Eval] Training error: {e}")
            traceback.print_exc()
            return None

    def evaluate_model(self, model_path: Path) -> Optional[dict]:
        """Evaluate a saved model with consistent protocol.

        Args:
            model_path: Path to the saved model .zip file.

        Returns:
            Dict with evaluation metrics including completion_rate.
        """
        try:
            import importlib
            import env as env_module
            importlib.reload(env_module)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = TD3.load(str(model_path), device=device)

            episode_rewards = []
            episode_lengths = []
            completed_episodes = 0

            for ep in range(self.eval_episodes):
                env_instance = env_module.Attitude_control_stage1(render=False)
                obs, _ = env_instance.reset()
                done = False
                total_reward = 0.0
                steps = 0

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env_instance.step(action)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                episode_rewards.append(total_reward)
                episode_lengths.append(steps)

                # Check if episode completed successfully (not terminated early)
                if steps >= env_instance.max_step_num:
                    completed_episodes += 1

                env_instance.close()

            completion_rate = completed_episodes / self.eval_episodes
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            mean_length = float(np.mean(episode_lengths))

            # Comprehensive scoring
            # accuracy: normalized mean reward (higher is better)
            # stability: inverse of std (lower std = more stable)
            # efficiency: completion rate
            accuracy = max(0, min(1, (mean_reward + 500) / 2000))  # Normalize to [0,1]
            stability = max(0, 1 - std_reward / max(abs(mean_reward), 1))
            efficiency = completion_rate

            comprehensive_score = 0.5 * accuracy + 0.3 * stability + 0.2 * efficiency

            return {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_episode_length": mean_length,
                "completion_rate": completion_rate,
                "completed_episodes": completed_episodes,
                "total_episodes": self.eval_episodes,
                "accuracy": accuracy,
                "stability": stability,
                "efficiency": efficiency,
                "comprehensive_score": comprehensive_score,
                "episode_rewards": episode_rewards,
            }

        except Exception as e:
            print(f"[Eval] Evaluation error: {e}")
            traceback.print_exc()
            return None

    def quick_evaluate(self, timesteps: int) -> Optional[dict]:
        """Train and evaluate in one step (for screening).

        Args:
            timesteps: Training timesteps.

        Returns:
            Dict with combined training + quick eval metrics.
        """
        try:
            import importlib
            import env as env_module
            importlib.reload(env_module)

            env_instance = env_module.Attitude_control_stage1(render=False)
            env_instance.record_flag = 1
            log_dir = str(self.project_root / "model" / "screen_logs")
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

            # Extract training metrics
            x, y = ts2xy(load_results(log_dir), "timesteps")
            if len(y) < 5:
                env_monitored.close()
                self._clean_logs(log_dir)
                return None

            last_20 = y[-20:] if len(y) >= 20 else y

            # Quick eval: run 10 episodes with trained model
            eval_rewards = []
            completed = 0
            for _ in range(10):
                obs, _ = env_instance.reset()
                done = False
                total_r = 0.0
                steps = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env_instance.step(action)
                    total_r += reward
                    steps += 1
                    done = terminated or truncated
                eval_rewards.append(total_r)
                if steps >= env_instance.max_step_num:
                    completed += 1

            env_monitored.close()
            self._clean_logs(log_dir)

            return {
                "mean_reward": float(np.mean(last_20)),
                "std_reward": float(np.std(last_20)),
                "total_episodes": len(y),
                "training_time_s": round(elapsed, 1),
                "timesteps": timesteps,
                "eval_mean_reward": float(np.mean(eval_rewards)),
                "completion_rate": completed / 10,
            }

        except Exception as e:
            print(f"[Eval] Quick eval error: {e}")
            traceback.print_exc()
            return None

    def _clean_logs(self, log_dir: str):
        """Clean up monitor log files."""
        for f in Path(log_dir).glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
