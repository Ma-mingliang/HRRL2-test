"""
第一阶段训练脚本（增强监控版 - 编码修复）：自适应LQR控制器 (Agent_LQR)

新增功能：
1. 每个episode详细记录姿态误差（倾斜角误差、角速度误差）
2. 实时对比RL性能与固定增益LQR基线
3. 保存完整的误差历史数据和训练报告
4. 可视化训练过程中的性能提升
5. ✨ 完全复刻原始代码的分阶段LQR控制策略
6. 🔧 修复Windows编码问题，支持emoji和中文
"""

import os
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from env import Attitude_control_stage1
from importlib import reload
import env
reload(env)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
MODEL_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "model"),
    os.path.join(WORKSPACE_ROOT, "model"),
]
MODEL_DIR = next((path for path in MODEL_DIR_CANDIDATES if os.path.isdir(path)), MODEL_DIR_CANDIDATES[0])


# ==================== 🆕 智能性能评估模块 ====================

class PerformanceEvaluator:
    """
    智能性能评估模块
    多维度评价控制系统质量
    """
    
    @staticmethod
    def evaluate_control_quality(summary, baseline_stats=None):
        """
        综合评估控制质量
        
        评估维度：
        1. 精度 (Accuracy) - 倾斜角误差越小越好 ⭐ 主要目标
        2. 稳定性 (Stability) - 误差标准差越小越好
        3. 效率 (Efficiency) - 用最少的控制代价达到最好的效果
        """
        
        results = {
            'scores': {},
            'grades': {},
            'comments': []
        }
        
        # ========== 1. 精度评估 (Accuracy) ==========
        avg_tilt_error = summary['avg_tilt_error']
        
        if avg_tilt_error < 0.01:  # < 0.57°
            accuracy_score = 100
            accuracy_grade = "A+"
            accuracy_comment = "卓越！误差极小，接近完美控制"
        elif avg_tilt_error < 0.02:  # < 1.15°
            accuracy_score = 90
            accuracy_grade = "A"
            accuracy_comment = "优秀！误差很小，控制精度高"
        elif avg_tilt_error < 0.03:  # < 1.72°
            accuracy_score = 80
            accuracy_grade = "B+"
            accuracy_comment = "良好！误差可接受，达到预期"
        elif avg_tilt_error < 0.05:  # < 2.87°
            accuracy_score = 70
            accuracy_grade = "B"
            accuracy_comment = "中等，误差偏大，建议继续优化"
        else:
            accuracy_score = 60
            accuracy_grade = "C"
            accuracy_comment = "需要改进，误差过大"
        
        results['scores']['accuracy'] = accuracy_score
        results['grades']['accuracy'] = accuracy_grade
        results['comments'].append(f"📐 精度: {accuracy_grade} - {accuracy_comment}")
        
        # ========== 2. 稳定性评估 (Stability) ==========
        std_tilt_error = summary['std_tilt_error']
        
        if std_tilt_error < 0.01:
            stability_score = 100
            stability_grade = "A+"
            stability_comment = "非常稳定，误差波动极小"
        elif std_tilt_error < 0.02:
            stability_score = 85
            stability_grade = "A"
            stability_comment = "稳定性好，误差波动小"
        elif std_tilt_error < 0.03:
            stability_score = 70
            stability_grade = "B"
            stability_comment = "稳定性一般，有一定波动"
        else:
            stability_score = 60
            stability_grade = "C"
            stability_comment = "不够稳定，波动较大"
        
        results['scores']['stability'] = stability_score
        results['grades']['stability'] = stability_grade
        results['comments'].append(f"📊 稳定性: {stability_grade} - {stability_comment}")
        
        # ========== 3. 控制效率评估 (Efficiency) ==========
        avg_angular_vel = summary['avg_angular_vel']
        
        if avg_angular_vel < 0.001:
            avg_angular_vel = 0.001
        
        efficiency_ratio = avg_tilt_error / avg_angular_vel
        
        if efficiency_ratio < 0.1:
            efficiency_score = 100
            efficiency_grade = "A+"
            efficiency_comment = "控制效率极高，动作经济"
        elif efficiency_ratio < 0.2:
            efficiency_score = 85
            efficiency_grade = "A"
            efficiency_comment = "控制效率高，动作合理"
        elif efficiency_ratio < 0.5:
            efficiency_score = 70
            efficiency_grade = "B"
            efficiency_comment = "控制效率一般"
        else:
            efficiency_score = 60
            efficiency_grade = "C"
            efficiency_comment = "控制效率偏低，动作偏多"
        
        results['scores']['efficiency'] = efficiency_score
        results['grades']['efficiency'] = efficiency_grade
        results['efficiency_ratio'] = efficiency_ratio
        results['comments'].append(f"⚡ 效率: {efficiency_grade} - {efficiency_comment}")
        
        # ========== 4. 综合评分 ==========
        overall_score = (accuracy_score * 0.5 + 
                        stability_score * 0.3 + 
                        efficiency_score * 0.2)
        
        if overall_score >= 90:
            overall_grade = "A+"
            overall_comment = "🏆 卓越表现！控制系统达到专业级水平"
        elif overall_score >= 80:
            overall_grade = "A"
            overall_comment = "⭐ 优秀表现！控制质量很高"
        elif overall_score >= 70:
            overall_grade = "B"
            overall_comment = "✅ 良好表现，达到预期目标"
        elif overall_score >= 60:
            overall_grade = "C"
            overall_comment = "⚠️  中等表现，建议继续优化"
        else:
            overall_grade = "D"
            overall_comment = "❌ 需要改进，检查训练配置"
        
        results['overall_score'] = overall_score
        results['overall_grade'] = overall_grade
        results['overall_comment'] = overall_comment
        
        # ========== 5. 与基线对比 ==========
        if baseline_stats:
            baseline_tilt = baseline_stats['mean_tilt_error']
            tilt_improvement = (1 - avg_tilt_error / baseline_tilt) * 100
            
            results['baseline_improvement'] = tilt_improvement
            
            if tilt_improvement > 20:
                results['adaptive_effect'] = "🎉 自适应LQR效果显著！大幅优于固定增益"
            elif tilt_improvement > 10:
                results['adaptive_effect'] = "✅ 自适应LQR有效，明显优于固定增益"
            elif tilt_improvement > 0:
                results['adaptive_effect'] = "☑️  自适应LQR略有帮助"
            elif tilt_improvement > -10:
                results['adaptive_effect'] = "⚠️  自适应效果不明显"
            else:
                results['adaptive_effect'] = "❌ 自适应反而变差，需要检查配置"
        
        return results
    
    @staticmethod
    def explain_reward_function():
        """解释奖励函数的设计理念"""
        explanation = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      🎯 奖励函数设计理念（第一阶段）                        ║
╚════════════════════════════════════════════════════════════════════════════╝

【主要目标】让倾斜角误差尽可能小 (Minimize Tilt Error)
  → 这是姿态控制的核心目标：保持自行车直立或跟踪目标倾斜角

【奖励函数设计】
  情况1: 当误差很小时 (|error| < 0.002 rad ≈ 0.11°)
    reward = 0.1 - |角速度|
    ↑ 意义：已经接近目标了，奖励"平稳控制"，避免过度调整
    
  情况2: 当误差较大时 (|error| ≥ 0.002 rad)
    reward = |旧误差| - |新误差|
    ↑ 意义：奖励"误差减小"，鼓励快速收敛到目标

【第一阶段的特殊性】
  • 动作空间：学习LQR增益参数 [kp, kd, k]
  • 目标：根据不同状态自适应调整增益
  • 固定增益基线：前100步 (kp=15.2, kd=2.96, k=0)
                   100步后 (kp=15.2, kd=2.96, k=12.3)
  • RL要学会：何时用大增益快速响应，何时用小增益保持稳定

【评价指标】
  ⭐⭐⭐⭐⭐ 倾斜角误差 (主要目标) - 越小越好
  ⭐⭐⭐⭐   稳定性 (误差标准差) - 越小越好
  ⭐⭐⭐     控制效率 (误差/角速度) - 越低越好

╚════════════════════════════════════════════════════════════════════════════╝
"""
        return explanation


# ==================== LQR基线控制器 ====================

class LQRBaselineController:
    """
    固定增益LQR控制器基线
    完全复刻原始代码的分阶段控制策略
    """
    
    def __init__(self):
        """初始化LQR控制器"""
        # 前100步的参数（稳定启动阶段）
        self.kp_init = 15.2491
        self.kd_init = 2.96
        self.k_init = 0
        
        # 100步后的参数（完整控制阶段）
        self.kp = 15.249
        self.kd = 2.96
        self.k = 12.3
        
        # 误差记录
        self.tilt_errors = []
        self.tilt_angle_values = []
        self.angular_velocity_errors = []
        self.velocity_values = []
        
        # Episode统计
        self.episode_summaries = []
        
        # 步数计数器
        self.current_step = 0
        
    def get_action(self, obs, target_theta=0):
        """
        计算LQR控制输出（完全复刻原代码的分阶段控制律）
        
        Args:
            obs: 归一化的观测 [倾斜角误差, 倾斜角, 角速度, 速度]
            target_theta: 目标倾斜角（弧度）
        
        Returns:
            3维动作 [kp, kd, k] 归一化到 [-1, 1]
        """
        # 🔥 分阶段控制策略
        if self.current_step < 100:
            kp = self.kp_init
            kd = self.kd_init
            k = self.k_init
        else:
            kp = self.kp
            kd = self.kd
            k = self.k
        
        # 归一化到 [-1, 1]
        kp_norm = (kp - 5) / (45 - 5) * 2 - 1
        kd_norm = (kd - 1) / (15 - 1) * 2 - 1
        k_norm = (k - 0.1) / (0.9 - 0.1) * 2 - 1
        
        self.current_step += 1
        
        return np.array([kp_norm, kd_norm, k_norm], dtype=np.float32)
    
    def record_step(self, obs):
        """记录单步状态"""
        tilt_error = abs(obs[0] * 1.57)
        tilt_angle = abs(obs[1] * 1.57)
        angular_velocity = abs(obs[2] * 10)
        velocity = obs[3] * 5
        
        self.tilt_errors.append(tilt_error)
        self.tilt_angle_values.append(tilt_angle)
        self.angular_velocity_errors.append(angular_velocity)
        self.velocity_values.append(velocity)
    
    def episode_end(self):
        """Episode结束时计算统计量"""
        if len(self.tilt_errors) > 0:
            avg_tilt_error = np.mean(self.tilt_errors)
            max_tilt_error = np.max(self.tilt_errors)
            avg_angular_vel = np.mean(self.angular_velocity_errors)
            avg_velocity = np.mean(self.velocity_values)
            steps = len(self.tilt_errors)
            
            summary = {
                'avg_tilt_error': avg_tilt_error,
                'max_tilt_error': max_tilt_error,
                'avg_angular_vel': avg_angular_vel,
                'avg_velocity': avg_velocity,
                'steps': steps
            }
            
            self.episode_summaries.append(summary)
            
            # 清空当前记录
            self.tilt_errors = []
            self.tilt_angle_values = []
            self.angular_velocity_errors = []
            self.velocity_values = []
            
            # 重置步数计数器
            self.current_step = 0
            
            return summary
        return None
    
    def get_statistics(self):
        """获取整体统计信息"""
        if len(self.episode_summaries) > 0:
            tilt_errors = [s['avg_tilt_error'] for s in self.episode_summaries]
            angular_vels = [s['avg_angular_vel'] for s in self.episode_summaries]
            
            return {
                'mean_tilt_error': np.mean(tilt_errors),
                'std_tilt_error': np.std(tilt_errors),
                'mean_angular_vel': np.mean(angular_vels),
                'std_angular_vel': np.std(angular_vels),
                'episodes': len(self.episode_summaries),
                'kp_init': self.kp_init,
                'kd_init': self.kd_init,
                'k_init': self.k_init,
                'kp': self.kp,
                'kd': self.kd,
                'k': self.k
            }
        return None
    
    def save_errors(self, save_path):
        """保存误差数据到CSV"""
        if len(self.episode_summaries) > 0:
            tilt_errors = [s['avg_tilt_error'] for s in self.episode_summaries]
            angular_vels = [s['avg_angular_vel'] for s in self.episode_summaries]
            
            np.savetxt(save_path, 
                      np.array([tilt_errors, angular_vels]).T,
                      delimiter=',',
                      header='tilt_error_rad,angular_velocity_rad_s',
                      comments='')
            print(f"\n📊 LQR基线误差已保存: {save_path}")


# ==================== 详细误差追踪器 ====================

class DetailedErrorTracker:
    """详细的误差追踪器"""
    
    def __init__(self, log_dir, baseline_stats=None):
        self.log_dir = log_dir
        self.baseline_stats = baseline_stats
        
        # 当前episode记录
        self.current_tilt_errors = []
        self.current_tilt_angles = []
        self.current_angular_velocities = []
        self.current_velocities = []
        self.current_actions = []
        
        # Episode摘要
        self.episode_summaries = []
        
        # CSV文件
        self.csv_path = os.path.join(log_dir, "rl_training_errors.csv")
        
        # 初始化CSV
        with open(self.csv_path, 'w', encoding='utf-8') as f:
            f.write("episode,steps,avg_tilt_error,std_tilt_error,max_tilt_error,"
                   "avg_angular_vel,std_angular_vel,avg_velocity,avg_kp,avg_kd,avg_k,reward\n")
    
    def record_step(self, obs, action, reward):
        """记录单步数据"""
        # 反归一化状态
        tilt_error = abs(obs[0] * 1.57)
        tilt_angle = abs(obs[1] * 1.57)
        angular_velocity = abs(obs[2] * 10)
        velocity = obs[3] * 5
        
        # 反归一化动作
        kp = (action[0] + 1) / 2 * (45 - 5) + 5
        kd = (action[1] + 1) / 2 * (15 - 1) + 1
        k = (action[2] + 1) / 2 * (0.9 - 0.1) + 0.1
        
        self.current_tilt_errors.append(tilt_error)
        self.current_tilt_angles.append(tilt_angle)
        self.current_angular_velocities.append(angular_velocity)
        self.current_velocities.append(velocity)
        self.current_actions.append([kp, kd, k])
    
    def episode_end(self, episode_num, episode_reward):
        """Episode结束处理"""
        if len(self.current_tilt_errors) == 0:
            return
        
        # 计算统计量
        avg_tilt_error = np.mean(self.current_tilt_errors)
        std_tilt_error = np.std(self.current_tilt_errors)
        max_tilt_error = np.max(self.current_tilt_errors)
        
        avg_angular_vel = np.mean(self.current_angular_velocities)
        std_angular_vel = np.std(self.current_angular_velocities)
        
        avg_velocity = np.mean(self.current_velocities)
        
        actions_array = np.array(self.current_actions)
        avg_kp = np.mean(actions_array[:, 0])
        avg_kd = np.mean(actions_array[:, 1])
        avg_k = np.mean(actions_array[:, 2])
        
        steps = len(self.current_tilt_errors)
        
        # 保存摘要
        summary = {
            'episode': episode_num,
            'steps': steps,
            'avg_tilt_error': avg_tilt_error,
            'std_tilt_error': std_tilt_error,
            'max_tilt_error': max_tilt_error,
            'avg_angular_vel': avg_angular_vel,
            'std_angular_vel': std_angular_vel,
            'avg_velocity': avg_velocity,
            'avg_kp': avg_kp,
            'avg_kd': avg_kd,
            'avg_k': avg_k,
            'reward': episode_reward
        }
        self.episode_summaries.append(summary)
        
        # 写入CSV
        with open(self.csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{episode_num},{steps},{avg_tilt_error:.6f},{std_tilt_error:.6f},{max_tilt_error:.6f},"
                   f"{avg_angular_vel:.6f},{std_angular_vel:.6f},{avg_velocity:.6f},"
                   f"{avg_kp:.2f},{avg_kd:.2f},{avg_k:.3f},{episode_reward:.2f}\n")
        
        # 打印详细信息
        self._print_episode_summary(summary)
        
        # 清空当前记录
        self.current_tilt_errors = []
        self.current_tilt_angles = []
        self.current_angular_velocities = []
        self.current_velocities = []
        self.current_actions = []
    
    def _print_episode_summary(self, summary):
        """打印Episode摘要（增强版：包含性能评估）"""
        episode = summary['episode']
        
        # 基础信息
        print(f"\n{'='*80}")
        print(f"📊 Episode {episode} 完成 - 第一阶段 (Adaptive LQR)")
        print(f"{'='*80}")
        print(f"🎯 基本信息:")
        print(f"   Steps: {summary['steps']:4d} | Reward: {summary['reward']:7.2f} | Velocity: {summary['avg_velocity']:.2f} m/s")
        
        # 误差详情
        print(f"\n📐 控制精度 (Control Accuracy):")
        print(f"   倾斜角误差: {summary['avg_tilt_error']:.4f} rad ({np.rad2deg(summary['avg_tilt_error']):.2f}°)")
        print(f"   误差标准差: {summary['std_tilt_error']:.4f} rad (稳定性指标)")
        print(f"   最大误差: {summary['max_tilt_error']:.4f} rad ({np.rad2deg(summary['max_tilt_error']):.2f}°)")
        
        print(f"\n🔄 控制活跃度 (Control Activity):")
        print(f"   平均角速度: {summary['avg_angular_vel']:.4f} rad/s")
        print(f"   角速度标准差: {summary['std_angular_vel']:.4f} rad/s")
        
        print(f"\n🎛️  LQR增益参数 (Learned Gains):")
        print(f"   kp: {summary['avg_kp']:5.2f} | kd: {summary['avg_kd']:5.2f} | k: {summary['avg_k']:.3f}")
        
        # 🆕 智能性能评估
        print(f"\n{'─'*80}")
        print(f"🎓 智能性能评估")
        print(f"{'─'*80}")
        
        eval_results = PerformanceEvaluator.evaluate_control_quality(
            summary, self.baseline_stats
        )
        
        # 显示各维度评分
        for comment in eval_results['comments']:
            print(f"   {comment}")
        
        # 控制效率详解
        print(f"\n   ⚖️  控制效率比 (误差/角速度): {eval_results['efficiency_ratio']:.3f}")
        print(f"       → 越低越好：用更少的调整达到更小的误差")
        
        # 综合评分
        print(f"\n   🏆 综合评分: {eval_results['overall_score']:.1f}/100 ({eval_results['overall_grade']})")
        print(f"       {eval_results['overall_comment']}")
        
        # 与固定LQR基线对比
        if self.baseline_stats:
            print(f"\n{'─'*80}")
            print(f"🔬 与固定LQR基线对比")
            print(f"{'─'*80}")
            
            tilt_diff = summary['avg_tilt_error'] - self.baseline_stats['mean_tilt_error']
            tilt_pct = (tilt_diff / self.baseline_stats['mean_tilt_error']) * 100
            
            if tilt_diff < 0:
                print(f"   ✅ 倾斜角误差: 优于基线 {abs(tilt_diff):.4f}rad ({abs(tilt_pct):.1f}% 更好)")
            else:
                print(f"   ⚠️  倾斜角误差: 差于基线 {tilt_diff:.4f}rad ({tilt_pct:.1f}% 更差)")
            
            print(f"   📊 基线参考: Tilt={self.baseline_stats['mean_tilt_error']:.4f}rad, "
                  f"AngVel={self.baseline_stats['mean_angular_vel']:.4f}rad/s")
            print(f"   📊 基线参数: 前100步 kp={self.baseline_stats['kp_init']:.1f}, "
                  f"100步后 kp={self.baseline_stats['kp']:.1f}")
            
            # 自适应效果
            if 'adaptive_effect' in eval_results:
                print(f"\n   {eval_results['adaptive_effect']}")
        
        print(f"{'='*80}\n")
    
    def get_recent_stats(self, last_n=20):
        """获取最近N个episode的统计"""
        if len(self.episode_summaries) < last_n:
            last_n = len(self.episode_summaries)
        
        if last_n == 0:
            return None
        
        recent = self.episode_summaries[-last_n:]
        
        return {
            'avg_tilt_error': np.mean([s['avg_tilt_error'] for s in recent]),
            'avg_angular_vel': np.mean([s['avg_angular_vel'] for s in recent]),
            'avg_reward': np.mean([s['reward'] for s in recent]),
            'episodes': last_n
        }
    
    def save_final_report(self):
        """保存最终报告"""
        if len(self.episode_summaries) == 0:
            return
        
        report_path = os.path.join(self.log_dir, "rl_training_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("强化学习训练完整报告 - 第一阶段：自适应LQR控制器\n")
            f.write("="*80 + "\n\n")
            
            # 整体统计
            all_tilt = [s['avg_tilt_error'] for s in self.episode_summaries]
            all_angular = [s['avg_angular_vel'] for s in self.episode_summaries]
            all_rewards = [s['reward'] for s in self.episode_summaries]
            
            f.write("📊 整体统计:\n")
            f.write(f"  总Episodes: {len(self.episode_summaries)}\n")
            f.write(f"  平均倾斜角误差: {np.mean(all_tilt):.6f} ± {np.std(all_tilt):.6f} rad "
                   f"({np.rad2deg(np.mean(all_tilt)):.2f}°)\n")
            f.write(f"  平均角速度: {np.mean(all_angular):.6f} ± {np.std(all_angular):.6f} rad/s\n")
            f.write(f"  平均奖励: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n\n")
            
            # 最近20集
            recent_stats = self.get_recent_stats(20)
            if recent_stats:
                f.write("📈 最近20集统计:\n")
                f.write(f"  平均倾斜角误差: {recent_stats['avg_tilt_error']:.6f} rad "
                       f"({np.rad2deg(recent_stats['avg_tilt_error']):.2f}°)\n")
                f.write(f"  平均角速度: {recent_stats['avg_angular_vel']:.6f} rad/s\n")
                f.write(f"  平均奖励: {recent_stats['avg_reward']:.2f}\n\n")
            
            # 与基线对比
            if self.baseline_stats:
                f.write("🔬 与LQR基线对比:\n")
                f.write(f"  基线倾斜角误差: {self.baseline_stats['mean_tilt_error']:.6f} rad "
                       f"({np.rad2deg(self.baseline_stats['mean_tilt_error']):.2f}°)\n")
                f.write(f"  基线角速度: {self.baseline_stats['mean_angular_vel']:.6f} rad/s\n")
                f.write(f"  基线参数 (前100步): kp={self.baseline_stats['kp_init']:.4f}, "
                       f"kd={self.baseline_stats['kd_init']:.2f}, k={self.baseline_stats['k_init']:.1f}\n")
                f.write(f"  基线参数 (100步后): kp={self.baseline_stats['kp']:.3f}, "
                       f"kd={self.baseline_stats['kd']:.2f}, k={self.baseline_stats['k']:.1f}\n\n")
                
                if recent_stats:
                    tilt_improvement = (1 - recent_stats['avg_tilt_error']/self.baseline_stats['mean_tilt_error']) * 100
                    angular_improvement = (1 - recent_stats['avg_angular_vel']/self.baseline_stats['mean_angular_vel']) * 100
                    
                    f.write("  性能提升:\n")
                    f.write(f"    倾斜角误差: {tilt_improvement:+.2f}%\n")
                    f.write(f"    角速度: {angular_improvement:+.2f}%\n\n")
            
            # 最佳Episode
            best_tilt_idx = np.argmin(all_tilt)
            best_episode = self.episode_summaries[best_tilt_idx]
            
            f.write("⭐ 最佳Episode (最小倾斜角误差):\n")
            f.write(f"  Episode: {best_episode['episode']}\n")
            f.write(f"  倾斜角误差: {best_episode['avg_tilt_error']:.6f} rad "
                   f"({np.rad2deg(best_episode['avg_tilt_error']):.2f}°)\n")
            f.write(f"  角速度: {best_episode['avg_angular_vel']:.6f} rad/s\n")
            f.write(f"  奖励: {best_episode['reward']:.2f}\n")
            f.write(f"  增益: kp={best_episode['avg_kp']:.2f}, "
                   f"kd={best_episode['avg_kd']:.2f}, k={best_episode['avg_k']:.3f}\n\n")
            
            # 🆕 最佳Episode与基线对比
            if self.baseline_stats:
                best_tilt_improvement = (1 - best_episode['avg_tilt_error']/self.baseline_stats['mean_tilt_error']) * 100
                best_angular_improvement = (1 - best_episode['avg_angular_vel']/self.baseline_stats['mean_angular_vel']) * 100
                
                f.write("🏆 最佳Episode相较基线的提升:\n")
                f.write(f"  倾斜角误差提升: {best_tilt_improvement:+.2f}%\n")
                f.write(f"  角速度提升: {best_angular_improvement:+.2f}%\n")
                
                if best_tilt_improvement > 30:
                    f.write(f"  评价: 🎉🎉🎉 最佳性能极其优秀！显著超越固定增益基线！\n")
                elif best_tilt_improvement > 20:
                    f.write(f"  评价: 🎉🎉 最佳性能非常优秀！大幅超越固定增益基线！\n")
                elif best_tilt_improvement > 10:
                    f.write(f"  评价: ✅✅ 最佳性能优秀！明显超越固定增益基线！\n")
                elif best_tilt_improvement > 0:
                    f.write(f"  评价: ✅ 最佳性能良好，略优于固定增益基线\n")
                else:
                    f.write(f"  评价: ⚠️  最佳性能未能超越基线，建议检查训练配置\n")
        
        print(f"\n📄 完整报告已保存: {report_path}")


# ==================== 增强版回调函数 ====================

class EnhancedTrainingCallback(BaseCallback):
    """增强版回调函数"""
    
    def __init__(self, check_freq: int, log_dir: str, error_tracker: DetailedErrorTracker, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.error_tracker = error_tracker
        self.save_path = os.path.join(log_dir, "stage1_agent_lqr_best")
        self.best_mean_reward = -np.inf
        
        # Episode追踪
        self.current_episode = 0
        self.episode_reward = 0
        self.episode_step = 0
    
    def _on_step(self) -> bool:
        # 记录当前步
        obs = self.locals['new_obs'][0] if len(self.locals['new_obs'].shape) > 1 else self.locals['new_obs']
        action = self.locals['actions'][0] if len(self.locals['actions'].shape) > 1 else self.locals['actions']
        reward = self.locals['rewards'][0] if hasattr(self.locals['rewards'], '__len__') else self.locals['rewards']
        
        self.error_tracker.record_step(obs, action, reward)
        self.episode_reward += reward
        self.episode_step += 1
        
        # 检查episode结束
        done = self.locals['dones'][0] if hasattr(self.locals['dones'], '__len__') else self.locals['dones']
        
        if done:
            self.current_episode += 1
            self.error_tracker.episode_end(self.current_episode, self.episode_reward)
            self.episode_reward = 0
            self.episode_step = 0
        
        # 定期保存最佳模型
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-20:])
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose >= 1:
                        print(f"\n{'='*60}")
                        print(f"🎉 新的最佳模型！平均奖励: {mean_reward:.2f}")
                        print(f"💾 保存到: {self.save_path}")
                        
                        recent_stats = self.error_tracker.get_recent_stats(20)
                        if recent_stats:
                            print(f"\n当前性能 (最近20集):")
                            print(f"  倾斜角误差: {recent_stats['avg_tilt_error']:.4f} rad "
                                 f"({np.rad2deg(recent_stats['avg_tilt_error']):.2f}°)")
                            print(f"  角速度: {recent_stats['avg_angular_vel']:.4f} rad/s")
                        
                        print(f"{'='*60}\n")
                    self.model.save(self.save_path)
        
        return True


# ==================== LQR基线评估 ====================

def evaluate_lqr_baseline(env, controller, num_episodes=50):
    """评估固定增益LQR控制器（分阶段控制）"""
    print("\n" + "="*80)
    print("📊 LQR基线控制器评估（分阶段控制策略）")
    print("="*80)
    print(f"评估Episodes: {num_episodes}")
    print(f"\n控制策略:")
    print(f"  前100步 - 稳定启动阶段:")
    print(f"    kp={controller.kp_init}, kd={controller.kd_init}, k={controller.k_init}")
    print(f"    目标: 保持直立 (target_theta=0)")
    print(f"\n  100步后 - 完整控制阶段:")
    print(f"    kp={controller.kp}, kd={controller.kd}, k={controller.k}")
    print(f"    目标: 跟踪动态倾斜角")
    print("="*80 + "\n")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        episode_reward = 0
        
        # 重置控制器的步数计数
        controller.current_step = 0
        
        # 模拟目标倾斜角的动态变化
        target_theta = 0
        
        while not (done or truncated):
            # 每100步改变目标倾斜角
            if (step_count % 100 == 0) and (step_count >= 100):
                sigma = min(0.3, (1.08**(episode)) * 0.01)
                target_theta = np.random.normal(loc=0.0, scale=sigma, size=None)
                target_theta = np.clip(target_theta, -np.pi/12, np.pi/12)
            
            # 获取LQR动作
            action = controller.get_action(obs, target_theta)
            
            # 执行动作
            obs, reward, done, truncated, _ = env.step(action)
            
            # 记录误差
            controller.record_step(obs)
            
            episode_reward += reward
            step_count += 1
        
        summary = controller.episode_end()
        
        if summary:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Steps: {step_count:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Tilt: {summary['avg_tilt_error']:.4f}rad ({np.rad2deg(summary['avg_tilt_error']):.2f}°) | "
                  f"AngVel: {summary['avg_angular_vel']:.4f}rad/s")
    
    stats = controller.get_statistics()
    
    print("\n" + "="*80)
    print("📈 LQR基线统计结果")
    print("="*80)
    if stats:
        print(f"平均倾斜角误差: {stats['mean_tilt_error']:.4f} ± {stats['std_tilt_error']:.4f} rad "
              f"({np.rad2deg(stats['mean_tilt_error']):.2f}°)")
        print(f"平均角速度: {stats['mean_angular_vel']:.4f} ± {stats['std_angular_vel']:.4f} rad/s")
        print(f"评估Episodes: {stats['episodes']}")
        print(f"\n控制参数:")
        print(f"  前100步: kp={stats['kp_init']:.4f}, kd={stats['kd_init']:.2f}, k={stats['k_init']:.1f}")
        print(f"  100步后: kp={stats['kp']:.3f}, kd={stats['kd']:.2f}, k={stats['k']:.1f}")
    print("="*80 + "\n")
    
    return stats


# ==================== 主训练流程 ====================

def main():
    """主训练函数"""
    
    print("\n" + "="*80)
    print("第一阶段（增强监控版）：自适应LQR控制器")
    print("="*80)
    print("✅ 每个Episode详细记录姿态误差")
    print("✅ 实时对比RL性能与固定LQR基线")
    print("✅ 保存完整的误差历史和增益参数")
    print("✅ 完全复刻原代码的分阶段LQR控制策略")
    print("✅ 🆕 智能性能评估 - 多维度评价控制质量")
    print("✅ 🆕 奖励函数透明化 - 清晰展示优化目标")
    print("="*80)
    
    # 🆕 显示奖励函数说明
    print(PerformanceEvaluator.explain_reward_function())
    input("按Enter键继续...")
    
    # 检查CUDA
    print(f"\n🔧 设备检测:")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建环境
    print(f"\n[1/6] 创建训练环境...")
    env = Attitude_control_stage1(render=1)
    env.record_flag = 1
    
    # 日志目录
    log_dir = os.path.join(MODEL_DIR, "stage1_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"   日志目录: {log_dir}")
    
    env = Monitor(env, log_dir)
    
    # LQR基线评估
    print(f"\n[2/6] 运行LQR基线评估（分阶段控制）...")
    
    lqr_controller = LQRBaselineController()
    baseline_stats = evaluate_lqr_baseline(env.env, lqr_controller, num_episodes=50)
    
    # 保存基线数据
    baseline_error_path = os.path.join(log_dir, "lqr_baseline_errors.csv")
    lqr_controller.save_errors(baseline_error_path)
    
    if baseline_stats:
        baseline_stats_path = os.path.join(log_dir, "lqr_baseline_stats.txt")
        with open(baseline_stats_path, 'w', encoding='utf-8') as f:
            f.write("LQR基线控制器统计信息（分阶段控制）\n")
            f.write("="*50 + "\n")
            f.write(f"平均倾斜角误差: {baseline_stats['mean_tilt_error']:.6f} ± "
                   f"{baseline_stats['std_tilt_error']:.6f} rad\n")
            f.write(f"平均角速度: {baseline_stats['mean_angular_vel']:.6f} ± "
                   f"{baseline_stats['std_angular_vel']:.6f} rad/s\n")
            f.write(f"评估Episodes: {baseline_stats['episodes']}\n\n")
            f.write(f"控制策略:\n")
            f.write(f"  前100步（稳定启动）:\n")
            f.write(f"    kp = {baseline_stats['kp_init']}\n")
            f.write(f"    kd = {baseline_stats['kd_init']}\n")
            f.write(f"    k = {baseline_stats['k_init']}\n")
            f.write(f"    target_theta = 0 (强制保持直立)\n\n")
            f.write(f"  100步后（完整控制）:\n")
            f.write(f"    kp = {baseline_stats['kp']}\n")
            f.write(f"    kd = {baseline_stats['kd']}\n")
            f.write(f"    k = {baseline_stats['k']}\n")
            f.write(f"    target_theta = 动态变化\n")
        print(f"📄 统计信息已保存: {baseline_stats_path}")
    
    # 创建误差追踪器
    print(f"\n[3/6] 初始化误差追踪器...")
    error_tracker = DetailedErrorTracker(log_dir, baseline_stats)
    print(f"   ✅ 将记录每个Episode的详细误差数据")
    print(f"   ✅ 将提供智能性能评估报告")
    
    # 创建回调
    print(f"\n[4/6] 配置增强版回调...")
    callback = EnhancedTrainingCallback(
        check_freq=env.env.max_step_num,
        log_dir=log_dir,
        error_tracker=error_tracker
    )
    
    # 动作噪声
    print(f"\n[5/6] 配置TD3算法...")
    n_actions = env.action_space.shape[-1]
    print(f"   动作空间维度: {n_actions}")
    print(f"   噪声σ: 0.1")
    
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TD3(
        "MlpPolicy",
        env=env,
        device=device,
        gamma=0.99,
        learning_rate=0.001,
        batch_size=128,
        buffer_size=1000000,
        action_noise=action_noise,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        learning_starts=128,
        train_freq=(1, "step"),
        gradient_steps=-1,
        policy_delay=2,
        seed=42,
        verbose=1
    )
    
    # 配置摘要
    print(f"\n[6/6] 训练配置摘要")
    print(f"="*80)
    print(f"设备: {device.upper()}")
    print(f"策略网络: MLP")
    print(f"\n超参数:")
    print(f"  • 学习率: 0.001")
    print(f"  • 批次大小: 128")
    print(f"  • 缓冲区: 1,000,000")
    print(f"  • 总步数: 120,000")
    print(f"  • 预计episodes: ~100")
    
    if baseline_stats:
        print(f"\nLQR基线性能（分阶段控制）:")
        print(f"  • 平均倾斜角误差: {baseline_stats['mean_tilt_error']:.4f} rad "
              f"({np.rad2deg(baseline_stats['mean_tilt_error']):.2f}°)")
        print(f"  • 平均角速度: {baseline_stats['mean_angular_vel']:.4f} rad/s")
        print(f"  • 前100步参数: kp={baseline_stats['kp_init']:.1f}, kd={baseline_stats['kd_init']:.1f}, k={baseline_stats['k_init']:.1f}")
        print(f"  • 100步后参数: kp={baseline_stats['kp']:.1f}, kd={baseline_stats['kd']:.1f}, k={baseline_stats['k']:.1f}")
        print(f"  ⚠️  训练目标: 学习自适应增益，超越固定分阶段基线")
    
    print("="*80)
    
    # 开始训练
    print(f"\n开始训练...")
    print(f"\n💡 训练提示:")
    print(f"   • 每个Episode结束后会显示详细的误差统计和智能评估")
    print(f"   • 性能评估包括：精度、稳定性、控制效率三个维度")
    print(f"   • 所有数据会保存到CSV文件供后续分析")
    print(f"   • 前30-50个episode可能表现较差，请耐心等待")
    print(f"   • RL将学习如何动态调整kp、kd、k参数")
    print("\n" + "="*80 + "\n")
    
    total_timesteps = 1000 * 000
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        print("\n✅ 训练成功完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断 (Ctrl+C)")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # 保存最终模型
        final_path = os.path.join(log_dir, "stage1_agent_lqr_final")
        model.save(final_path)
        print(f"\n💾 最终模型: {final_path}")
        
        # 保存完整报告
        error_tracker.save_final_report()
        
        # 统计
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(y) > 0:
            final_reward = np.mean(y[-20:])
            best_reward = np.max([np.mean(y[i:i+20]) for i in range(max(0, len(y)-20))])
            
            print(f"\n📊 训练统计:")
            print(f"   Episodes: {len(y)}")
            print(f"   最终20集平均: {final_reward:.2f}")
            print(f"   历史最佳20集: {best_reward:.2f}")
            
            # 性能评估
            recent_stats = error_tracker.get_recent_stats(20)
            if recent_stats:
                print(f"\n最近20集性能:")
                print(f"   倾斜角误差: {recent_stats['avg_tilt_error']:.4f} rad "
                      f"({np.rad2deg(recent_stats['avg_tilt_error']):.2f}°)")
                print(f"   角速度: {recent_stats['avg_angular_vel']:.4f} rad/s")
                
                if baseline_stats:
                    tilt_improvement = (1 - recent_stats['avg_tilt_error']/baseline_stats['mean_tilt_error']) * 100
                    
                    print(f"\n与LQR基线对比:")
                    print(f"   倾斜角误差提升: {tilt_improvement:+.2f}%")
                    
                    if tilt_improvement > 20:
                        print(f"\n   🎉🎉🎉 自适应LQR大获成功！性能大幅提升！")
                    elif tilt_improvement > 10:
                        print(f"\n   ✅✅ 自适应LQR非常成功！显著优于固定增益！")
                    elif tilt_improvement > 0:
                        print(f"\n   ✅ 自适应LQR有效，性能略有提升")
                    else:
                        print(f"\n   ⚠️  自适应效果不明显，可能需要调整超参数")
            
            if final_reward > 8000:
                print(f"\n⭐⭐⭐⭐⭐ 优秀！姿态控制性能非常好")
            elif final_reward > 6000:
                print(f"\n⭐⭐⭐⭐ 良好！性能可以接受")
            elif final_reward > 4000:
                print(f"\n⭐⭐⭐ 中等，建议继续训练或调整参数")
            else:
                print(f"\n⭐⭐ 需要改进，检查奖励函数和环境设置")
        
        print(f"\n" + "="*80)
        print("📁 生成的文件:")
        print(f"   • {os.path.join(log_dir, 'rl_training_errors.csv')} - Episode摘要")
        print(f"   • {os.path.join(log_dir, 'rl_training_report.txt')} - 完整报告")
        print(f"   • {os.path.join(log_dir, 'lqr_baseline_errors.csv')} - 基线数据")
        print(f"   • {final_path}.zip - 最终模型")
        print("="*80 + "\n")
        
        print("✅ 第一阶段训练完成！")
        print("📁 下一步：使用此模型训练第二阶段（内环残差控制器）")
        print("="*80 + "\n")
        
        env.close()


if __name__ == "__main__":
    main()
