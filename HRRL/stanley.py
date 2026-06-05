"""
第三阶段训练脚本（完整增强版+倾斜角误差追踪+姿态环模式选择） - 自适应Stanley控制器
修改保存逻辑后
🆕🆕🆕 从第四阶段移植的新特性（只增加，不删除原有功能）：
  ✅ 姿态环模式选择 - 5种内环控制器可选（新增配置项ATTITUDE_CONTROL_MODE）
  ✅ 静态单阶段LQR控制器（新增StaticLQRController类）
  ✅ 静态分阶段LQR控制器（新增StagedStaticLQRController类）
  ✅ 保留原有USE_STAGE2_RESIDUAL开关（向后兼容）
  ✅ 新增LQR参数配置区域

原有特性（完整保留）：
  ✅ 每个episode详细记录横向误差和航向误差
  ✅ 实时对比RL性能与Stanley基线
  ✅ 保存完整的误差历史数据
  ✅ UTF-8编码 - 支持emoji字符
  ✅ 最优模型追踪 - 记录最佳性能
  ✅ 修复episode_end和best_reward计算bug
  ✅ 路径进度追踪 - 知道走了多远
  ✅ 详细终止原因分类 - 6种终止类型
  ✅ 智能性能评估系统 - A+/A/B/C/D分级
  ✅ 完成率统计 - 激励信息
  ✅ 增强错误处理 - 防止训练崩溃
  ✅ 倾斜角误差追踪 - 详细记录每个回合的倾斜角误差
"""

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import env
from env import Path_tracking_stage3
import gymnasium as gym
import math
from collections import deque  
from importlib import reload
reload(env)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
MODEL_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "model"),
    os.path.join(WORKSPACE_ROOT, "model"),
]
MODEL_DIR = next((path for path in MODEL_DIR_CANDIDATES if os.path.isdir(path)), MODEL_DIR_CANDIDATES[0])


def _model_path(*parts):
    return os.path.join(MODEL_DIR, *parts)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║       🆕🆕🆕 新增：姿态环模式选择配置区（从第四阶段移植）                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ========== 🆕🆕🆕 姿态环（内环）控制器模式选择（新增配置项）==========
ATTITUDE_CONTROL_MODE = 'dynamic_lqr'  # 🎯🆕 新增配置项！

"""
🔬 姿态环（内环）控制器模式选择（从第四阶段移植）：

可选模式：
  'dynamic_lqr'              = 动态LQR（仅第一阶段）

"""

# ========== 🆕🆕🆕 姿态环LQR参数配置（新增配置区）==========
# 静态单阶段LQR参数（ATTITUDE_CONTROL_MODE='static_lqr'时使用）
STATIC_LQR_PARAMS = {
    'kp': 15.2491,
    'kd': 2.96,
    'k': 12.3,
    'description': '静态单阶段LQR参数'
}

# 静态分阶段LQR参数（ATTITUDE_CONTROL_MODE='staged_static_lqr'或'staged_static_lqr_residual'时使用）
STAGED_LQR_PARAMS = {
    'stage1_warmup': {  # 前100步：稳定启动
        'kp': 15.2491,
        'kd': 2.96,
        'k': 0.0,
        'description': '稳定启动阶段（0-100步）'
    },
    'stage2_full': {  # 100步后：完整控制
        'kp': 15.249,
        'kd': 2.96,
        'k': 12.3,
        'description': '完整控制阶段（100步+）'
    },
    'warmup_steps': 100  # 稳定启动步数
}

# LQR参数映射范围（用于归一化）
LQR_PARAM_RANGES = {
    'kp_range': (5, 25),
    'kd_range': (1, 5),
    'k_range': (0, 20)
}


# ==================== 运动学辅助函数 ====================

def steady_state_calculation(x, v):
    """计算稳态倾斜角"""
    l = 1.489
    l1 = 0.7
    g = 9.8
    
    theta = (math.tan(x) * (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)) + 0.4407)) / \
            (((l / v)**2) * g)
    theta = math.atan(theta)
    
    return theta


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          🆕🆕🆕 新增：静态LQR控制器类（从第四阶段移植）                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class StaticLQRController:
    """
    🆕 静态单阶段LQR控制器（从第四阶段移植）
    全程使用固定的kp、kd、k参数
    """
    
    def __init__(self, kp, kd, k, kp_range, kd_range, k_range):
        self.kp = kp
        self.kd = kd
        self.k = k
        
        # 归一化到[-1, 1]
        self.kp_normalized = self._normalize_param(kp, kp_range)
        self.kd_normalized = self._normalize_param(kd, kd_range)
        self.k_normalized = self._normalize_param(k, k_range)
        
        print(f"\n📐 静态单阶段LQR控制器已初始化")
        print(f"   原始参数: kp={kp:.4f}, kd={kd:.3f}, k={k:.3f}")
        print(f"   归一化值: [{self.kp_normalized:.3f}, {self.kd_normalized:.3f}, {self.k_normalized:.3f}]")
    
    def _normalize_param(self, value, param_range):
        """将参数值归一化到[-1, 1]"""
        normalized = (value - param_range[0]) / (param_range[1] - param_range[0]) * 2 - 1
        return np.clip(normalized, -1.0, 1.0)
    
    def predict(self, observation, deterministic=True, current_step=None):
        """返回固定的LQR参数"""
        action = np.array([self.kp_normalized, self.kd_normalized, self.k_normalized])
        return action, None


class StagedStaticLQRController:
    """
    🆕 静态分阶段LQR控制器（从第四阶段移植）
    前100步：稳定启动（k=0）
    100步后：完整控制（k=12.3）
    """
    
    def __init__(self, 
                 warmup_kp, warmup_kd, warmup_k,
                 full_kp, full_kd, full_k,
                 warmup_steps, kp_range, kd_range, k_range):
        self.warmup_steps = warmup_steps
        
        # 稳定启动阶段参数
        self.warmup_kp = warmup_kp
        self.warmup_kd = warmup_kd
        self.warmup_k = warmup_k
        
        # 完整控制阶段参数
        self.full_kp = full_kp
        self.full_kd = full_kd
        self.full_k = full_k
        
        # 归一化稳定启动参数
        self.warmup_kp_norm = self._normalize_param(warmup_kp, kp_range)
        self.warmup_kd_norm = self._normalize_param(warmup_kd, kd_range)
        self.warmup_k_norm = self._normalize_param(warmup_k, k_range)
        
        # 归一化完整控制参数
        self.full_kp_norm = self._normalize_param(full_kp, kp_range)
        self.full_kd_norm = self._normalize_param(full_kd, kd_range)
        self.full_k_norm = self._normalize_param(full_k, k_range)
        
        print(f"\n📐 静态分阶段LQR控制器已初始化")
        print(f"   阶段1 (0-{warmup_steps}步):")
        print(f"     原始: kp={warmup_kp:.4f}, kd={warmup_kd:.3f}, k={warmup_k:.3f}")
        print(f"     归一化: [{self.warmup_kp_norm:.3f}, {self.warmup_kd_norm:.3f}, {self.warmup_k_norm:.3f}]")
        print(f"   阶段2 ({warmup_steps}步+):")
        print(f"     原始: kp={full_kp:.4f}, kd={full_kd:.3f}, k={full_k:.3f}")
        print(f"     归一化: [{self.full_kp_norm:.3f}, {self.full_kd_norm:.3f}, {self.full_k_norm:.3f}]")
    
    def _normalize_param(self, value, param_range):
        """将参数值归一化到[-1, 1]"""
        normalized = (value - param_range[0]) / (param_range[1] - param_range[0]) * 2 - 1
        return np.clip(normalized, -1.0, 1.0)
    
    def predict(self, observation, deterministic=True, current_step=None):
        """根据当前步数返回对应阶段的LQR参数"""
        if current_step is not None and current_step < self.warmup_steps:
            # 稳定启动阶段
            action = np.array([self.warmup_kp_norm, self.warmup_kd_norm, self.warmup_k_norm])
        else:
            # 完整控制阶段
            action = np.array([self.full_kp_norm, self.full_kd_norm, self.full_k_norm])
        
        return action, None


# ==================== 特征提取器（保持原样）====================

class LightweightAttentionExtractor(BaseFeaturesExtractor):
    """轻量级注意力提取器"""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128,
                 d_model: int = 64, nhead: int = 2, dropout: float = 0.1):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.input_proj = nn.Linear(n_input, d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(observations)
        x = x.unsqueeze(1)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        return self.output_proj(x)


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """标准Transformer特征提取器"""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.input_embedding = nn.Linear(n_input, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(observations)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.output_layer(x)


class MultiHeadAttentionFeaturesExtractor(BaseFeaturesExtractor):
    """进阶多头注意力提取器"""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 3, 
                 feature_groups: int = 3, dropout: float = 0.1):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.feature_groups = feature_groups
        group_size = max(1, n_input // feature_groups)
        group_dim = d_model // feature_groups
        
        self.group_projections = nn.ModuleList([
            nn.Linear(group_size if i < feature_groups - 1 else n_input - i * group_size, 
                     group_dim)
            for i in range(feature_groups)
        ])
        
        concat_dim = group_dim * feature_groups
        if concat_dim != d_model:
            self.align_layer = nn.Linear(concat_dim, d_model)
        else:
            self.align_layer = nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        n_input = observations.shape[1]
        group_size = n_input // self.feature_groups
        
        group_features = []
        for i, proj in enumerate(self.group_projections):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < self.feature_groups - 1 else n_input
            group_feat = proj(observations[:, start_idx:end_idx])
            group_features.append(group_feat)
        
        x = torch.cat(group_features, dim=-1)
        x = self.align_layer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        
        return self.output_layer(x)


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """LSTM特征提取器"""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128,
                 lstm_hidden_size: int = 128, num_lstm_layers: int = 2):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU()
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self._hidden = None
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        observations = observations.unsqueeze(1)
        
        if self._hidden is None or self._hidden[0].shape[1] != batch_size:
            self._hidden = (
                torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size,
                           device=observations.device),
                torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size,
                           device=observations.device)
            )
        
        lstm_out, self._hidden = self.lstm(observations, self._hidden)
        lstm_out = lstm_out[:, -1, :]
        
        return self.linear(lstm_out)
    
    def reset_hidden(self):
        self._hidden = None


# ==================== Stanley基线控制器 ====================

class StanleyBaselineController:
    """纯Stanley控制器基线"""
    
    def __init__(self, k_lateral=0.6, k_course=0.4):
        self.k_lateral = k_lateral
        self.k_course = k_course
        
        self.lateral_errors = []
        self.course_errors = []
        self.episode_errors = []
        
    def compute_target_attitude(self, lateral_error, course_error_angle, velocity):
        if abs(velocity) < 0.1:
            velocity = 0.1
        
        x = math.atan(self.k_lateral * lateral_error / velocity) + \
            course_error_angle * self.k_course
        
        tar_attitude = steady_state_calculation(x, velocity)
        tar_attitude = np.clip(tar_attitude, -math.pi/6, math.pi/6)
        
        return tar_attitude
    
    def record_step(self, lateral_error, course_error):
        self.lateral_errors.append(abs(lateral_error))
        self.course_errors.append(abs(course_error))
    
    def episode_end(self):
        if len(self.lateral_errors) > 0:
            avg_lateral = np.mean(self.lateral_errors)
            avg_course = np.mean(self.course_errors)
            self.episode_errors.append({
                'lateral': avg_lateral,
                'course': avg_course,
                'steps': len(self.lateral_errors)
            })
            
            self.lateral_errors = []
            self.course_errors = []
            
            return avg_lateral, avg_course
        return 0, 0
    
    def get_statistics(self):
        if len(self.episode_errors) > 0:
            laterals = [e['lateral'] for e in self.episode_errors]
            courses = [e['course'] for e in self.episode_errors]
            
            return {
                'mean_lateral': np.mean(laterals),
                'std_lateral': np.std(laterals),
                'mean_course': np.mean(courses),
                'std_course': np.std(courses),
                'episodes': len(self.episode_errors)
            }
        return None
    
    def save_errors(self, save_path):
        """保存误差数据到CSV（使用UTF-8编码）"""
        if len(self.episode_errors) > 0:
            lateral_errors = [e['lateral'] for e in self.episode_errors]
            course_errors = [e['course'] for e in self.episode_errors]
            
            # 使用UTF-8编码保存
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('lateral_error,course_error\n')
                for lateral, course in zip(lateral_errors, course_errors):
                    f.write(f'{lateral:.6f},{course:.6f}\n')
            
            print(f"\n📊 Stanley基线误差已保存: {save_path}")


# ==================== 🆕🆕 智能性能评估模块 ====================

class PathTrackingEvaluator:
    """路径跟踪智能性能评估模块（新增）"""
    
    @staticmethod
    def evaluate_tracking_quality(summary, baseline_stats=None, path_type="complex"):
        """
        综合评估路径跟踪质量
        
        返回:
            results: 包含评分、等级、评价的字典
        """
        results = {
            'scores': {},
            'grades': {},
            'comments': []
        }
        
        # ========== 1. 跟踪精度评估 ==========
        avg_lateral = summary['avg_lateral']
        
        if avg_lateral < 0.1:
            accuracy_score = 100
            accuracy_grade = "A+"
            accuracy_comment = f"卓越！{path_type}路径跟踪精度极高"
        elif avg_lateral < 0.2:
            accuracy_score = 90
            accuracy_grade = "A"
            accuracy_comment = f"优秀！{path_type}路径跟踪精度很高"
        elif avg_lateral < 0.3:
            accuracy_score = 80
            accuracy_grade = "B+"
            accuracy_comment = f"良好！{path_type}路径跟踪精度可接受"
        elif avg_lateral < 0.5:
            accuracy_score = 70
            accuracy_grade = "B"
            accuracy_comment = "中等，跟踪精度偏低"
        else:
            accuracy_score = 60
            accuracy_grade = "C"
            accuracy_comment = "需要改进，跟踪偏差过大"
        
        results['scores']['accuracy'] = accuracy_score
        results['grades']['accuracy'] = accuracy_grade
        results['comments'].append(f"📍 跟踪精度: {accuracy_grade} - {accuracy_comment}")
        
        # ========== 2. 稳定性评估 ==========
        std_lateral = summary['std_lateral']
        
        if std_lateral < 0.05:
            stability_score = 100
            stability_grade = "A+"
            stability_comment = "非常稳定，误差波动极小"
        elif std_lateral < 0.1:
            stability_score = 85
            stability_grade = "A"
            stability_comment = "稳定性好，弯道切换平滑"
        elif std_lateral < 0.2:
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
        
        # ========== 3. 控制平滑度评估 ==========
        avg_action_change = summary.get('avg_action_change', 0)
        
        if avg_action_change < 0.01:
            smoothness_score = 100
            smoothness_grade = "A+"
            smoothness_comment = "控制非常平滑"
        elif avg_action_change < 0.05:
            smoothness_score = 85
            smoothness_grade = "A"
            smoothness_comment = "控制平滑，动作合理"
        elif avg_action_change < 0.1:
            smoothness_score = 70
            smoothness_grade = "B"
            smoothness_comment = "控制一般，略有波动"
        else:
            smoothness_score = 60
            smoothness_grade = "C"
            smoothness_comment = "控制不够平滑"
        
        results['scores']['smoothness'] = smoothness_score
        results['grades']['smoothness'] = smoothness_grade
        results['comments'].append(f"🌊 平滑度: {smoothness_grade} - {smoothness_comment}")
        
        # ========== 4. 航向控制评估 ==========
        avg_course = summary['avg_course']
        course_deg = np.rad2deg(avg_course)
        
        if avg_course < 0.1:
            results['comments'].append(f"🎯 航向控制: 优秀 ({course_deg:.2f}°)")
        elif avg_course < 0.2:
            results['comments'].append(f"🎯 航向控制: 良好 ({course_deg:.2f}°)")
        else:
            results['comments'].append(f"⚠️  航向控制: 需改进 ({course_deg:.2f}°)")
        
        # ========== 🆕🆕🆕 5. 倾斜角控制评估 ==========
        avg_tilt = summary.get('avg_tilt', 0)
        tilt_deg = np.rad2deg(avg_tilt)
        
        if avg_tilt < 0.05:
            results['comments'].append(f"⚖️  倾斜角控制: 优秀 ({tilt_deg:.2f}°)")
        elif avg_tilt < 0.1:
            results['comments'].append(f"⚖️  倾斜角控制: 良好 ({tilt_deg:.2f}°)")
        else:
            results['comments'].append(f"⚠️  倾斜角控制: 需改进 ({tilt_deg:.2f}°)")
        
        # ========== 6. 综合评分 ==========
        overall_score = (accuracy_score * 0.5 + 
                        stability_score * 0.3 + 
                        smoothness_score * 0.2)
        
        if overall_score >= 90:
            overall_grade = "A+"
            overall_comment = "🏆 卓越表现！"
        elif overall_score >= 80:
            overall_grade = "A"
            overall_comment = "⭐ 优秀表现！"
        elif overall_score >= 70:
            overall_grade = "B"
            overall_comment = "✅ 良好表现"
        elif overall_score >= 60:
            overall_grade = "C"
            overall_comment = "⚠️  中等表现"
        else:
            overall_grade = "D"
            overall_comment = "❌ 需要改进"
        
        results['overall_score'] = overall_score
        results['overall_grade'] = overall_grade
        results['overall_comment'] = overall_comment
        
        # ========== 7. 与基线对比 ==========
        if baseline_stats:
            baseline_lateral = baseline_stats['mean_lateral']
            lateral_improvement = (1 - avg_lateral / baseline_lateral) * 100
            results['baseline_improvement'] = lateral_improvement
            
            if lateral_improvement > 20:
                results['baseline_effect'] = "🎉 显著优于基线！"
            elif lateral_improvement > 10:
                results['baseline_effect'] = "✅ 明显优于基线"
            elif lateral_improvement > 0:
                results['baseline_effect'] = "☑️  略优于基线"
            else:
                results['baseline_effect'] = "⚠️  不如基线"
        
        return results


# ==================== 🆕🆕🆕 完整增强版误差追踪器（含倾斜角） ====================

class DetailedErrorTracker:
    """
    详细的误差追踪器（完整增强版+倾斜角误差）
    新增：路径进度、终止原因、完成率统计、智能评估、倾斜角误差
    """
    
    def __init__(self, log_dir, path_type="complex", baseline_stats=None):
        self.log_dir = log_dir
        self.path_type = path_type  # 🆕 路径类型
        self.baseline_stats = baseline_stats
        
        # 当前episode的记录
        self.current_lateral_errors = []
        self.current_course_errors = []
        self.current_tilt_errors = []  # 🆕🆕🆕 倾斜角误差
        self.current_velocities = []
        self.current_actions = []
        self.current_positions = []  # 🆕 位置记录
        
        # 历史记录
        self.episode_summaries = []
        
        # 🆕 最优模型追踪
        self.best_lateral_error = float('inf')
        self.best_episode = None
        self.best_episode_summary = None
        
        # 🆕🆕 完成率统计
        self.completed_episodes = []
        self.completion_steps = []
        
        # CSV文件路径
        self.csv_path = os.path.join(log_dir, "rl_training_errors.csv")
        self.detailed_csv_path = os.path.join(log_dir, "rl_detailed_errors.csv")
        
        # 🆕🆕🆕 初始化CSV文件（增强版列 + 倾斜角误差）
        with open(self.csv_path, 'w', encoding='utf-8') as f:
            f.write("episode,steps,completed,completion_steps,termination_reason,progress_percent,"
                   "avg_lateral,std_lateral,max_lateral,avg_course,std_course,max_course,"
                   "avg_tilt,std_tilt,max_tilt,"  # 🆕🆕🆕 倾斜角误差列
                   "avg_velocity,avg_action_change,reward\n")
    
    def record_step(self, obs, action, reward, info=None):
        """记录单步数据（增强版，支持info + 倾斜角完整追踪）"""
        # 反归一化
        lateral_error = abs(obs[0] * 10)
        course_error = abs(obs[1] * 1.57)
        velocity = obs[2] * 5
        
        # 🆕🆕🆕 获取倾斜角 - 完整版（误差+实际值）
        tilt_error = 0
        actual_tilt = 0
        
        # 优先从info中获取（最准确）
        if info and isinstance(info, dict):
            # 方案1：info中有明确的倾斜角信息
            if 'tilt_error' in info:
                tilt_error = abs(info['tilt_error'])
            if 'actual_tilt' in info or 'tilt' in info:
                actual_tilt = abs(info.get('actual_tilt', info.get('tilt', 0)))
            
            # 如果info中没有，尝试从obs中获取
            if tilt_error == 0 and len(obs) > 3:
                # 尝试不同的归一化系数
                # 0.524 rad = 30度, 0.349 rad = 20度
                tilt_error = abs(obs[3] * 0.524)
        else:
            # 备用方案：从observation中提取
            if len(obs) > 3:
                tilt_error = abs(obs[3] * 0.524)
        
        # 🆕🆕🆕 记录倾斜角误差（如果有实际值，优先使用实际值）
        if actual_tilt > 0:
            self.current_tilt_errors.append(actual_tilt)  # 记录实际倾斜角
        else:
            self.current_tilt_errors.append(tilt_error)   # 记录倾斜角误差
        
        self.current_lateral_errors.append(lateral_error)
        self.current_course_errors.append(course_error)
        self.current_velocities.append(velocity)
        self.current_actions.append(action.copy())
        
        # 🆕🆕 从info中获取位置信息
        if info and isinstance(info, dict):
            x = info.get('x', info.get('forward_wheel_x', None))
            y = info.get('y', info.get('forward_wheel_y', None))
            if x is not None and y is not None:
                self.current_positions.append((x, y))
    
    # 🆕🆕 路径进度追踪模块
    def get_path_progress(self):
        """计算路径完成进度"""
        if len(self.current_positions) == 0:
            return 0.0
        
        if self.path_type == "s_line":
            # S型路径：y从0到60
            max_y = max(y for _, y in self.current_positions)
            progress = min(100.0, (max_y / 60.0) * 100)
        else:
            # 复杂路径：根据位置判断完成了哪些段
            max_y = max(y for _, y in self.current_positions) if self.current_positions else 0
            
            # 简化版进度估算
            if max_y < 5:
                progress = (max_y / 5) * 20  # 第1段：0-20%
            elif max_y < 35:
                progress = 20 + ((max_y - 5) / 30) * 40  # 第2-3段：20-60%
            elif max_y < 55:
                progress = 60 + ((max_y - 35) / 20) * 30  # 第4-5段：60-90%
            else:
                progress = 90 + min(10, (max_y - 55) / 5 * 10)  # 第6段：90-100%
        
        return min(100.0, progress)
    
    def get_segment_info(self):
        """返回当前所在路径段信息"""
        if len(self.current_positions) == 0:
            return "未开始", 0.0
        
        final_x, final_y = self.current_positions[-1]
        
        if self.path_type == "s_line":
            if final_y < 15:
                segment = "第一段圆弧-前半"
                progress = (final_y / 15) * 100
            elif final_y < 30:
                segment = "第一段圆弧-后半"
                progress = ((final_y - 15) / 15) * 100
            elif final_y < 45:
                segment = "第二段圆弧-前半"
                progress = ((final_y - 30) / 15) * 100
            else:
                segment = "第二段圆弧-后半"
                progress = ((final_y - 45) / 15) * 100
        else:
            # 复杂路径段判断
            if final_y < 5:
                segment = "第1段直线"
                progress = (final_y / 5) * 100
            elif final_y < 15:
                segment = "第2段圆弧"
                progress = ((final_y - 5) / 10) * 100
            elif final_y < 35:
                segment = "第3段直线"
                progress = ((final_y - 15) / 20) * 100
            elif final_y < 50:
                segment = "第4段圆弧"
                progress = ((final_y - 35) / 15) * 100
            elif final_y < 35 and final_x < 40:
                segment = "第5段圆弧"
                progress = 50.0
            else:
                segment = "第6段直线"
                progress = 80.0
        
        return segment, min(100.0, progress)
    
    def check_path_completion(self):
        """检查是否完成了完整路径"""
        if len(self.current_positions) < 10:
            return False, 0
        
        y_positions = [y for _, y in self.current_positions]
        max_y = max(y_positions)
        final_x, final_y = self.current_positions[-1]
        
        if self.path_type == "s_line":
            # S型：y > 59.5
            if final_y > 59.5 and max_y > 25 and -20 < final_x < 20:
                for i, (x, y) in enumerate(self.current_positions):
                    if y > 59.5:
                        return True, i + 1
        else:
            # 复杂路径：到达终点区域
            if final_y > 55 and 0 < final_x < 28:
                for i, (x, y) in enumerate(self.current_positions):
                    if y > 55:
                        return True, i + 1
        
        return False, 0
    
    def episode_end(self, episode_num, episode_reward, max_steps, termination_reason="unknown"):
        """Episode结束时的处理（增强版 + 倾斜角）"""
        if len(self.current_lateral_errors) == 0:
            return
        
        # 🆕🆕 检查是否完成路径
        is_completed, completion_step = self.check_path_completion()
        
        if is_completed and termination_reason != "task_complete":
            termination_reason = "task_complete"
        
        if is_completed:
            self.completed_episodes.append(episode_num)
            self.completion_steps.append(completion_step)
        
        # 🆕🆕 计算路径进度
        progress_percent = self.get_path_progress()
        
        # 计算统计量
        avg_lateral = np.mean(self.current_lateral_errors)
        std_lateral = np.std(self.current_lateral_errors)
        max_lateral = np.max(self.current_lateral_errors)
        
        avg_course = np.mean(self.current_course_errors)
        std_course = np.std(self.current_course_errors)
        max_course = np.max(self.current_course_errors)
        
        # 🆕🆕🆕 计算倾斜角统计量
        avg_tilt = np.mean(self.current_tilt_errors)
        std_tilt = np.std(self.current_tilt_errors)
        max_tilt = np.max(self.current_tilt_errors)
        
        avg_velocity = np.mean(self.current_velocities)
        
        # 🆕 计算动作变化（平滑度）
        actions_array = np.array(self.current_actions)
        if len(actions_array) > 1:
            action_changes = np.abs(np.diff(actions_array, axis=0))
            avg_action_change = np.mean(action_changes)
        else:
            avg_action_change = 0
        
        steps = len(self.current_lateral_errors)
        
        # 保存摘要（含倾斜角）
        summary = {
            'episode': episode_num,
            'steps': steps,
            'completed': is_completed,
            'completion_steps': completion_step if is_completed else 0,
            'termination_reason': termination_reason,
            'progress_percent': progress_percent,
            'avg_lateral': avg_lateral,
            'std_lateral': std_lateral,
            'max_lateral': max_lateral,
            'avg_course': avg_course,
            'std_course': std_course,
            'max_course': max_course,
            'avg_tilt': avg_tilt,  # 🆕🆕🆕
            'std_tilt': std_tilt,  # 🆕🆕🆕
            'max_tilt': max_tilt,  # 🆕🆕🆕
            'avg_velocity': avg_velocity,
            'avg_action_change': avg_action_change,
            'reward': episode_reward
        }
        self.episode_summaries.append(summary)
        
        # 🆕🆕🆕 写入增强版CSV（含倾斜角误差）
        with open(self.csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{episode_num},{steps},{int(is_completed)},{completion_step},"
                   f"{termination_reason},{progress_percent:.2f},"
                   f"{avg_lateral:.6f},{std_lateral:.6f},{max_lateral:.6f},"
                   f"{avg_course:.6f},{std_course:.6f},{max_course:.6f},"
                   f"{avg_tilt:.6f},{std_tilt:.6f},{max_tilt:.6f},"  # 🆕🆕🆕 倾斜角数据
                   f"{avg_velocity:.2f},{avg_action_change:.6f},{episode_reward:.2f}\n")
        
        # 🆕 更新最优模型 - 只考虑完成度较高的episode
        completion_ratio = steps / max_steps
        MIN_COMPLETION = 0.8
        
        if completion_ratio >= MIN_COMPLETION and avg_lateral < self.best_lateral_error:
            self.best_lateral_error = avg_lateral
            self.best_episode = episode_num
            self.best_episode_summary = summary.copy()
            print(f"\n   🏆 新的最优Episode！横向误差: {avg_lateral:.6f} m (完成度: {completion_ratio*100:.1f}%)")
        
        # 打印增强版摘要
        self._print_episode_summary(summary)
        
        # 清空当前记录
        self.current_lateral_errors = []
        self.current_course_errors = []
        self.current_tilt_errors = []  # 🆕🆕🆕
        self.current_velocities = []
        self.current_actions = []
        self.current_positions = []
    
    def _print_episode_summary(self, summary):
        """打印episode摘要（增强版，显示路径进度和智能评估 + 倾斜角）"""
        episode = summary['episode']
        
        # 🆕🆕 终止原因映射
        termination_reason_map = {
            "task_complete": "✅ 任务完成",
            "lateral_error": "❌ 横向误差过大",
            "tilt_error": "❌ 倾斜角过大",
            "heading_error": "❌ 航向误差过大",
            "timeout": "⏱️  超时",
            "unknown": "❓ 未知原因"
        }
        
        print(f"\n{'='*80}")
        print(f"📊 Episode {episode} 完成")
        print(f"{'='*80}")
        
        # 🆕🆕 显示终止原因
        termination_reason = summary.get('termination_reason', 'unknown')
        termination_desc = termination_reason_map.get(termination_reason, "❓ 未知")
        print(f"🔚 终止原因: {termination_desc}")
        
        # 🆕🆕 显示路径进度
        progress = summary['progress_percent']
        segment, segment_progress = self.get_segment_info()
        
        print(f"🛣️  路径进度: {progress:.1f}% (当前: {segment} {segment_progress:.1f}%)")
        
        # 🆕🆕 显示完成状态
        if summary['completed']:
            print(f"🎉 路径完成: ✅ ({summary['completion_steps']} 步)")
            print(f"   完成率: {len(self.completed_episodes)}/{episode} = "
                  f"{len(self.completed_episodes)/episode*100:.1f}%")
        else:
            if progress > 90:
                print(f"⚠️  路径完成: 未完成（但已完成 {progress:.1f}%，非常接近！）")
            elif progress > 80:
                print(f"❌ 路径完成: 未完成（已完成 {progress:.1f}%）")
            else:
                print(f"❌ 路径完成: 未完成")
        
        print(f"\n🎯 基本信息:")
        print(f"   Steps: {summary['steps']:4d} | Reward: {summary['reward']:7.2f} | Velocity: {summary['avg_velocity']:.2f} m/s")
        
        # 误差详情
        print(f"\n📏 横向误差 (Lateral Error):")
        print(f"   平均: {summary['avg_lateral']:.4f} m | 标准差: {summary['std_lateral']:.4f} m | 最大: {summary['max_lateral']:.4f} m")
        
        print(f"\n🧭 航向误差 (Course Error):")
        print(f"   平均: {summary['avg_course']:.4f} rad | 标准差: {summary['std_course']:.4f} rad | 最大: {summary['max_course']:.4f} rad")
        
        # 🆕🆕🆕 显示倾斜角误差（增强诊断版）
        print(f"\n⚖️  倾斜角误差 (Tilt Error):")
        print(f"   平均: {summary['avg_tilt']:.4f} rad ({np.rad2deg(summary['avg_tilt']):.2f}°) | "
              f"标准差: {summary['std_tilt']:.4f} rad | 最大: {summary['max_tilt']:.4f} rad ({np.rad2deg(summary['max_tilt']):.2f}°)")
        
        # 🆕🆕🆕 倾斜角倾倒诊断
        if termination_reason == "tilt_error":
            print(f"\n   ⚠️  倾倒诊断:")
            print(f"      - 最大倾斜角: {np.rad2deg(summary['max_tilt']):.2f}° (可能触发终止阈值)")
            print(f"      - 平均倾斜角: {np.rad2deg(summary['avg_tilt']):.2f}° (正常)")
            print(f"      - 建议: 检查环境倾斜角阈值设置，或检查obs[3]是否为实际倾斜角而非误差")
            
            # 如果最大值明显大于平均值
            if summary['max_tilt'] > summary['avg_tilt'] * 3:
                print(f"      - ⚠️  检测到倾斜角突变！最大值是平均值的{summary['max_tilt']/summary['avg_tilt']:.1f}倍")
                print(f"      - 可能原因：控制器输出不稳定，建议降低学习率或增加训练步数")
        
        # 🆕 显示控制平滑度
        print(f"\n🌊 控制平滑度: {summary['avg_action_change']:.6f}")
        
        # 与基线对比
        if self.baseline_stats:
            print(f"\n🔬 与Stanley基线对比:")
            
            lateral_diff = summary['avg_lateral'] - self.baseline_stats['mean_lateral']
            lateral_pct = (lateral_diff / self.baseline_stats['mean_lateral']) * 100
            
            if lateral_diff < 0:
                print(f"   横向误差: ✅ 优于基线 {abs(lateral_diff):.4f}m ({abs(lateral_pct):.1f}% 更好)")
            else:
                print(f"   横向误差: ⚠️  差于基线 {lateral_diff:.4f}m ({lateral_pct:.1f}% 更差)")
            
            course_diff = summary['avg_course'] - self.baseline_stats['mean_course']
            course_pct = (course_diff / self.baseline_stats['mean_course']) * 100
            
            if course_diff < 0:
                print(f"   航向误差: ✅ 优于基线 {abs(course_diff):.4f}rad ({abs(course_pct):.1f}% 更好)")
            else:
                print(f"   航向误差: ⚠️  差于基线 {course_diff:.4f}rad ({course_pct:.1f}% 更差)")
            
            print(f"\n   基线参考: Lateral={self.baseline_stats['mean_lateral']:.4f}m, "
                  f"Course={self.baseline_stats['mean_course']:.4f}rad")
        
        # 🆕🆕 智能评估
        eval_results = PathTrackingEvaluator.evaluate_tracking_quality(
            summary, self.baseline_stats, self.path_type
        )
        
        print(f"\n{'─'*80}")
        print(f"🎓 性能评估")
        print(f"{'─'*80}")
        for comment in eval_results['comments']:
            print(f"   {comment}")
        
        print(f"\n   🏆 综合: {eval_results['overall_score']:.1f}/100 "
              f"({eval_results['overall_grade']}) - {eval_results['overall_comment']}")
        
        print(f"{'='*80}\n")
    
    def get_recent_stats(self, last_n=20):
        """获取最近N个episode的统计（增强版 + 倾斜角）"""
        if len(self.episode_summaries) < 1:
            return None
        
        last_n = min(last_n, len(self.episode_summaries))
        recent = self.episode_summaries[-last_n:]
        
        # 🆕🆕 完成率统计
        recent_completed = [s for s in recent if s['completed']]
        completion_rate = len(recent_completed) / last_n * 100 if last_n > 0 else 0
        avg_completion_steps = (np.mean([s['completion_steps'] for s in recent_completed]) 
                               if recent_completed else 0)
        avg_progress = np.mean([s['progress_percent'] for s in recent])
        
        return {
            'avg_lateral': np.mean([s['avg_lateral'] for s in recent]),
            'avg_course': np.mean([s['avg_course'] for s in recent]),
            'avg_tilt': np.mean([s['avg_tilt'] for s in recent]),  # 🆕🆕🆕
            'avg_reward': np.mean([s['reward'] for s in recent]),
            'completion_rate': completion_rate,
            'avg_completion_steps': avg_completion_steps,
            'avg_progress': avg_progress,
            'episodes': last_n
        }
    
    def save_final_report(self):
        """保存最终报告（增强版 + 倾斜角）"""
        if len(self.episode_summaries) == 0:
            return
        
        report_path = os.path.join(self.log_dir, "rl_training_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"强化学习训练完整报告 - {self.path_type.upper()}路径\n")
            f.write("="*80 + "\n\n")
            
            # 整体统计
            all_lateral = [s['avg_lateral'] for s in self.episode_summaries]
            all_course = [s['avg_course'] for s in self.episode_summaries]
            all_tilt = [s['avg_tilt'] for s in self.episode_summaries]  # 🆕🆕🆕
            all_rewards = [s['reward'] for s in self.episode_summaries]
            all_progress = [s['progress_percent'] for s in self.episode_summaries]
            
            # 🆕🆕 完成率统计
            total_episodes = len(self.episode_summaries)
            total_completed = len(self.completed_episodes)
            completion_rate = (total_completed / total_episodes * 100) if total_episodes > 0 else 0
            
            f.write("📊 整体统计:\n")
            f.write(f"  总Episodes: {total_episodes}\n")
            f.write(f"  路径完成数: {total_completed} ({completion_rate:.1f}%)\n")
            f.write(f"  平均路径进度: {np.mean(all_progress):.1f}%\n")
            f.write(f"  平均横向误差: {np.mean(all_lateral):.6f} ± {np.std(all_lateral):.6f} m\n")
            f.write(f"  平均航向误差: {np.mean(all_course):.6f} ± {np.std(all_course):.6f} rad\n")
            f.write(f"  平均倾斜角误差: {np.mean(all_tilt):.6f} ± {np.std(all_tilt):.6f} rad ({np.rad2deg(np.mean(all_tilt)):.2f}°)\n")  # 🆕🆕🆕
            f.write(f"  平均奖励: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n\n")
            
            # 最近20集统计
            recent_stats = self.get_recent_stats(20)
            if recent_stats:
                f.write("📈 最近20集统计:\n")
                f.write(f"  完成率: {recent_stats['completion_rate']:.1f}%\n")
                f.write(f"  平均进度: {recent_stats['avg_progress']:.1f}%\n")
                f.write(f"  平均横向误差: {recent_stats['avg_lateral']:.6f} m\n")
                f.write(f"  平均航向误差: {recent_stats['avg_course']:.6f} rad\n")
                f.write(f"  平均倾斜角误差: {recent_stats['avg_tilt']:.6f} rad ({np.rad2deg(recent_stats['avg_tilt']):.2f}°)\n")  # 🆕🆕🆕
                f.write(f"  平均奖励: {recent_stats['avg_reward']:.2f}\n\n")
            
            # 🆕 最优Episode详情
            f.write("="*80 + "\n")
            f.write("🏆 最优模型 (训练过程中横向误差最小的Episode):\n")
            f.write("="*80 + "\n")
            if self.best_episode_summary:
                best = self.best_episode_summary
                f.write(f"  Episode编号: {best['episode']}\n")
                f.write(f"  步数: {best['steps']}\n")
                f.write(f"  路径进度: {best['progress_percent']:.1f}%\n")
                f.write(f"  完成状态: {'✅' if best['completed'] else '❌'}\n")
                f.write(f"  横向误差: {best['avg_lateral']:.6f} m (±{best['std_lateral']:.6f})\n")
                f.write(f"  最大横向误差: {best['max_lateral']:.6f} m\n")
                f.write(f"  航向误差: {best['avg_course']:.6f} rad (±{best['std_course']:.6f})\n")
                f.write(f"  最大航向误差: {best['max_course']:.6f} rad\n")
                f.write(f"  倾斜角误差: {best['avg_tilt']:.6f} rad (±{best['std_tilt']:.6f}) ({np.rad2deg(best['avg_tilt']):.2f}°)\n")  # 🆕🆕🆕
                f.write(f"  最大倾斜角误差: {best['max_tilt']:.6f} rad ({np.rad2deg(best['max_tilt']):.2f}°)\n")  # 🆕🆕🆕
                f.write(f"  奖励: {best['reward']:.2f}\n\n")
                
                # 🆕 最优模型与基线对比
                if self.baseline_stats:
                    baseline_lateral = self.baseline_stats['mean_lateral']
                    baseline_course = self.baseline_stats['mean_course']
                    best_lateral = best['avg_lateral']
                    best_course = best['avg_course']
                    
                    lateral_improvement = (1 - best_lateral / baseline_lateral) * 100
                    course_improvement = (1 - best_course / baseline_course) * 100
                    
                    f.write("  🔬 最优模型 vs Stanley基线:\n")
                    f.write(f"     基线横向误差: {baseline_lateral:.6f} m\n")
                    f.write(f"     最优横向误差: {best_lateral:.6f} m\n")
                    f.write(f"     横向误差提升: {lateral_improvement:+.2f}%\n\n")
                    
                    f.write(f"     基线航向误差: {baseline_course:.6f} rad\n")
                    f.write(f"     最优航向误差: {best_course:.6f} rad\n")
                    f.write(f"     航向误差提升: {course_improvement:+.2f}%\n\n")
                    
                    if lateral_improvement > 20 and course_improvement > 20:
                        f.write(f"     评价: 🎉🎉🎉 卓越！RL控制器大幅超越Stanley基线\n\n")
                    elif lateral_improvement > 10 and course_improvement > 10:
                        f.write(f"     评价: ✅✅ 优秀！RL控制器显著优于基线\n\n")
                    elif lateral_improvement > 5 and course_improvement > 5:
                        f.write(f"     评价: ✅ 良好！RL控制器有效提升性能\n\n")
                    elif lateral_improvement > 0 or course_improvement > 0:
                        f.write(f"     评价: ☑️  一般，略有提升\n\n")
                    else:
                        f.write(f"     评价: ⚠️  需要改进，未能超越基线\n\n")
            
            # 🆕🆕 最高进度Episode
            if all_progress:
                max_progress_idx = np.argmax(all_progress)
                max_progress_ep = self.episode_summaries[max_progress_idx]
                f.write("🏆 最高进度Episode:\n")
                f.write(f"  Episode: {max_progress_ep['episode']}\n")
                f.write(f"  路径进度: {max_progress_ep['progress_percent']:.1f}%\n")
                f.write(f"  位置偏差: {max_progress_ep['avg_lateral']:.6f} m\n")
                f.write(f"  倾斜角误差: {max_progress_ep['avg_tilt']:.6f} rad ({np.rad2deg(max_progress_ep['avg_tilt']):.2f}°)\n")  # 🆕🆕🆕
                f.write(f"  完成状态: {'✅' if max_progress_ep['completed'] else '❌'}\n\n")
            
            # 与基线对比（整体）
            if self.baseline_stats:
                f.write("="*80 + "\n")
                f.write("🔬 与Stanley基线对比 (整体表现):\n")
                f.write("="*80 + "\n")
                f.write(f"  基线横向误差: {self.baseline_stats['mean_lateral']:.6f} m\n")
                f.write(f"  基线航向误差: {self.baseline_stats['mean_course']:.6f} rad\n\n")
                
                if recent_stats:
                    lateral_improvement = (1 - recent_stats['avg_lateral']/self.baseline_stats['mean_lateral']) * 100
                    course_improvement = (1 - recent_stats['avg_course']/self.baseline_stats['mean_course']) * 100
                    
                    f.write("  性能提升 (最近20集):\n")
                    f.write(f"    横向误差: {lateral_improvement:+.2f}%\n")
                    f.write(f"    航向误差: {course_improvement:+.2f}%\n\n")
                    
                    if lateral_improvement > 0 and course_improvement > 0:
                        f.write("  ✅ RL控制器成功超越Stanley基线\n\n")
                    elif lateral_improvement > 0 or course_improvement > 0:
                        f.write("  ☑️  RL控制器在某些方面优于基线\n\n")
                    else:
                        f.write("  ⚠️  RL控制器未能超越基线，建议检查配置\n\n")
            
            f.write("="*80 + "\n")
            f.write("📝 训练总结:\n")
            f.write("="*80 + "\n")
            f.write(f"  ✅ 完成{len(self.episode_summaries)}个Episodes的训练\n")
            f.write(f"  ✅ 详细误差数据已保存至: {self.csv_path}\n")
            f.write(f"  ✅ 倾斜角误差追踪已启用\n")  # 🆕🆕🆕
            
            if self.best_episode_summary and self.baseline_stats:
                lateral_improvement = (1 - self.best_episode_summary['avg_lateral'] / 
                                     self.baseline_stats['mean_lateral']) * 100
                course_improvement = (1 - self.best_episode_summary['avg_course'] / 
                                    self.baseline_stats['mean_course']) * 100
                
                if lateral_improvement > 0 and course_improvement > 0:
                    f.write(f"  ✅ 最优模型相较基线提升:\n")
                    f.write(f"     横向误差: {lateral_improvement:.2f}%\n")
                    f.write(f"     航向误差: {course_improvement:.2f}%\n")
                else:
                    f.write(f"  ⚠️  建议检查训练配置和超参数\n")
            
            f.write("="*80 + "\n")
        
        print(f"\n📄 完整报告已保存: {report_path}")


# ==================== 🆕🆕 增强版回调函数（带错误处理）====================

class EnhancedTrainingCallback(BaseCallback):
    """
    增强版回调函数（完整版）
    新增：错误处理、info利用、路径进度追踪
    """
    
    def __init__(self, check_freq: int, log_dir: str, error_tracker: DetailedErrorTracker, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.error_tracker = error_tracker
        self.save_path = os.path.join(log_dir, "stage3_agent_stanley_best")
        self.best_mean_reward = -np.inf
        
        # Episode追踪
        self.current_episode = 0
        self.episode_reward = 0
        self.episode_step = 0
        
        # 🆕 最优模型追踪文件
        self.best_model_info_path = os.path.join(log_dir, "best_model_info.txt")
        self.qualified_rewards = deque(maxlen=20)  # 存储最近20个合格回合的奖励
        self.best_filtered_reward = -float('inf')  # 记录过滤后的最佳平均分
        self.filtered_save_path = os.path.join(log_dir, "best_filtered_model") # 专用保存路径

    def _on_step(self) -> bool:
        # 🆕🆕 增强错误处理
        try:
            # 安全获取obs
            obs = (self.locals['new_obs'][0] if len(self.locals['new_obs'].shape) > 1 
                   else self.locals['new_obs'])
            
            # 安全获取action
            action = (self.locals['actions'][0] if len(self.locals['actions'].shape) > 1 
                     else self.locals['actions'])
            
            # 安全获取reward
            reward = (self.locals['rewards'][0] if hasattr(self.locals['rewards'], '__len__') 
                     else self.locals['rewards'])
            
            # 🆕🆕 安全获取info
            infos = self.locals.get('infos', [{}])
            info = infos[0] if isinstance(infos, list) and len(infos) > 0 else {}
            
            # 记录步骤（传入info）
            self.error_tracker.record_step(obs, action, reward, info)
            self.episode_reward += reward
            self.episode_step += 1
            
            # 检查episode是否结束
            dones = self.locals.get('dones', [False])
            done = dones[0] if hasattr(dones, '__len__') else dones
            
            if done:
                if self.episode_step > 1800 and self.episode_reward > 0:
                    
                    # 1. 加入合格队列
                    # 这里的 deque(maxlen=20) 会自动处理 "第21个来了挤走第1个" 的逻辑
                    self.qualified_rewards.append(self.episode_reward)
                    
                    # 2. 计算当前队列的平均值
                    # 这里的 np.mean 会自动处理分母：
                    # - 只有1个时，除以1
                    # - 有19个时，除以19
                    # - 有20个时，除以20
                    avg_filtered = np.mean(self.qualified_rewards)
                    
                    if self.verbose > 0:
                        print(f"   ✅ 合格回合! 步数:{self.episode_step}, 奖励:{self.episode_reward:.1f}")
                        print(f"      当前{len(self.qualified_rewards)}个合格回合均分: {avg_filtered:.2f}")

                    # 3. 只要比历史最佳高，立刻保存！
                    # (不需要等凑齐20个，从第1个就开始PK)
                    if avg_filtered > self.best_filtered_reward:
                        self.best_filtered_reward = avg_filtered
                        
                        print(f"\n{'='*60}")
                        print(f"🔥 发现新的【最佳过滤模型】(>1000步 & >0分)")
                        print(f"   当前均分: {avg_filtered:.2f} (基于最近 {len(self.qualified_rewards)} 个合格数据)")
                        print(f"   💾 保存到: {self.filtered_save_path}")
                        print(f"{'='*60}\n")
                        
                        self.model.save(self.filtered_save_path)
                self.current_episode += 1
                
                # 🆕🆕 获取终止原因
                termination_reason = "unknown"
                if isinstance(info, dict) and 'termination_reason' in info:
                    termination_reason = info['termination_reason']
                
                # 获取max_steps
                max_steps = self.training_env.get_attr('max_step_num')[0]
                
                # 调用episode_end（传入终止原因）
                self.error_tracker.episode_end(
                    self.current_episode, 
                    self.episode_reward, 
                    max_steps,
                    termination_reason  # 🆕
                )
                
                self.episode_reward = 0
                self.episode_step = 0
            
            # 定期检查并保存最佳模型
            if self.n_calls % self.check_freq == 0:
                try:
                    x, y = ts2xy(load_results(self.log_dir), "timesteps")
                    if len(x) > 0:
                        mean_reward = np.mean(y[-20:]) if len(y) >= 20 else np.mean(y)
                        
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            if self.verbose >= 1:
                                print(f"\n{'='*60}")
                                print(f"🎉 新的最佳模型！平均奖励: {mean_reward:.2f}")
                                print(f"💾 保存到: {self.save_path}")
                                
                                # 显示当前性能
                                recent_stats = self.error_tracker.get_recent_stats(20)
                                if recent_stats:
                                    print(f"\n当前性能 (最近20集):")
                                    print(f"  完成率: {recent_stats['completion_rate']:.1f}%")
                                    print(f"  平均进度: {recent_stats['avg_progress']:.1f}%")
                                    print(f"  横向误差: {recent_stats['avg_lateral']:.4f} m")
                                    print(f"  航向误差: {recent_stats['avg_course']:.4f} rad")
                                    print(f"  倾斜角误差: {recent_stats['avg_tilt']:.4f} rad ({np.rad2deg(recent_stats['avg_tilt']):.2f}°)")  # 🆕🆕🆕
                                    
                                    # 与基线对比
                                    if self.error_tracker.baseline_stats:
                                        baseline_lateral = self.error_tracker.baseline_stats['mean_lateral']
                                        baseline_course = self.error_tracker.baseline_stats['mean_course']
                                        lateral_improvement = (1 - recent_stats['avg_lateral']/baseline_lateral) * 100
                                        course_improvement = (1 - recent_stats['avg_course']/baseline_course) * 100
                                        
                                        print(f"  相比Stanley基线:")
                                        print(f"    横向误差: {lateral_improvement:+.2f}%")
                                        print(f"    航向误差: {course_improvement:+.2f}%")
                                        
                                        # 🆕 保存最优模型信息到文件
                                        with open(self.best_model_info_path, 'w', encoding='utf-8') as f:
                                            f.write("="*60 + "\n")
                                            f.write("🏆 最优模型信息 (基于最近20集平均奖励)\n")
                                            f.write("="*60 + "\n\n")
                                            f.write(f"模型路径: {self.save_path}.zip\n")
                                            f.write(f"Episode: {self.current_episode}\n")
                                            f.write(f"平均奖励 (最近20集): {mean_reward:.2f}\n\n")
                                            f.write("性能指标 (最近20集):\n")
                                            f.write(f"  完成率: {recent_stats['completion_rate']:.1f}%\n")
                                            f.write(f"  平均进度: {recent_stats['avg_progress']:.1f}%\n")
                                            f.write(f"  横向误差: {recent_stats['avg_lateral']:.6f} m\n")
                                            f.write(f"  航向误差: {recent_stats['avg_course']:.6f} rad\n")
                                            f.write(f"  倾斜角误差: {recent_stats['avg_tilt']:.6f} rad ({np.rad2deg(recent_stats['avg_tilt']):.2f}°)\n\n")  # 🆕🆕🆕
                                            f.write("与Stanley基线对比:\n")
                                            f.write(f"  基线横向误差: {baseline_lateral:.6f} m\n")
                                            f.write(f"  当前横向误差: {recent_stats['avg_lateral']:.6f} m\n")
                                            f.write(f"  横向误差提升: {lateral_improvement:+.2f}%\n\n")
                                            f.write(f"  基线航向误差: {baseline_course:.6f} rad\n")
                                            f.write(f"  当前航向误差: {recent_stats['avg_course']:.6f} rad\n")
                                            f.write(f"  航向误差提升: {course_improvement:+.2f}%\n")
                                
                                print(f"{'='*60}\n")
                            self.model.save(self.save_path)
                
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"⚠️  保存模型时出错: {e}")
        
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️  回调处理出错: {e}")
        
        return True


# ==================== 🆕🆕🆕 倾斜角诊断工具 ====================

def diagnose_tilt_issue(env, num_steps=100):
    """
    倾斜角问题诊断工具
    运行若干步，打印observation和info的详细信息
    """
    print("\n" + "="*80)
    print("🔬 倾斜角诊断工具 - 分析observation和info内容")
    print("="*80)
    
    obs, info = env.reset()
    
    print(f"\n📊 Observation结构:")
    print(f"   长度: {len(obs)}")
    print(f"   内容: {obs}")
    
    if len(obs) > 3:
        print(f"\n   obs[0] (归一化): {obs[0]:.6f} → 横向误差: {abs(obs[0] * 10):.4f} m")
        print(f"   obs[1] (归一化): {obs[1]:.6f} → 航向误差: {abs(obs[1] * 1.57):.4f} rad")
        print(f"   obs[2] (归一化): {obs[2]:.6f} → 速度: {obs[2] * 5:.2f} m/s")
        print(f"   obs[3] (归一化): {obs[3]:.6f} → ???")
        print(f"\n   🤔 obs[3]可能是:")
        print(f"      - 倾斜角误差 (±30°): {abs(obs[3] * 0.524):.4f} rad ({np.rad2deg(abs(obs[3] * 0.524)):.2f}°)")
        print(f"      - 倾斜角误差 (±20°): {abs(obs[3] * 0.349):.4f} rad ({np.rad2deg(abs(obs[3] * 0.349)):.2f}°)")
        print(f"      - 实际倾斜角 (±30°): {abs(obs[3] * 0.524):.4f} rad ({np.rad2deg(abs(obs[3] * 0.524)):.2f}°)")
    
    print(f"\n📋 Info结构:")
    print(f"   keys: {info.keys() if isinstance(info, dict) else 'Not a dict'}")
    if isinstance(info, dict):
        for key, value in info.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
    
    print(f"\n🔄 运行{num_steps}步，观察倾斜角变化...")
    
    tilt_values = []
    max_tilt_obs3 = []
    
    for step in range(num_steps):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if len(obs) > 3:
            tilt_values.append(obs[3])
            max_tilt_obs3.append(abs(obs[3] * 0.524))
        
        if done or truncated:
            if isinstance(info, dict) and 'termination_reason' in info:
                print(f"\n   ⚠️  Step {step}: 终止! 原因={info['termination_reason']}")
                print(f"      obs[3]={obs[3]:.6f}, 倾斜角(±30°)={abs(obs[3] * 0.524):.4f}rad ({np.rad2deg(abs(obs[3] * 0.524)):.2f}°)")
                if 'actual_tilt' in info or 'tilt' in info:
                    actual = info.get('actual_tilt', info.get('tilt', 0))
                    print(f"      实际倾斜角={actual:.4f}rad ({np.rad2deg(abs(actual)):.2f}°)")
            break
    
    if len(tilt_values) > 0:
        print(f"\n📊 obs[3]统计 ({len(tilt_values)}步):")
        print(f"   范围: [{min(tilt_values):.4f}, {max(tilt_values):.4f}]")
        print(f"   平均: {np.mean(tilt_values):.4f}")
        print(f"   最大(绝对值): {max(map(abs, tilt_values)):.4f}")
        
        print(f"\n   如果obs[3]是倾斜角(±30°归一化):")
        print(f"      平均倾斜角: {np.mean(max_tilt_obs3):.4f} rad ({np.rad2deg(np.mean(max_tilt_obs3)):.2f}°)")
        print(f"      最大倾斜角: {max(max_tilt_obs3):.4f} rad ({np.rad2deg(max(max_tilt_obs3)):.2f}°)")
    
    print("\n" + "="*80)
    print("💡 诊断建议:")
    print("   1. 检查上方info中是否有'tilt_error'或'actual_tilt'字段")
    print("   2. 对比obs[3]和info中的倾斜角数据，确定obs[3]的真实含义")
    print("   3. 如果最大倾斜角>20°就终止，说明环境阈值可能设置为±0.35rad(20°)")
    print("   4. 确认obs[3]是'误差'还是'实际值'，然后修改record_step()函数")
    print("="*80 + "\n")


# ==================== 配置预设 ====================

def get_default_policy_kwargs(extractor_type: str):
    """获取预设配置"""
    
    configs = {
        "lightweight": {
            "features_extractor_class": LightweightAttentionExtractor,
            "features_extractor_kwargs": {
                "features_dim": 128,
                "d_model": 64,
                "nhead": 2,
                "dropout": 0.1
            }
        },
        "standard": {
            "features_extractor_class": AttentionFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1
            }
        },
        "advanced": {
            "features_extractor_class": MultiHeadAttentionFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "d_model": 128,
                "nhead": 8,
                "num_layers": 3,
                "feature_groups": 3,
                "dropout": 0.1
            }
        },
        "lstm": {
            "features_extractor_class": LSTMFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 128,
                "lstm_hidden_size": 128,
                "num_lstm_layers": 2
            }
        }
    }
    
    return configs.get(extractor_type, {})


# ==================== 配置区域（原有配置，完整保留）====================

EXTRACTOR_TYPE = "mlp"
USE_CUSTOM_CONFIG = False

# 🆕 是否使用第二阶段的残差控制器（保留原有开关，向后兼容）
USE_STAGE2_RESIDUAL = True  # True: 使用Agent_LQR+Agent_Residual, False: 只使用Agent_LQR

CUSTOM_CONFIG = {
    "extractor_class": AttentionFeaturesExtractor,
    "features_dim": 256,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.1
}

# ==================== 训练超参数 ====================

LEARNING_RATE = 0.0003
TOTAL_TIMESTEPS = 1500 * 500
BATCH_SIZE = 1024
BUFFER_SIZE = 2000000
ACTION_NOISE_SIGMA = 0.025

TARGET_POLICY_NOISE = 0.05
TARGET_NOISE_CLIP = 0.2
LEARNING_STARTS = 256

PATH_TYPE = "complex"

# ==================== Stanley基线参数 ====================

RUN_BASELINE = True
BASELINE_EPISODES = 2
STANLEY_K_LATERAL = 0.6
STANLEY_K_COURSE = 0.4

# ==================== 🆕🆕🆕 倾斜角诊断模式（保留原有配置）====================
DIAGNOSE_TILT_FIRST = False  # 建议先设置为True运行一次诊断


# ==================== Stanley基线评估函数 ====================

def evaluate_stanley_baseline(env, controller, num_episodes=50):
    """评估纯Stanley控制器的性能"""
    print("\n" + "="*80)
    print("📊 Stanley基线控制器评估")
    print("="*80)
    print(f"评估Episodes: {num_episodes}")
    print(f"参数: k_lateral={controller.k_lateral}, k_course={controller.k_course}")
    print("="*80 + "\n")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        episode_reward = 0
        
        while not (done or truncated):
            lateral_error = obs[0] * 10
            course_error_angle = obs[1] * 1.57
            velocity = obs[2] * 5
            
            k_lat_normalized = (controller.k_lateral - 0.2) / (1.0 - 0.2) * 2 - 1
            k_course_normalized = (controller.k_course - 0.2) / (0.8 - 0.2) * 2 - 1
            
            action = np.array([k_lat_normalized, k_course_normalized])
            action = np.clip(action, -1.0, 1.0)
            
            obs, reward, done, truncated, _ = env.step(action)
            
            controller.record_step(lateral_error, course_error_angle)
            
            episode_reward += reward
            step_count += 1
        
        avg_lateral, avg_course = controller.episode_end()
        
        print(f"Episode {episode+1}/{num_episodes} | "
              f"Steps: {step_count:4d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Lateral: {avg_lateral:.4f}m | "
              f"Course: {avg_course:.4f}rad")
    
    stats = controller.get_statistics()
    
    print("\n" + "="*80)
    print("📈 Stanley基线统计结果")
    print("="*80)
    if stats:
        print(f"平均横向误差: {stats['mean_lateral']:.4f} ± {stats['std_lateral']:.4f} 米")
        print(f"平均航向误差: {stats['mean_course']:.4f} ± {stats['std_course']:.4f} 弧度")
        print(f"评估Episodes: {stats['episodes']}")
    print("="*80 + "\n")
    
    return stats


# ==================== 配置选择函数 ====================

def get_policy_kwargs(extractor_type: str = "standard", custom_config: dict = None):
    """根据配置类型返回策略配置"""
    
    if custom_config is not None:
        extractor_class = custom_config.pop("extractor_class")
        print(f"\n✨ 使用自定义配置")
        print(f"   特征提取器: {extractor_class.__name__}")
        print(f"   参数: {custom_config}")
        return {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": custom_config
        }
    
    if extractor_type == "mlp":
        print(f"\n📊 使用MLP (Baseline)")
        return {
            "net_arch": dict(
                pi=[256, 256, 128],
                qf=[256, 256, 128]
            )
        }
    
    elif extractor_type in ["lightweight", "standard", "advanced", "lstm"]:
        config = get_default_policy_kwargs(extractor_type)
        extractor_name = config["features_extractor_class"].__name__
        
        icons = {
            "lightweight": "⚡",
            "standard": "🎯",
            "advanced": "🚀",
            "lstm": "🧠"
        }
        icon = icons.get(extractor_type, "✨")
        
        print(f"\n{icon} 使用{extractor_type.upper()}配置")
        print(f"   特征提取器: {extractor_name}")
        print(f"   参数: {config['features_extractor_kwargs']}")
        return config
    
    else:
        raise ValueError(f"未知的 extractor_type: {extractor_type}")


# ==================== 主训练流程（增强版，整合新功能）====================
def main():
    """主训练函数"""
    
    print("\n" + "="*80)
    print("第三阶段（完整增强版+倾斜角误差追踪+姿态环模式选择）: 自适应Stanley控制器")
    print("="*80)
    print(f"✅ 每个Episode详细记录横向误差和航向误差")
    print(f"✅ 实时对比RL性能与Stanley基线")
    print(f"✅ 保存完整的误差历史数据")
    print(f"✅ 🔧 修复UTF-8编码 - 支持emoji字符")
    print(f"✅ 🏆 最优模型追踪 - 记录最佳性能")
    print(f"✅ 🐛 修复episode_end和best_reward计算bug")
    print(f"✅ 🆕🆕 路径进度追踪 - 知道走了多远")
    print(f"✅ 🆕🆕 详细终止原因分类 - 6种终止类型")
    print(f"✅ 🆕🆕 智能性能评估系统 - A+/A/B/C/D分级")
    print(f"✅ 🆕🆕 完成率统计 - 激励信息")
    print(f"✅ 🆕🆕 增强错误处理 - 防止训练崩溃")
    print(f"✅ 🆕🆕🆕 倾斜角误差追踪 - 详细记录每个回合的倾斜角误差")
    print(f"✅ 🆕🆕🆕🆕 姿态环模式选择 - 5种内环控制器可选（从第四阶段移植）")
    
    # 🆕🆕🆕 打印姿态环配置信息
    attitude_mode_name_map = {
        'dynamic_lqr': '动态LQR（纯神经网络）',
        'dynamic_lqr_residual': '动态LQR + 残差（完整内环）',
        'static_lqr': '静态单阶段LQR（固定参数）',
        'staged_static_lqr': '静态分阶段LQR（两阶段）',
        'staged_static_lqr_residual': '静态分阶段LQR + 残差'
    }
    
    print(f"\n🔧 姿态环（内环）控制器配置:")
    print(f"   模式: {ATTITUDE_CONTROL_MODE}")
    print(f"   类型: {attitude_mode_name_map.get(ATTITUDE_CONTROL_MODE, '未知')}")
    
    # 显示原有USE_STAGE2_RESIDUAL开关状态（向后兼容）
    if USE_STAGE2_RESIDUAL:
        print(f"   原有开关: USE_STAGE2_RESIDUAL = True (保留，向后兼容)")
    else:
        print(f"   原有开关: USE_STAGE2_RESIDUAL = False (保留，向后兼容)")
    
    print("="*80)
    
    # 检查CUDA
    print(f"\n🔧 设备检测:")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ╔════════════════════════════════════════════════════════════════════════════╗
    # ║          🆕🆕🆕 准备姿态环控制器（新增：支持5种模式）                        ║
    # ╚════════════════════════════════════════════════════════════════════════════╝
    
    print(f"\n[1/8] 准备姿态环控制器...")
    
    # 初始化变量
    agent_lqr = None
    agent_residual = None
    agent_lqr_path = None
    agent_residual_path = None
    static_controller = None  # 🆕 用于存储静态控制器
    
    # 🆕🆕🆕 根据姿态环模式加载/创建控制器
    if ATTITUDE_CONTROL_MODE == 'dynamic_lqr':
        # ========== 模式1: 动态LQR（仅第一阶段）==========
        print(f"   模式: 动态LQR（仅第一阶段）")
        agent_lqr_path = _model_path("stage1_logs_complex", "stage1_agent_lqr_best.zip")
        agent_residual_path = "dummy_non_existent_residual_path.zip"
        if not os.path.exists(agent_lqr_path):
            print(f"   ❌ 错误：找不到 Agent_LQR 模型")
            print(f"      路径: {agent_lqr_path}")
            return
        
        print(f"   ✅ 加载Agent_LQR: {agent_lqr_path}")
        agent_lqr = TD3.load(agent_lqr_path)
        agent_residual = None
        agent_residual_path = None
    
    elif ATTITUDE_CONTROL_MODE == 'dynamic_lqr_residual':
        # ========== 模式2: 动态LQR + 残差（第一+第二阶段）==========
        print(f"   模式: 动态LQR + 残差（第一+第二阶段）")
        agent_lqr_path = _model_path("stage1_logs", "stage1_agent_lqr_best.zip")
        agent_residual_path = _model_path("stage2_logs_dynamic_lqr", "stage2_agent_residual_best.zip")
        # agent_residual_path = _model_path("stage2_logs_dynamic_lqr_ema70", "stage2_agent_residual_best.zip")
        
        if not os.path.exists(agent_lqr_path):
            print(f"   ❌ 错误：找不到 Agent_LQR 模型")
            print(f"      路径: {agent_lqr_path}")
            return
        
        if not os.path.exists(agent_residual_path):
            print(f"   ❌ 错误：找不到 Agent_Residual 模型")
            print(f"      路径: {agent_residual_path}")
            return
        
        print(f"   ✅ 加载Agent_LQR: {agent_lqr_path}")
        print(f"   ✅ 加载Agent_Residual: {agent_residual_path}")
        agent_lqr = TD3.load(agent_lqr_path)
        agent_residual = TD3.load(agent_residual_path)
    
    elif ATTITUDE_CONTROL_MODE == 'static_lqr':
        # ========== 模式3: 静态单阶段LQR（🆕 新增模式）==========
        print(f"   模式: 静态单阶段LQR（新增模式）")
        
        # ✅ 使用真实存在的模型路径（只为通过环境初始化）
        agent_lqr_path = _model_path("stage1_logs", "stage1_agent_lqr_best.zip")
        agent_residual_path = _model_path("stage2", "stage2_agent_residual_best.zip")
        
        # 检查路径是否存在
        if not os.path.exists(agent_lqr_path):
            print(f"   ❌ 错误：找不到临时LQR模型文件（用于环境初始化）")
            print(f"      请确保以下路径存在: {agent_lqr_path}")
            print(f"      或修改为您系统中存在的任意LQR模型路径")
            return
        
        if not os.path.exists(agent_residual_path):
            print(f"   ❌ 错误：找不到临时Residual模型文件（用于环境初始化）")
            print(f"      请确保以下路径存在: {agent_residual_path}")
            print(f"      或修改为您系统中存在的任意Residual模型路径")
            return
        
        print(f"   ✅ 使用临时路径通过环境初始化:")
        print(f"      LQR: {agent_lqr_path}")
        print(f"      Residual: {agent_residual_path}")
        print(f"   💡 这些模型会被加载但立即被静态控制器替换")
        
        # 🆕 创建静态LQR控制器（稍后替换到环境中）
        static_controller = StaticLQRController(
            kp=STATIC_LQR_PARAMS['kp'],
            kd=STATIC_LQR_PARAMS['kd'],
            k=STATIC_LQR_PARAMS['k'],
            kp_range=LQR_PARAM_RANGES['kp_range'],
            kd_range=LQR_PARAM_RANGES['kd_range'],
            k_range=LQR_PARAM_RANGES['k_range']
        )
    
    elif ATTITUDE_CONTROL_MODE == 'staged_static_lqr':
        # ========== 模式4: 静态分阶段LQR（🆕 新增模式）==========
        print(f"   模式: 静态分阶段LQR（新增模式）")
        
        # ✅ 使用真实存在的模型路径
        agent_lqr_path = _model_path("stage1_logs", "stage1_agent_lqr_best.zip")
        agent_residual_path = _model_path("stage2_logs_staged_lqr", "stage2_agent_residual_best.zip")
        
        if not os.path.exists(agent_lqr_path):
            print(f"   ❌ 错误：找不到临时LQR模型文件")
            print(f"      路径: {agent_lqr_path}")
            return
        
        if not os.path.exists(agent_residual_path):
            print(f"   ❌ 错误：找不到临时Residual模型文件")
            print(f"      路径: {agent_residual_path}")
            return
        
        print(f"   ✅ 使用临时路径通过环境初始化:")
        print(f"      LQR: {agent_lqr_path}")
        print(f"      Residual: {agent_residual_path}")
        
        # 🆕 创建静态分阶段LQR控制器
        static_controller = StagedStaticLQRController(
            warmup_kp=STAGED_LQR_PARAMS['stage1_warmup']['kp'],
            warmup_kd=STAGED_LQR_PARAMS['stage1_warmup']['kd'],
            warmup_k=STAGED_LQR_PARAMS['stage1_warmup']['k'],
            full_kp=STAGED_LQR_PARAMS['stage2_full']['kp'],
            full_kd=STAGED_LQR_PARAMS['stage2_full']['kd'],
            full_k=STAGED_LQR_PARAMS['stage2_full']['k'],
            warmup_steps=STAGED_LQR_PARAMS['warmup_steps'],
            kp_range=LQR_PARAM_RANGES['kp_range'],
            kd_range=LQR_PARAM_RANGES['kd_range'],
            k_range=LQR_PARAM_RANGES['k_range']
        )
    
    elif ATTITUDE_CONTROL_MODE == 'staged_static_lqr_residual':
        # ========== 模式5: 静态分阶段LQR + 残差（🆕 新增模式）==========
        print(f"   模式: 静态分阶段LQR + 残差（新增模式）")
        
        # 需要真实的残差模型
        agent_residual_path = _model_path("stage2_logs_staged_lqr", "stage2_agent_residual_best.zip")
        
        if not os.path.exists(agent_residual_path):
            print(f"   ❌ 错误：找不到 Agent_Residual 模型")
            print(f"      路径: {agent_residual_path}")
            return
        
        # 使用临时LQR路径（会被替换）
        agent_lqr_path = _model_path("stage1_logs", "stage1_agent_lqr_best.zip")
        
        if not os.path.exists(agent_lqr_path):
            print(f"   ❌ 错误：找不到临时LQR模型文件")
            return
        
        print(f"   临时LQR路径: {agent_lqr_path}")
        print(f"   ✅ 加载Agent_Residual: {agent_residual_path}")
        
        agent_residual = TD3.load(agent_residual_path)
        
        # 🆕 创建静态分阶段LQR控制器
        static_controller = StagedStaticLQRController(
            warmup_kp=STAGED_LQR_PARAMS['stage1_warmup']['kp'],
            warmup_kd=STAGED_LQR_PARAMS['stage1_warmup']['kd'],
            warmup_k=STAGED_LQR_PARAMS['stage1_warmup']['k'],
            full_kp=STAGED_LQR_PARAMS['stage2_full']['kp'],
            full_kd=STAGED_LQR_PARAMS['stage2_full']['kd'],
            full_k=STAGED_LQR_PARAMS['stage2_full']['k'],
            warmup_steps=STAGED_LQR_PARAMS['warmup_steps'],
            kp_range=LQR_PARAM_RANGES['kp_range'],
            kd_range=LQR_PARAM_RANGES['kd_range'],
            k_range=LQR_PARAM_RANGES['k_range']
        )
    
    else:
        print(f"   ❌ 错误：未知的姿态环模式 '{ATTITUDE_CONTROL_MODE}'")
        return
    
    # 创建环境
    print(f"\n[2/8] 创建训练环境...")
    print(f"   路径类型: {PATH_TYPE}")
    
    # 🆕🆕🆕 根据USE_STAGE2_RESIDUAL开关决定残差路径（保留向后兼容）
    if not USE_STAGE2_RESIDUAL and ATTITUDE_CONTROL_MODE == 'dynamic_lqr_residual':
        print(f"   ⚠️  警告：USE_STAGE2_RESIDUAL=False 与 ATTITUDE_CONTROL_MODE='dynamic_lqr_residual' 冲突")
        print(f"   将使用ATTITUDE_CONTROL_MODE的设置（推荐使用新配置）")
    
    env = Path_tracking_stage3(
        render=1,
        agent_lqr_path=agent_lqr_path,
        agent_residual_path=agent_residual_path,
        path_type=PATH_TYPE
    )
    
    # ✅ 关键步骤：替换姿态环控制器（如果需要）
    if static_controller is not None:
        env.agent_lqr = static_controller
        
        # 对于不使用残差的模式，清空残差控制器
        if ATTITUDE_CONTROL_MODE in ['static_lqr', 'staged_static_lqr']:
            env.agent_residual = None
            print(f"   ✅ 已替换为 {ATTITUDE_CONTROL_MODE} 控制器（无残差）")
        else:
            print(f"   ✅ 已替换为 {ATTITUDE_CONTROL_MODE} 控制器（保留残差）")
    
    env.record_flag = 1
    
    # 日志目录 - 🆕 根据姿态环模式生成不同的目录名
    log_dir = _model_path(f"stage3_{PATH_TYPE}_{EXTRACTOR_TYPE}_{ATTITUDE_CONTROL_MODE}_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"   日志: {log_dir}")
    
    # 🆕🆕 添加allow_early_resets参数
    env = Monitor(env, log_dir, allow_early_resets=True)
    
    # 🆕🆕🆕 倾斜角诊断（可选，保留原有功能）
    if DIAGNOSE_TILT_FIRST:
        print(f"\n{'='*80}")
        print("🔬 运行倾斜角诊断工具...")
        print(f"{'='*80}")
        diagnose_tilt_issue(env.env, num_steps=200)
        
        user_input = input("\n是否继续训练？(y/n): ")
        if user_input.lower() != 'y':
            print("诊断完成，退出训练。")
            env.close()
            return
        print("\n继续训练...\n")
    
    # Stanley基线评估（保留原有功能）
    baseline_stats = None
    if RUN_BASELINE:
        print(f"\n[3/8] 运行Stanley基线评估...")
        
        stanley_controller = StanleyBaselineController(
            k_lateral=STANLEY_K_LATERAL,
            k_course=STANLEY_K_COURSE
        )
        
        baseline_stats = evaluate_stanley_baseline(
            env.env,
            stanley_controller,
            num_episodes=BASELINE_EPISODES
        )
        
        baseline_error_path = os.path.join(log_dir, "stanley_baseline_errors.csv")
        stanley_controller.save_errors(baseline_error_path)
        
        if baseline_stats:
            baseline_stats_path = os.path.join(log_dir, "stanley_baseline_stats.txt")
            with open(baseline_stats_path, 'w', encoding='utf-8') as f:
                f.write("Stanley基线控制器统计信息\n")
                f.write("="*50 + "\n")
                f.write(f"平均横向误差: {baseline_stats['mean_lateral']:.6f} ± {baseline_stats['std_lateral']:.6f} 米\n")
                f.write(f"平均航向误差: {baseline_stats['mean_course']:.6f} ± {baseline_stats['std_course']:.6f} 弧度\n")
                f.write(f"评估Episodes: {baseline_stats['episodes']}\n")
                f.write(f"参数设置:\n")
                f.write(f"  k_lateral = {STANLEY_K_LATERAL}\n")
                f.write(f"  k_course = {STANLEY_K_COURSE}\n")
            print(f"📄 统计信息已保存: {baseline_stats_path}")
    else:
        print(f"\n[3/8] 跳过Stanley基线评估 (RUN_BASELINE=False)")
    
    # 🆕🆕🆕 创建增强版误差追踪器（传入路径类型 + 倾斜角）
    print(f"\n[4/8] 初始化增强版误差追踪器...")
    error_tracker = DetailedErrorTracker(log_dir, PATH_TYPE, baseline_stats)
    print(f"   ✅ 将记录每个Episode的详细误差数据")
    print(f"   ✅ 包含路径进度、终止原因、完成率统计")
    print(f"   ✅ 智能性能评估系统已启用")
    print(f"   ✅ 倾斜角误差追踪已启用")
    
    # 🆕🆕 创建增强版回调
    print(f"\n[5/8] 配置增强版回调...")
    callback = EnhancedTrainingCallback(
        check_freq=env.env.max_step_num,
        log_dir=log_dir,
        error_tracker=error_tracker
    )
    print(f"   ✅ 增强错误处理已启用")
    
    # 动作噪声
    print(f"\n[6/8] 配置动作噪声...")
    n_actions = env.action_space.shape[-1]
    print(f"   动作维度: {n_actions}")
    print(f"   噪声σ: {ACTION_NOISE_SIGMA}")
    
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=ACTION_NOISE_SIGMA * np.ones(n_actions)
    )
    
    # 策略配置
    print(f"\n[7/8] 配置策略网络...")
    
    if USE_CUSTOM_CONFIG:
        policy_kwargs = get_policy_kwargs(custom_config=CUSTOM_CONFIG.copy())
    else:
        policy_kwargs = get_policy_kwargs(extractor_type=EXTRACTOR_TYPE)
    
    # 创建模型
    print(f"\n[8/8] 创建TD3模型...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TD3(
        "MlpPolicy",
        env=env,
        device=device,
        gamma=0.99,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        target_policy_noise=TARGET_POLICY_NOISE,
        target_noise_clip=TARGET_NOISE_CLIP,
        learning_starts=LEARNING_STARTS,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=2,
        seed=42,
        verbose=1
    )
    
    # 配置摘要
    print(f"\n" + "="*80)
    print(f"📋 训练配置摘要")
    print(f"="*80)
    print(f"设备: {device.upper()}")
    print(f"特征提取器: {EXTRACTOR_TYPE.upper()}")
    print(f"路径类型: {PATH_TYPE}")
    print(f"姿态环模式: {ATTITUDE_CONTROL_MODE} ({attitude_mode_name_map[ATTITUDE_CONTROL_MODE]})")
    print(f"倾斜角误差追踪: ✅ 已启用")
    print(f"\n超参数:")
    print(f"  • 学习率: {LEARNING_RATE}")
    print(f"  • 批次大小: {BATCH_SIZE}")
    print(f"  • 缓冲区: {BUFFER_SIZE:,}")
    print(f"  • 总步数: {TOTAL_TIMESTEPS:,}")
    print(f"  • 预计episodes: ~{TOTAL_TIMESTEPS // env.env.max_step_num}")
    print(f"\nTD3参数:")
    print(f"  • 动作噪声: {ACTION_NOISE_SIGMA}")
    print(f"  • 目标噪声: {TARGET_POLICY_NOISE}")
    print(f"  • 初始探索: {LEARNING_STARTS}")
    
    if RUN_BASELINE and baseline_stats:
        print(f"\nStanley基线性能:")
        print(f"  • 平均横向误差: {baseline_stats['mean_lateral']:.4f}m")
        print(f"  • 平均航向误差: {baseline_stats['mean_course']:.4f}rad")
        print(f"  ⚠️  训练目标: 超越基线性能")
    
    print("="*80)
    
    # 训练
    print(f"\n开始训练...")
    print(f"\n💡 训练提示:")
    print(f"   • 每个Episode结束后会显示详细的误差统计")
    print(f"   • 横向误差和航向误差会实时与Stanley基线对比")
    print(f"   • 🆕 显示路径进度和完成率统计")
    print(f"   • 🆕 智能性能评估系统（A+/A/B/C/D分级）")
    print(f"   • 🆕🆕🆕 倾斜角误差详细追踪")
    print(f"   • 🆕🆕🆕🆕 新的姿态环模式: {ATTITUDE_CONTROL_MODE}")
    print(f"   • 所有数据会保存到CSV文件供后续分析")
    print(f"   • 前50-100个episode可能表现较差，请耐心等待")
    print("\n" + "="*80 + "\n")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
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
        final_path = os.path.join(log_dir, "stage3_agent_stanley_final")
        model.save(final_path)
        print(f"\n💾 最终模型: {final_path}")
        
        # 保存完整报告
        error_tracker.save_final_report()
        
        # 🆕🆕🆕 打印最优模型信息（含倾斜角）
        if error_tracker.best_episode_summary:
            print(f"\n{'='*80}")
            print(f"🏆 最优模型信息（训练过程中表现最好的Episode）")
            print(f"   姿态环模式: {ATTITUDE_CONTROL_MODE}")
            print(f"{'='*80}")
            best = error_tracker.best_episode_summary
            print(f"Episode编号: {best['episode']}")
            print(f"路径进度: {best['progress_percent']:.1f}%")
            print(f"完成状态: {'✅' if best['completed'] else '❌'}")
            print(f"横向误差: {best['avg_lateral']:.6f} m (±{best['std_lateral']:.6f})")
            print(f"航向误差: {best['avg_course']:.6f} rad (±{best['std_course']:.6f})")
            print(f"倾斜角误差: {best['avg_tilt']:.6f} rad ({np.rad2deg(best['avg_tilt']):.2f}°) (±{best['std_tilt']:.6f})")
            print(f"奖励: {best['reward']:.2f}")
            
            if baseline_stats:
                lateral_improvement = (1 - best['avg_lateral'] / baseline_stats['mean_lateral']) * 100
                course_improvement = (1 - best['avg_course'] / baseline_stats['mean_course']) * 100
                
                print(f"\n相较Stanley基线:")
                print(f"  基线横向误差: {baseline_stats['mean_lateral']:.6f} m")
                print(f"  最优横向误差: {best['avg_lateral']:.6f} m")
                print(f"  横向误差提升: {lateral_improvement:+.2f}%")
                
                print(f"\n  基线航向误差: {baseline_stats['mean_course']:.6f} rad")
                print(f"  最优航向误差: {best['avg_course']:.6f} rad")
                print(f"  航向误差提升: {course_improvement:+.2f}%")
                
                if lateral_improvement > 20 and course_improvement > 20:
                    print(f"\n  🎉🎉🎉 卓越！RL控制器大幅超越Stanley基线！")
                elif lateral_improvement > 10 and course_improvement > 10:
                    print(f"\n  ✅✅ 优秀！RL控制器显著优于基线！")
                elif lateral_improvement > 5 and course_improvement > 5:
                    print(f"\n  ✅ 良好！RL控制器有效提升性能")
                elif lateral_improvement > 0 or course_improvement > 0:
                    print(f"\n  ☑️  一般，在某些方面略有提升")
                else:
                    print(f"\n  ⚠️  需要改进，未能超越基线")
            print(f"{'='*80}\n")
        
        # 统计
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(y) > 0:
            final_reward = np.mean(y[-20:]) if len(y) >= 20 else np.mean(y)
            
            # 🐛 修复：检查数组长度避免空数组错误
            if len(y) >= 20:
                best_reward = np.max([np.mean(y[i:i+20]) for i in range(len(y)-19)])
            else:
                best_reward = np.mean(y) if len(y) > 0 else 0
            
            print(f"\n📊 训练统计:")
            print(f"   Episodes: {len(y)}")
            print(f"   最终20集平均: {final_reward:.2f}")
            print(f"   历史最佳20集: {best_reward:.2f}")
            
            # 性能评估
            recent_stats = error_tracker.get_recent_stats(20)
            if recent_stats:
                print(f"\n最近20集性能:")
                print(f"   完成率: {recent_stats['completion_rate']:.1f}%")
                print(f"   平均进度: {recent_stats['avg_progress']:.1f}%")
                print(f"   横向误差: {recent_stats['avg_lateral']:.4f}m")
                print(f"   航向误差: {recent_stats['avg_course']:.4f}rad")
                print(f"   倾斜角误差: {recent_stats['avg_tilt']:.4f}rad ({np.rad2deg(recent_stats['avg_tilt']):.2f}°)")  # 🆕🆕🆕
                
                if baseline_stats:
                    lateral_improvement = (1 - recent_stats['avg_lateral']/baseline_stats['mean_lateral']) * 100
                    course_improvement = (1 - recent_stats['avg_course']/baseline_stats['mean_course']) * 100
                    
                    print(f"\n与Stanley基线对比:")
                    print(f"   横向误差提升: {lateral_improvement:+.2f}%")
                    print(f"   航向误差提升: {course_improvement:+.2f}%")
                    
                    if lateral_improvement > 0 and course_improvement > 0:
                        print(f"\n   ✅✅ RL控制器成功超越Stanley基线")
                    elif lateral_improvement > 0 or course_improvement > 0:
                        print(f"\n   ☑️  RL控制器在某些方面优于基线")
                    else:
                        print(f"\n   ⚠️  RL控制器未能超越基线，建议调整参数")
            
            if final_reward > -20:
                print(f"\n⭐⭐⭐⭐⭐ 优秀！路径跟踪性能很好")
            elif final_reward > -40:
                print(f"\n⭐⭐⭐⭐ 良好！性能可以接受")
            elif final_reward > -60:
                print(f"\n⭐⭐⭐ 中等，建议继续训练或调整参数")
            else:
                print(f"\n⭐⭐ 需要改进，检查内环性能和奖励函数")
        
        print(f"\n" + "="*80)
        print("📁 生成的文件:")
        print(f"   • {os.path.join(log_dir, 'rl_training_errors.csv')} - Episode摘要（增强版+倾斜角）")
        print(f"   • {os.path.join(log_dir, 'rl_training_report.txt')} - 完整报告（含最优模型详情+倾斜角）")
        print(f"   • {os.path.join(log_dir, 'best_model_info.txt')} - 最优模型信息（基于奖励）")
        print(f"   • {os.path.join(log_dir, 'stanley_baseline_errors.csv')} - Stanley基线数据")
        print(f"   • {callback.save_path}.zip - 最佳模型（基于奖励）")
        print(f"   • {final_path}.zip - 最终模型")
        print("="*80 + "\n")
        
        print("✅ 第三阶段训练完成！")
        print("📁 三阶段层次化强化学习训练全部完成")
        print("🆕🆕🆕 倾斜角误差数据已完整记录")
        print(f"🆕🆕🆕🆕 姿态环模式: {ATTITUDE_CONTROL_MODE} ({attitude_mode_name_map[ATTITUDE_CONTROL_MODE]})")
        print("="*80 + "\n")
        
        env.close()

if __name__ == "__main__":
    main()
