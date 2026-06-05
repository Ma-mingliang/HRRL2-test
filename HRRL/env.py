"""
自行车自平衡与路径跟踪控制系统 - 完整版
修复：数据记录长度不一致问题

包含：
- 第一阶段：自适应LQR控制器（修复版）
- 第二阶段：内环残差修正（修复版）
- 第三阶段：自适应Stanley控制器
- 第四阶段：路径跟踪残差修正
- 完整训练程序
"""

import random
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

ASSET_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "3D"),
    os.path.join(WORKSPACE_ROOT, "3D"),
]
MODEL_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "model"),
    os.path.join(WORKSPACE_ROOT, "model"),
]

ASSET_DIR = next((path for path in ASSET_DIR_CANDIDATES if os.path.isdir(path)), ASSET_DIR_CANDIDATES[0])
MODEL_DIR = next((path for path in MODEL_DIR_CANDIDATES if os.path.isdir(path)), MODEL_DIR_CANDIDATES[0])


def _asset_path(*parts):
    return os.path.join(ASSET_DIR, *parts)


def _model_path(*parts):
    return os.path.join(MODEL_DIR, *parts)

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


# ==================== 第一阶段：自适应LQR控制器训练环境 ====================

class Attitude_control_stage1(gym.Env):
    """第一阶段：训练自适应LQR控制器 - 修复版"""

    def __init__(self, render: bool = False, 
                 failure_penalty: float = -10.0,
                 early_termination_penalty: float = -20.0,
                 action_repeat: int = 1):
        self._render = render
        
        # 惩罚参数
        self.FAILURE_PENALTY = failure_penalty
        self.EARLY_TERMINATION_PENALTY = early_termination_penalty
        self.action_repeat = action_repeat
        
        print(f"[Stage 1 - 修复版] 初始化环境:")
        print(f"  - 失败惩罚: {self.FAILURE_PENALTY}")
        print(f"  - 早期失败额外惩罚: {self.EARLY_TERMINATION_PENALTY}")
        print(f"  - 动作重复: {self.action_repeat}")
        
        # 动作空间和状态空间
        self.action_space = spaces.Box(
            low=np.array([-1., -1., -1.]),
            high=np.array([1., 1., 1.]),
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
        self.min_success_steps = 300
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
        self.target_handle_angle_csv = []
        self.handle_angle_csv = []
        self.kp_csv = []
        self.kd_csv = []
        self.k_csv = []

        # 加载环境
        p.loadURDF(r"plane.urdf", globalScaling=15)

        # p.loadURDF(_asset_path("generated_terrain_lab.urdf"), globalScaling=2)
        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = _asset_path("bike", "urdf", "bike.urdf")
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

    def _map_action_to_gains(self, action):
        """将归一化的动作映射到实际的LQR增益范围"""
        kp = (action[0] + 1) / 2 * (25 - 5) + 5
        kd = (action[1] + 1) / 2 * (5 - 1) + 1
        k = (action[2] + 1) / 2 * (20 - 0) + 0
        
        return kp, kd, k

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

    def __calculate_reward(self, state_last, state):
        """改进的奖励函数"""
        state_last_raw = self.__observation_reduction(state_last)
        state_raw = self.__observation_reduction(state)
        
        current_error = abs(state_raw[0])
        angular_velocity = abs(state_raw[2])
        
        # 1. 核心跟踪奖励
        max_penalty = 2.0
        tracking_reward = -min(current_error**2, max_penalty)
        
        # 2. 高精度奖励
        bonus_reward = 0.0
        if current_error < 0.005:
            bonus_reward = 1.0
        elif current_error < 0.01:
            bonus_reward = 0.5
        elif current_error < 0.02:
            bonus_reward = 0.2
        
        # 3. 平顺性惩罚
        smoothness_penalty = -0.05 * angular_velocity
        
        # 4. 改进奖励
        improvement_reward = 0.0
        error_reduction = abs(state_last_raw[0]) - current_error
        if error_reduction > 0:
            improvement_reward = 0.3 * error_reduction
        
        reward = tracking_reward + bonus_reward + smoothness_penalty + improvement_reward
        
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
        self.kp_csv = []
        self.kd_csv = []
        self.k_csv = []

        startPos = [0, 0, 0.75]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        
        p.resetBasePositionAndOrientation(self.bike, startPos, startOrientation)
        p.resetJointState(self.bike, 0, 0, 0)
        p.resetJointState(self.bike, 1, 0, 0)
        p.resetJointState(self.bike, 2, 0, 0)
        
        p.stepSimulation()

        return self.__get_observation(self.target_theta), {}

    def step(self, action):
        """【修复版】执行带动作重复的动作步 - 增益数据现在每个时间步都记录"""
        cumulative_reward = 0.0
        terminated = False
        truncated = False
        
        # 映射动作到增益
        kp, kd, k = self._map_action_to_gains(action)
        
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
            
            # 【修复】每个时间步都记录增益，确保与姿态数据长度一致
            if self.record_flag:
                self.kp_csv.append(kp)
                self.kd_csv.append(kd)
                self.k_csv.append(k)
            
            #目标倾斜角的动态变化
            # if (self.step_num + repeat_idx) % 100 == 0:
            #     sigma = min(0.3, (1.08**(self.epoch_num)) * 0.01)
            #     self.target_theta = np.random.normal(loc=0.0, scale=sigma, size=None)
            #     if abs(self.target_theta) >= math.pi/12:
            #         self.target_theta = random.uniform(-math.pi/12, math.pi/12)
            
            # 前100步稳定启动
            if self.step_num + repeat_idx < 100:
                k_used = 0
                target_theta_used = 0
            else:
                k_used = k
                target_theta_used = self.target_theta
            
            # LQR控制律
            pid_control = kp * state_last[1] * 1.57 + \
                          kd * state_last[2] * 10 - \
                          target_theta_used * k_used
            
            pid_control = math.atan((pid_control * 2.25 / (4**2)))
            pid_control = np.clip(pid_control, -0.785, 0.785)
            
            # 执行电机控制
            p.setJointMotorControl2(self.bike, 0, p.VELOCITY_CONTROL, 
                                    targetVelocity=-10, force=20)
            p.setJointMotorControl2(self.bike, 2, p.VELOCITY_CONTROL, 
                                    targetVelocity=-10, force=20)
            p.setJointMotorControl2(self.bike, 1, p.POSITION_CONTROL,
                                    targetPosition=pid_control, force=200, positionGain=0.3)
            
            p.stepSimulation()
            
            state = self.__get_observation(self.target_theta)
            reward = self.__calculate_reward(state_last, state)
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
        
        # 【修复】数据保存逻辑 - 现在所有数据长度一致，不需要repeat
        if terminated or truncated:
            if self.step_num > 0:
                self.angle_error_csv.append(self.angle_error / self.step_num)
                np.savetxt(_model_path("stage1_angle_error.csv"), 
                          np.array([self.angle_error_csv]).T, delimiter=',')
            
            if self.record_flag == 1:
                # 确保所有数据长度一致
                min_len = min(
                    len(self.target_theta_csv),
                    len(self.theta0_csv),
                    len(self.handle_angle_csv),
                    len(self.kp_csv),
                    len(self.kd_csv),
                    len(self.k_csv)
                )
                
                print(f"[Stage 1] 保存数据: {min_len} 个时间步")
                
                np.savetxt(_model_path("stage1_data.csv"), np.array([
                    self.target_theta_csv[:min_len],
                    self.theta0_csv[:min_len],
                    self.handle_angle_csv[:min_len],
                    self.kp_csv[:min_len],
                    self.kd_csv[:min_len],
                    self.k_csv[:min_len]
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
                agent_residual_path: str = None,
                path_type: str = "s_line",
                action_repeat: int = 1):
        """
        初始化环境
        
        参数:
            render: 是否可视化
            agent_lqr_path: LQR模型路径
            agent_residual_path: 残差模型路径
            path_type: 路径类型 ("s_line" 或 "complex")
            action_repeat: 动作重复次数
        """
        self._render = render
        self.path_type = path_type
        self.action_repeat = action_repeat
        
        # ✅ 路径特定参数
        if path_type == "s_line":
            self.max_step_num = 255 * action_repeat
            self.early_failure_threshold = 20
            # ✅ S型完成判定：y >= 61.5 且 -30 < x < 30
            self.completion_y = 61.5
            self.completion_x_min = -30
            self.completion_x_max = 30
            self.baseline_lateral = 0.50  # 静态Stanley基线
            self.baseline_course = 0.11
            print(f"[Stage 3] S型路径参数:")
            print(f"  完成条件: y>={self.completion_y} and {self.completion_x_min}<x<{self.completion_x_max}")
        else:
            self.max_step_num = 2500 * action_repeat
            self.early_failure_threshold = 50
            # ✅ 复杂路径完成判定：y > 55 且 0 < x < 28
            self.completion_y = 55.0
            self.completion_x_min = 0.0
            self.completion_x_max = 28.0
            self.baseline_lateral = 0.50
            self.baseline_course = 0.11
            print(f"[Stage 3] 复杂路径参数:")
            print(f"  完成条件: y>{self.completion_y} and {self.completion_x_min}<x<{self.completion_x_max}")
        
        print(f"  最大步数: {self.max_step_num} (={self.max_step_num//action_repeat}个动作)")
        print(f"  早期失败阈值: {self.early_failure_threshold}个动作")
        print(f"  基线: Lateral={self.baseline_lateral}m, Course={self.baseline_course}rad")

        # 验证模型
        #if agent_lqr_path is None or agent_residual_path is None:
        if agent_lqr_path is None :    
            raise ValueError("必须提供模型路径")
        
        if os.path.exists(agent_lqr_path):
            self.agent_lqr = TD3.load(agent_lqr_path)
            print(f"  ✅ Agent_LQR 已加载")
        else:
            self.agent_lqr = None
            print(f"  ⏭️  Agent_LQR 外部提供")
        #修改处
        # if os.path.exists(agent_residual_path):
        #     self.agent_residual = TD3.load(agent_residual_path)
        #     print(f"  ✅ Agent_Residual 已加载")
        # else:
        #     self.agent_residual = None
        #     print(f"  ⏭️  Agent_Residual 外部提供")
        if agent_residual_path is not None and os.path.exists(agent_residual_path):
            # --- 修改结束 ---
            self.agent_residual = TD3.load(agent_residual_path)
            print("  ✅ Agent_Residual 已加载")
        else:
            self.agent_residual = None
            # 根据 agent_residual_path 是否为 None 给出更精确的提示
            if agent_residual_path is None:
                print("  ℹ️  Agent_Residual 未提供 (例如 dynamic_lqr 模式)")
            else:
                print(f"  ⚠️  Agent_Residual 路径无效或文件不存在: {agent_residual_path}")
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
            startPos = [0, 0, 0.7]
            print("  ✅ S型路径已加载")
        else:
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=_asset_path("complex_path.obj"),
                rgbaColor=[255/255, 50/255, 0/255, 0.5],
                specularColor=[0.4, 0.4, 0.8],
                visualFramePosition=[-1, -1, 0],
                meshScale=[0.001, 0.001, 0.001]
            )
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id)
            # p.loadURDF(_asset_path("generated_terrain_lab.urdf"), globalScaling=2)
            # p.loadURDF(_asset_path("generated_terrain_lab.urdf"), globalScaling=5)
            p.loadURDF("plane.urdf", globalScaling=2)
            startPos = [0, 0, 0.78]
            print("  ✅ 复杂路径已加载")
        
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = _asset_path("bike", "urdf", "bike.urdf")
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
            
            # 根据路径类型计算进度
            if self.path_type == "s_line":
                # S型路径：主要沿y方向前进
                forward_progress = forward_wheel_point_y - getattr(self, 'last_y_position', 0)
                self.last_y_position = forward_wheel_point_y
            else:
                # 复杂路径：综合考虑x和y方向
                # 可以根据当前路径段优化进度计算
                if hasattr(self, 'current_path_segment'):
                    if self.current_path_segment in [1, 2]:  # 段1和2主要沿x方向
                        forward_progress = forward_wheel_point_x - self.last_x_position
                    elif self.current_path_segment in [3, 4]:  # 段3和4主要沿y方向
                        forward_progress = forward_wheel_point_y - getattr(self, 'last_y_position', 0)
                        self.last_y_position = forward_wheel_point_y
                    else:  # 段5和6：综合方向
                        dx = forward_wheel_point_x - self.last_x_position
                        dy = forward_wheel_point_y - getattr(self, 'last_y_position', 0)
                        forward_progress = math.sqrt(dx**2 + dy**2) * 0.5  # 降低权重
                        self.last_y_position = forward_wheel_point_y
                else:
                    forward_progress = forward_wheel_point_x - self.last_x_position
            
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

        # ✅ 根据路径类型设置初始位置
        if self.path_type == "s_line":
            startPos = [0, 0, 0.7]
        else:
            sigma = min(2.5, (1.1**(self.epoch_num)) * 0.01)
            y = np.clip(np.random.normal(loc=0.0, scale=sigma, size=None), -4.5, 4.5)
            startPos = [0, y, 0.78]
        
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
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

            #修改处
            # residual_action = self.agent_residual.predict(attitude_state, deterministic=True)[0]
            # residual = residual_action[0] * 0.1
            # final_handle_angle = pid_control + residual
            
            if self.agent_residual is not None:
                residual_action = self.agent_residual.predict(attitude_state, deterministic=True)[0]
                residual = residual_action[0] * 0.1
                final_handle_angle = pid_control + residual
            else:
                # 如果 agent_residual 为 None（例如 dynamic_lqr 模式），则不使用残差
                final_handle_angle = pid_control
                residual = 0  # （可选）记录残差为 0

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
        if self.path_type == "s_line":
            progress_percent = min(100.0, (y_s / 60.0) * 100)
            path_completed = (y_s >= self.completion_y and 
                            self.completion_x_min < x_s < self.completion_x_max)
        else:
            if y_s < 5:
                progress_percent = (y_s / 5) * 20
            elif y_s < 34:
                progress_percent = 30 + ((y_s - 5) / 30) * 40
            else:
                progress_percent = 90 + min(10, (y_s - 55) / 5 * 10)
            
            path_completed = (y_s > self.completion_y and 
                            self.completion_x_min < x_s < self.completion_x_max)
        
        # ========== ✅ 2. 完成判定（最高优先级）==========
        if path_completed:
            termination_reason = "task_complete"
            truncated = True  # 完成用truncated
            
            # 计算平均误差
            if len(self.later_error_csv) > 0:
                mean_lateral = sum(map(abs, self.later_error_csv)) / len(self.later_error_csv)
            else:
                mean_lateral = 1.0
            
            # 基础完成奖励
            base_completion_reward = 500.0
            
            # 质量加成
            baseline = self.baseline_lateral
            
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
            
            # 速度加成
            expected_actions = self.max_step_num // self.action_repeat
            
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
                    base_early_penalty = -400.0
                    
                    if self.action_num < 5:
                        severity_multiplier = 1.5
                    elif self.action_num < 10:
                        severity_multiplier = 1.2
                    else:
                        severity_multiplier = 1.0
                    
                    early_penalty = base_early_penalty * (1 - early_ratio) * severity_multiplier
                    cumulative_reward += early_penalty
                    
                    print(f"\n❌ 早期失败！[{self.path_type}] 动作={self.action_num}/{self.early_failure_threshold}, "
                        f"惩罚={early_penalty:.1f}, 原因={termination_reason}")
                
                # 中期失败
                elif self.action_num < 150:
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



