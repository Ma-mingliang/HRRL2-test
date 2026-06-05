# HRRL2 使用说明

## 1. 项目当前范围

这个目录当前只保留两部分主线：

- 第 1 部分：`LQR.py`
  - 用强化学习从零训练纯 RL 车把角平衡控制器。
- 第 3 部分：`stanley.py`
  - 用强化学习训练自适应 `Stanley` 外环。
  - 内环本次未改，仍固定调用旧的自适应 `LQR` 模型。

已经移除当前主线中的第 2、4 部分环境逻辑。历史日志目录还在 `model/` 下保留，但它们不是当前代码主线。

## 2. 关键文件

- `env.py`
  - 环境定义。
  - 当前主用环境类：
    - `Attitude_control_stage1`
    - `Path_tracking_stage3`
- `LQR.py`
  - 第 1 部分训练入口。
- `stanley.py`
  - 第 3 部分训练入口。
- `3D/`
  - 路径 mesh、地形、车辆 URDF 等资源。
- `model/`
  - 训练输出目录。
  - 已存在旧 LQR 内环可用模型：
    - `model/stage1_logs/stage1_agent_lqr_best.zip`
- `verify_full_logic_smoke.py`
  - 主线烟雾测试。
- `verify_single_turn_assets.py`
  - 路径资源加载检查。

## 3. 依赖

至少需要这些 Python 包：

- `gymnasium`
- `torch`
- `stable_baselines3`
- `pybullet`
- `pybullet_data`
- `numpy`
- `pandas`

如果你的环境里还没有这些包，需要先自行安装。

## 4. 推荐使用顺序

### 4.1 先跑验证

在项目根目录执行：

```powershell
python .\verify_full_logic_smoke.py
python .\verify_single_turn_assets.py
```

作用：

- `verify_full_logic_smoke.py`
  - 检查第 1 部分和第 3 部分环境能否创建并 `step`。
- `verify_single_turn_assets.py`
  - 检查路径资源和 URDF 引用是否正常。

### 4.2 训练第 1 部分

```powershell
python .\LQR.py
```

说明：

- 第 1 部分默认会直接开始训练。
- 第 1 部分现在直接输出最终车把目标角，不叠加 `LQR` 基准控制量。
- 第 1 部分默认 `RENDER = 1`，会打开 GUI。
- 如果是无图形环境，先把 `LQR.py` 里的 `RENDER` 改成 `0`。

训练输出默认保存在：

- `model/stage1_pure_rl_logs/`

关键输出文件：

- `model/stage1_pure_rl_logs/stage1_agent_pure_rl_balance_best.zip`
- `model/stage1_pure_rl_logs/stage1_agent_pure_rl_balance_final.zip`

### 4.3 训练第 3 部分

```powershell
python .\stanley.py
```

说明：

- 第 3 部分本次未适配纯 RL 车把角模型，仍读取旧 LQR 最优模型：
  - `model/stage1_logs/stage1_agent_lqr_best.zip`
- 如果这个文件不存在，需要恢复或重新训练旧 LQR 内环模型。

当前默认配置：

- 默认路径：`complex`
- 默认渲染：`RENDER = 0`
- 默认内环：旧的自适应 `LQR`

训练输出默认保存在：

- `model/stage3_complex_mlp_dynamic_lqr_seed42_hreset_legacy_logs/`
  - 实际目录名会由 `PATH_TYPE`、`SEED`、`HEADING_OFFSET_RESET_MODE` 等配置决定。

## 5. 如何切换配置

### 5.1 切换第 3 部分路径

在 `stanley.py` 里修改：

- `PATH_TYPE`

可选值：

- `"s_line"`
- `"complex"`
- `"single_turn_90"`
- `"single_turn_wide"`
- `"single_turn_exit"`

### 5.2 调整第 3 部分训练参数

在 `stanley.py` 里：

- 通用默认参数在文件顶部。
- 路径定制参数在 `PATH_TRAINING_OVERRIDES`。

目前 `complex` 已单独配置了更适合完赛目标的训练参数，包括：

- 更长总步数
- 更高探索噪声
- 更长 warmup
- 更多 baseline 评估局数

### 5.3 调整第 3 部分终点和奖励

第 3 部分环境逻辑都在 `env.py`：

- 路径配置：`_get_path_tracking_config`
- complex 课程与 reset：`_get_complex_reset_profile`
- complex 进度定义：`_compute_complex_progress_distance`
- complex 奖励：`_calculate_complex_reward_core`

## 6. 当前代码设计要点

### 6.1 第 3 部分只保留一种主线

当前 `stanley.py` 不再使用旧的多模式兼容逻辑，主线只有一种：

- 外环：自适应 `Stanley`
- 内环：旧的自适应 `LQR`

### 6.2 complex 路径已经做过目标对齐

为了让训练目标和任务目标一致，当前代码已经对齐了：

- 完成区域
- 几何进度定义
- reset 课程难度
- 完成奖励与失败惩罚

所以如果后续要继续调 `complex`，建议优先在现有逻辑上微调，不要再恢复旧的 `y` 进度近似写法。

## 7. 常见问题

### 7.1 第 3 部分报找不到第 1 部分模型

先确认这个文件存在：

- `model/stage1_logs/stage1_agent_lqr_best.zip`

如果不存在，需要恢复或重新训练旧 LQR 内环模型。本次第一阶段新训练出的 pure RL 模型不会被第三阶段直接使用。


### 7.2 无图形环境运行卡住

把下面两个文件里的 `RENDER` 改成 `0`：

- `LQR.py`
- `stanley.py`

当前 `stanley.py` 已默认是 `0`，`LQR.py` 仍默认 GUI。

### 7.3 `model/` 下还有 stage2/stage4 相关历史目录

这是历史训练结果保留，不代表当前主线仍依赖这些部分。当前主线只看：

- 第 1 部分 pure RL 平衡控制输出
- 第 3 部分输出和旧 LQR 内环模型

## 8. 给接手者的最短流程

如果只是想确认这个项目还能跑，按下面顺序执行即可：

```powershell
python .\verify_full_logic_smoke.py
python .\verify_single_turn_assets.py
python .\LQR.py
python .\stanley.py
```

如果只是想直接看第 3 部分训练，且旧 LQR 内环模型已存在，可以跳过 `LQR.py`。
