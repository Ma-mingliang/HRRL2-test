# HRRL2 Stage 1 Reward Optimization Changelog

Started: 2026-06-06T03:43:27.520330+00:00
Runtime: 5 hours
Timesteps per iteration: 50000

## v0 — Baseline [ACCEPTED]

- Mean Reward: 657.43
- Episodes: 129
- Training Time: 350s

## v1 — Residual action magnitude penalty [ACCEPTED]

- **Category**: F_residual_aware_reward
- **ID**: F1_residual_action_penalty
- **Description**: Penalize large steering angles proportional to squared magnitude (lambda=0.05)
- **Mean Reward**: 677.07 (baseline: 657.43, change: -3.0%)
- **Episodes**: 112
- **Training Time**: 354s
- **Timestamp**: 2026-06-06T03:49:34.371418+00:00

## v2 — Residual action smoothness penalty [REJECTED]

- **Category**: F_residual_aware_reward
- **ID**: F2_residual_smoothness
- **Description**: Penalize change in steering angle between steps (smoothness) + squared action penalty
- **Mean Reward**: 638.63 (baseline: 677.07, change: +5.7%)
- **Episodes**: 104
- **Training Time**: 324s
- **Timestamp**: 2026-06-06T03:54:58.852471+00:00

## v3 — Adaptive residual lambda (decreases as error shrinks) [REJECTED]

- **Category**: F_residual_aware_reward
- **ID**: F3_adaptive_residual_lambda
- **Description**: Residual penalty lambda decreases as tracking improves, encouraging exploration early
- **Mean Reward**: 668.36 (baseline: 677.07, change: +1.3%)
- **Episodes**: 121
- **Training Time**: 325s
- **Timestamp**: 2026-06-06T04:00:24.896872+00:00

## v4 — Potential-based reward shaping [REJECTED]

- **Category**: A_potential_based_reward
- **ID**: A1_potential_shaping
- **Description**: Add gamma*Phi(s') - Phi(s) shaping where Phi = -error^2 (policy invariant)
- **Mean Reward**: 648.61 (baseline: 677.07, change: +4.2%)
- **Episodes**: 126
- **Training Time**: 317s
- **Timestamp**: 2026-06-06T04:05:43.207249+00:00

## v5 — Potential shaping with velocity component [REJECTED]

- **Category**: A_potential_based_reward
- **ID**: A2_potential_velocity
- **Description**: Potential function includes both error and velocity: Phi = -(error^2 + 0.1*omega^2)
- **Mean Reward**: 670.22 (baseline: 677.07, change: +1.0%)
- **Episodes**: 118
- **Training Time**: 317s
- **Timestamp**: 2026-06-06T04:11:01.128578+00:00

## v6 — Enhanced angular velocity safety penalty [REJECTED]

- **Category**: B_safety_constraint_reward
- **ID**: B1_angular_velocity_penalty
- **Description**: Quadratic angular velocity penalty (instead of linear) to strongly discourage oscillation
- **Mean Reward**: 652.31 (baseline: 677.07, change: +3.7%)
- **Episodes**: 63
- **Training Time**: 318s
- **Timestamp**: 2026-06-06T04:16:19.571778+00:00

## v7 — Tilt angle safety barrier [REJECTED]

- **Category**: B_safety_constraint_reward
- **ID**: B2_tilt_safety_barrier
- **Description**: Exponential penalty that increases sharply as tilt approaches failure threshold (pi/3)
- **Mean Reward**: 655.44 (baseline: 677.07, change: +3.2%)
- **Episodes**: 97
- **Training Time**: 352s
- **Timestamp**: 2026-06-06T04:22:12.119127+00:00

## v8 — Combined safety: barrier + quadratic velocity [REJECTED]

- **Category**: B_safety_constraint_reward
- **ID**: B3_combined_safety
- **Description**: Tilt barrier + quadratic angular velocity penalty + action smoothness
- **Mean Reward**: 659.00 (baseline: 677.07, change: +2.7%)
- **Episodes**: 74
- **Training Time**: 320s
- **Timestamp**: 2026-06-06T04:27:32.502519+00:00

## v9 — Hierarchical error-stage reward [ACCEPTED]

- **Category**: E_hierarchical_reward
- **ID**: E1_hierarchical_error_stages
- **Description**: Different reward scales for coarse (>0.05), medium (0.02-0.05), fine (<0.02) error stages
- **Mean Reward**: 710.23 (baseline: 677.07, change: -4.9%)
- **Episodes**: 93
- **Training Time**: 316s
- **Timestamp**: 2026-06-06T04:32:52.711031+00:00

## v10 — Progressive difficulty curriculum [ACCEPTED]

- **Category**: E_hierarchical_reward
- **ID**: E2_progressive_difficulty
- **Description**: Reward scales increase with episode count (curriculum), early episodes are more forgiving
- **Mean Reward**: 999.39 (baseline: 710.23, change: -40.7%)
- **Episodes**: 111
- **Training Time**: 313s
- **Timestamp**: 2026-06-06T04:38:09.332415+00:00

## v11 — Subgoal milestone rewards [REJECTED]

- **Category**: C_curriculum_subgoal_reward
- **ID**: C1_subgoal_milestones
- **Description**: Extra bonus for sustained precision (consecutive steps under threshold)
- **Mean Reward**: 839.41 (baseline: 999.39, change: +16.0%)
- **Episodes**: 129
- **Training Time**: 319s
- **Timestamp**: 2026-06-06T04:43:28.918684+00:00

## v12 — Adaptive component weights [REJECTED]

- **Category**: D_adaptive_dynamic_reward
- **ID**: D1_adaptive_weight_combination
- **Description**: Tracking weight increases, smoothness weight decreases as training progresses
- **Mean Reward**: 638.13 (baseline: 999.39, change: +36.1%)
- **Episodes**: 104
- **Training Time**: 317s
- **Timestamp**: 2026-06-06T04:48:46.128084+00:00

