# 当前项目奖励函数与比重（源码对齐）

## 1. 统计范围
- 任务注册来源：`legged_gym/legged_gym/envs/__init__.py`
- 奖励实现来源：
  - `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
  - `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`
- 权重来源：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py` 的各配置类 `rewards.scales`

## 2. 奖励总式与生效权重
总奖励：

\[
r_t=\sum_i \left(f_i \cdot w_i\right)
\]

其中：
- `f_i`：对应 `_reward_<name>()`
- `w_i`：配置中的 `rewards.scales.<name>`

运行时会对每个非零 scale 乘 `dt`（`_prepare_reward_function` in `legged_gym/legged_gym/envs/base/legged_robot.py`），本项目配置下：
- `sim.dt = 0.002`
- `control.decimation = 10`
- 训练步长 `dt = 0.02`
- 生效权重 `w_eff = w_raw * 0.02`

备注：
- 当前这些配置里 `regularization_names` 为空，因此 `regularization_scale` 不会额外作用到任何项。

## 3. 任务与权重模板映射
- `g1_priv_mimic` -> `G1MimicPrivCfg.rewards.scales`
- `g1_stu_rl` -> `G1MimicStuRLCfg.rewards.scales`
- `g1_stu_rl_cmg_slow / medium / fast / fast_low / fast_low_cmdswitch / fast_narrow` -> `G1MimicStuRLCMGBaseCfg.rewards.scales`（各子类奖励权重一致）
- `g1_cmg_slow / medium / fast` -> `G1MimicCMGBaseCfg.rewards.scales`（三者奖励权重一致）

说明：`G1MimicStuRLCMGBaseCfg.rewards.scales` 与 `G1MimicCMGBaseCfg.rewards.scales` 在当前代码中数值相同。

## 4. 所有奖励函数与比重（含中文解释）
表中 `-` 表示该奖励函数已实现，但该任务配置未设置 scale（不参与总奖励）。

| reward function | 中文解释 | g1_priv_mimic | g1_stu_rl | g1_stu_rl_cmg_* / g1_cmg_* |
|---|---|---:|---:|---:|
| `tracking_joint_dof` | 跟踪参考关节角，姿态误差越小奖励越高 | 0.6 | 0.65 | 0.6 |
| `tracking_joint_vel` | 跟踪参考关节角速度，动作节奏越接近参考越好 | 0.2 | 0.22 | 0.2 |
| `tracking_root_pose` | 跟踪躯干根部位置与朝向 | 0.6 | 0.6 | 0.2 |
| `tracking_root_vel` | 跟踪躯干根部线速度与角速度 | 1.2 | 1.3 | 0.8 |
| `tracking_keybody_pos` | 跟踪关键身体点（脚/膝/手/肘/头等）位置 | 2.4 | 2.6 | 2.0 |
| `tracking_keybody_pos_upper` | 仅跟踪上半身关键点（手、肘、头） | - | - | 0.3 |
| `tracking_cmd_vel` | 跟踪 CMG 命令线速度 `vx, vy` | - | - | 1.5 |
| `tracking_cmd_yaw` | 跟踪 CMG 命令偏航角速度 `yaw_rate` | - | - | 1.0 |
| `action_symmetry` | 鼓励左右肢体动作近似镜像对称 | - | - | 0.1 |
| `feet_slip` | 惩罚脚接触地面时的水平滑移 | -0.1 | -0.055 | -0.1 |
| `feet_contact_forces` | 惩罚足端接触力过大（超过阈值） | -5e-4 | -4e-4 | -5e-4 |
| `feet_stumble` | 惩罚绊脚/异常碰撞（水平冲击过强） | -1.25 | -1.25 | -1.25 |
| `dof_pos_limits` | 惩罚关节角超出安全范围 | -5.0 | -5.0 | -5.0 |
| `dof_torque_limits` | 惩罚关节扭矩接近或超过软限位 | -1.0 | -1.0 | -1.0 |
| `dof_vel` | 惩罚关节速度过大（抑制抖动） | -1e-4 | -1e-4 | -1e-4 |
| `dof_acc` | 惩罚关节加速度过大（抑制突变） | -5e-8 | -5e-8 | -5e-8 |
| `action_rate` | 惩罚相邻时刻动作变化过快 | -0.01 | -0.005 | -0.01 |
| `feet_air_time` | 调整步态摆动相位，鼓励合适的腾空时长 | 5.0 | 4.5 | 5.0 |
| `ang_vel_xy` | 惩罚机体绕 x/y 轴角速度（抑制横滚俯仰晃动） | -0.01 | -0.01 | -0.01 |
| `total_angular_momentum` | 惩罚全身总角动量过大，提升稳定性 | -0.01 | -0.01 | -0.01 |
| `ankle_dof_acc` | 专门惩罚踝关节加速度过大 | -1e-7 | -1e-7 | -1e-7 |
| `ankle_dof_vel` | 专门惩罚踝关节速度过大 | -2e-4 | -2e-4 | -2e-4 |
| `alive` | 存活奖励（每步常数奖励） | - | - | - |
| `tracking_feet_height` | 跟踪双脚高度与参考高度的一致性 | - | - | - |
| `collision` | 惩罚非允许身体部位碰撞 | - | - | - |
| `feet_height` | 约束脚部高度接近目标高度 | - | - | - |
| `lin_vel_z` | 惩罚机体竖直方向速度过大 | - | - | - |
| `orientation` | 惩罚机体姿态偏离竖直 | - | - | - |
| `torque_penalty` | 惩罚关节扭矩平方和（抑制能耗/暴力控制） | - | - | - |
| `waist_dof_acc` | 专门惩罚腰部关节加速度过大 | - | - | - |
| `waist_dof_vel` | 专门惩罚腰部关节速度过大 | - | - | - |
| `ankle_action` | 专门惩罚踝关节动作幅度过大 | - | - | - |

## 5. 常用奖励函数定义位置
- `tracking_joint_dof`: `humanoid_mimic.py::_reward_tracking_joint_dof`
- `tracking_joint_vel`: `humanoid_mimic.py::_reward_tracking_joint_vel`
- `tracking_root_pose`: `humanoid_mimic.py::_reward_tracking_root_pose`
- `tracking_root_vel`: `humanoid_mimic.py::_reward_tracking_root_vel`
- `tracking_keybody_pos`: `humanoid_mimic.py::_reward_tracking_keybody_pos`
- `tracking_keybody_pos_upper`: `humanoid_mimic.py::_reward_tracking_keybody_pos_upper`
- `tracking_cmd_vel`: `humanoid_mimic.py::_reward_tracking_cmd_vel`
- `tracking_cmd_yaw`: `humanoid_mimic.py::_reward_tracking_cmd_yaw`
- `action_symmetry`: `humanoid_mimic.py::_reward_action_symmetry`
- `total_angular_momentum`: `humanoid_mimic.py::_reward_total_angular_momentum`
- `ankle_dof_acc`: `g1_mimic_distill.py::_reward_ankle_dof_acc`
- `ankle_dof_vel`: `g1_mimic_distill.py::_reward_ankle_dof_vel`
