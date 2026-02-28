# Reward Function 总结（当前项目）

## 1. 总体形式
项目中的总奖励按以下方式计算：

\[
r_t = \sum_i \big(w_i \cdot f_i(s_t, a_t)\big)
\]

其中：
- `f_i` 是具体 reward function（定义在 `legged_gym/legged_gym/envs/base/humanoid_mimic.py` 和 `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`）
- `w_i` 是配置里的 `rewards.scales.<name>`
- 代码里会先把所有非零 `w_i` 乘以 `dt`（见 `legged_gym/legged_gym/envs/base/legged_robot.py::_prepare_reward_function`）

注意：下面表格给的是**配置原始权重**（未乘 `dt`）。

---

## 2. 当前项目主要任务与对应权重
仓库注册了这些任务（`legged_gym/legged_gym/envs/__init__.py`）：
- `g1_priv_mimic`（mocap teacher）
- `g1_stu_rl`（student）
- `g1_cmg_slow / g1_cmg_medium / g1_cmg_fast`（CMG teacher）

其中 `g1_cmg_slow/medium/fast` 的 reward 权重相同，只是速度命令范围不同。

### 2.1 `g1_priv_mimic` / `g1_stu_rl`（非 CMG）
来源：`G1MimicPrivCfg.rewards.scales` 与 `G1MimicStuRLCfg.rewards.scales`

| reward_function | 含义（简述） | 权重 |
|---|---|---:|
| `tracking_joint_dof` | 跟踪参考关节角（exp 形式） | 0.6 |
| `tracking_joint_vel` | 跟踪参考关节角速度（exp） | 0.2 |
| `tracking_root_pose` | 跟踪根部位姿（位置+朝向，exp） | 0.6 |
| `tracking_root_vel` | 跟踪根部线/角速度（exp） | 1.0 |
| `tracking_keybody_pos` | 跟踪关键点位置（exp） | 2.0 |
| `feet_slip` | 惩罚接触时脚底滑移 | -0.1 |
| `feet_contact_forces` | 惩罚过大足端接触力 | -5e-4 |
| `feet_stumble` | 惩罚绊脚接触 | -1.25 |
| `dof_pos_limits` | 惩罚超关节限位 | -5.0 |
| `dof_torque_limits` | 惩罚超软扭矩限 | -1.0 |
| `dof_vel` | 惩罚关节速度平方和 | -1e-4 |
| `dof_acc` | 惩罚关节加速度平方和 | -5e-8 |
| `action_rate` | 惩罚动作变化过快 | -0.01 |
| `feet_air_time` | 足端腾空时间项 | 5.0 |
| `ang_vel_xy` | 惩罚机体横滚/俯仰角速度 | -0.01 |
| `total_angular_momentum` | 惩罚全身总角动量（绕系统质心） | -0.01 |
| `ankle_dof_acc` | 惩罚踝关节加速度平方和 | -1e-7 |
| `ankle_dof_vel` | 惩罚踝关节速度平方和 | -2e-4 |

### 2.2 `g1_cmg_slow` / `g1_cmg_medium` / `g1_cmg_fast`（CMG）
来源：`G1MimicCMGBaseCfg.rewards.scales`

| reward_function | 含义（简述） | 权重 |
|---|---|---:|
| `tracking_joint_dof` | 跟踪参考关节角（exp） | 0.6 |
| `tracking_joint_vel` | 跟踪参考关节角速度（exp） | 0.2 |
| `tracking_root_pose` | 跟踪根部位姿（exp） | 0.2 |
| `tracking_root_vel` | 跟踪根部速度（exp） | 0.8 |
| `tracking_keybody_pos` | 跟踪关键点位置（CMG 下主要下半身） | 2.0 |
| `tracking_keybody_pos_upper` | 上半身关键点跟踪（当前关闭） | 0.0 |
| `tracking_cmd_vel` | 跟踪 CMG 命令速度 `vx, vy`（exp） | 1.5 |
| `tracking_cmd_yaw` | 跟踪 CMG 命令偏航角速度（exp） | 1.0 |
| `action_symmetry` | 左右动作镜像对称性奖励（exp） | 0.1 |
| `feet_slip` | 惩罚接触时脚底滑移 | -0.1 |
| `feet_contact_forces` | 惩罚过大足端接触力 | -5e-4 |
| `feet_stumble` | 惩罚绊脚接触 | -1.25 |
| `dof_pos_limits` | 惩罚超关节限位 | -5.0 |
| `dof_torque_limits` | 惩罚超软扭矩限 | -1.0 |
| `dof_vel` | 惩罚关节速度平方和 | -1e-4 |
| `dof_acc` | 惩罚关节加速度平方和 | -5e-8 |
| `action_rate` | 惩罚动作变化过快 | -0.01 |
| `feet_air_time` | 足端腾空时间项 | 5.0 |
| `ang_vel_xy` | 惩罚机体横滚/俯仰角速度 | -0.01 |
| `total_angular_momentum` | 惩罚全身总角动量（绕系统质心） | -0.01 |
| `ankle_dof_acc` | 惩罚踝关节加速度平方和（继承） | -1e-7 |
| `ankle_dof_vel` | 惩罚踝关节速度平方和（继承） | -2e-4 |

---

## 3. 关键 reward function 定义（简化）
- `tracking_joint_dof`: `exp(-0.15 * Σ(w_dof * (q_ref-q)^2))`
- `tracking_joint_vel`: `exp(-0.01 * Σ(w_dof * (dq_ref-dq)^2))`
- `tracking_root_pose`: `exp(-5.0 * (||Δp||^2 + 0.1*Δrot^2))`
- `tracking_root_vel`: `exp(-1.0 * (||Δv||^2 + 0.5*||Δω||^2))`
- `tracking_keybody_pos`: `exp(-10.0 * Σ||Δkeybody||^2)`
- `tracking_keybody_pos_upper`: `exp(-5.0 * Σ||Δupper_keybody||^2)`
- `tracking_cmd_vel`: `exp(-2.0 * ((v_x^cmd-v_x)^2 + (v_y^cmd-v_y)^2))`
- `tracking_cmd_yaw`: `exp(-(ω_z^cmd-ω_z)^2)`
- `action_symmetry`: `exp(-0.5 * MSE(a_left - mirror(a_right)))`
- `total_angular_momentum`: `||L/M||^2`，其中 `L = Σ((r_i-r_com) × m_i(v_i-v_com))`
