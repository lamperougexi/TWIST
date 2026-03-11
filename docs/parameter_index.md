# 项目可调参数总览（CMG / Student-Teacher）

本文档整理当前仓库中“可调参数”的主要入口、含义与路径，面向日常训练、评估和播放。

## 0. 范围说明

- 覆盖范围：`train_*`/`play_*` 脚本参数、`train.py/play.py` 可传 CLI 参数、核心配置文件参数（`HumanoidCharCfg`/`HumanoidMimicCfg`/`g1_mimic_distill_config.py`/`CMGMotionLib`）。
- 不覆盖：第三方库内部参数（Isaac Gym 内部未在本仓库暴露的隐藏参数）。
- 参数可调方式分三层：
  1. 命令行（不改代码）
  2. 配置类字段（改配置文件）
  3. 运行时逻辑参数（改实现文件）

## 1. 参数生效优先级

1. 脚本参数（如 `train_student_cmg.sh` 传入）
2. `train.py` / `play.py` 的 CLI 参数（见 `legged_gym/legged_gym/gym_utils/helpers.py`）
3. `update_cfg_from_args(...)` 覆盖配置（`helpers.py`）
4. 任务默认配置类（`*_config.py`）
5. 代码硬编码默认（如 `play.py:set_play_cfg`）

路径：`legged_gym/legged_gym/gym_utils/helpers.py`

---

## 2. 入口脚本参数（无需改代码）

### 2.1 `train_teacher_cmg.sh`

路径：`/root/TWIST-jiarui/train_teacher_cmg.sh`

| 参数 | 默认值 | 含义 | 生效路径 |
|---|---:|---|---|
| `speed_mode` | `slow` | 速度档位，决定任务 | `--task g1_cmg_{slow/medium/fast}` |
| `exptid` | `cmg_${speed_mode}_v1` | 本次实验名（日志目录） | `--exptid` |
| `device` | `cuda:0` | 训练设备 | `--device` |
| `resumeid` | 空 | 从哪个 run 恢复 | `--resumeid` |
| `checkpoint` | 空 | 恢复的 checkpoint 编号 | `--checkpoint` |
| `num_envs` | 空 | 并行环境数覆盖 | `--num_envs` |
| `max_iterations` | 空 | 最大迭代次数覆盖 | `--max_iterations` |
| `unlock_upper_body` | `unlock` | 是否解锁上肢跟踪 | `--unlock_upper_body` |

环境变量：
- `CONDA_PREFIX` / `LD_LIBRARY_PATH`：Isaac Gym 运行库路径
- `PYTORCH_CUDA_ALLOC_CONF`：CUDA 显存分配策略

### 2.2 `train_student_cmg.sh`

路径：`/root/TWIST-jiarui/train_student_cmg.sh`

| 参数 | 默认值 | 含义 | 生效路径 |
|---|---:|---|---|
| `speed_mode` | `medium` | 速度档位 | 决定 teacher 项目与 student 任务 |
| `student_exptid` | `cmg_student_${speed_mode}_v1` | student 实验名 | `--exptid` |
| `teacher_exptid` | `cmg_teacher_${speed_mode}_v1` | teacher 实验名 | `--teacher_exptid` |
| `device` | `cuda:0` | 训练设备 | `--device` |
| `teacher_checkpoint` | `-1` | teacher checkpoint（`-1` 最新） | `--teacher_checkpoint` |
| `resumeid` | 空 | student 恢复 run 名 | `--resumeid` |
| `checkpoint` | 空 | student 恢复 checkpoint | `--checkpoint` |
| `num_envs` | `2048` | 并行环境数 | `--num_envs` |
| `max_iterations` | 空 | 最大迭代数 | `--max_iterations` |
| `unlock_upper_body` | `unlock` | 是否解锁上肢 | `--unlock_upper_body` |
| `curriculum_stage` | `full` | fast 子任务阶段 | `full/low/narrow/cmdswitch` |
| `resume_proj_name` | 空 | 跨项目恢复目录 | `--resume_proj_name` |
| `startup_check` | `1` | 启动 30s 健康检查 | 脚本内部行为 |
| `eval_student` | `0` | student-only（禁蒸馏） | `--eval_student` |

任务映射（fast）：
- `full` -> `g1_stu_rl_cmg_fast`
- `low` -> `g1_stu_rl_cmg_fast_low`
- `narrow` -> `g1_stu_rl_cmg_fast_narrow`
- `cmdswitch` -> `g1_stu_rl_cmg_fast_low_cmdswitch`

### 2.3 `play_teacher.sh`

路径：`/root/TWIST-jiarui/play_teacher.sh`

| 参数 | 默认值 | 含义 | 生效路径 |
|---|---:|---|---|
| `teacher_exptid` | 必填 | teacher 实验名 | `--exptid` |
| `teacher_proj_name` | 自动推断 | teacher 项目名 | `--proj_name` + `--task` |
| `extra_play_args...` | 空 | 透传给 `play.py` | 例如 `--cmd_vx/--record_log` |

### 2.4 `play_student.sh`

路径：`/root/TWIST-jiarui/play_student.sh`

| 参数 | 默认值 | 含义 | 生效路径 |
|---|---:|---|---|
| `student_exptid` | 必填 | student 实验名 | `--exptid` |
| `teacher_exptid` | 必填 | teacher 实验名（用于 DAgger/映射） | `--teacher_exptid` |
| `teacher_proj_name` | 自动推断 | teacher 项目名 | `--teacher_proj_name` |
| `extra_play_args...` | 空 | 透传 `play.py` | 例如 `--cmd_vx --cmd_vy --cmd_yaw` |
| `--cmdswitch` | 关闭 | 启用随机时刻切换指令播放模式 | 切到任务 `g1_stu_rl_cmg_fast_low_cmdswitch` |

`--cmdswitch` 目前仅支持 `g1_cmg_fast`。

### 2.5 `test_cmg.sh`

路径：`/root/TWIST-jiarui/test_cmg.sh`

| 参数 | 默认值 | 含义 | 生效路径 |
|---|---:|---|---|
| `run_dir` | 必填 | 模型目录（含 `model_*.pt`） | 主输入 |
| `cmd_template` | 空 | 每个 checkpoint 执行的自定义评估命令 | 占位符 `{checkpoint}/{model_path}/{run_dir}` |

---

## 3. `train.py` / `play.py` CLI 参数

统一定义路径：`legged_gym/legged_gym/gym_utils/helpers.py`

### 3.1 Isaac Gym 基础参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--sim_device` | `cuda:0` | 物理仿真设备 |
| `--pipeline` | `gpu` | Tensor 管线（`cpu/gpu`） |
| `--graphics_device_id` | `0` | 图形设备 ID |
| `--flex` | `False` | 使用 Flex 物理引擎 |
| `--physx` | `False` | 使用 PhysX 物理引擎 |
| `--num_threads` | `0` | PhysX 线程数 |
| `--subscenes` | `0` | PhysX 子场景数 |
| `--slices` | `subscenes` | 客户端线程切片数 |

### 3.2 项目自定义参数（完整）

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--task` | `h1_mimic` | 任务名 |
| `--resume` | `False` | 恢复训练 |
| `--experiment_name` | 空 | 实验名（runner 级） |
| `--run_name` | 空 | run 名 |
| `--load_run` | 空 | 指定恢复 run |
| `--checkpoint` | `-1` | checkpoint（`-1` 最新） |
| `--headless` | `False` | 无图形模式 |
| `--horovod` | `False` | 多 GPU Horovod |
| `--rl_device` | `cuda:0` | RL 设备 |
| `--num_envs` | 空 | 环境数覆盖 |
| `--seed` | 空 | 随机种子 |
| `--max_iterations` | 空 | 最大迭代数 |
| `--device` | `cuda:0` | 统一设备（覆盖 sim/rl） |
| `--rows` | 空 | 地形行数 |
| `--cols` | 空 | 地形列数 |
| `--debug` | `False` | 调试模式（禁 wandb） |
| `--proj_name` | `h1` | 项目日志目录名 |
| `--exptid` | 空 | 实验 ID |
| `--entity` | 空 | wandb entity |
| `--resumeid` | 空 | 恢复实验 ID |
| `--resume_proj_name` | 空 | 跨项目恢复目录 |
| `--use_jit` | `False` | 播放时加载 JIT policy |
| `--draw` | `False` | 播放时画图 |
| `--save` | `False` | 评估数据保存 |
| `--action_delay` | `False` | 动作延迟 |
| `--web` | `False` | web viewer |
| `--no_wandb` | `False` | 禁用 wandb |
| `--viz` | `False` | 训练可视化 |
| `--record_video` | `False` | 录视频 |
| `--eval_randomized` | `False` | 播放时启用随机化 |
| `--fix_action_std` | `False` | 固定动作 std |
| `--no_rand` | `False` | 关闭域随机化 |
| `--teleop_mode` | `False` | 遥操作模式 |
| `--record_log` | `False` | 播放记录日志 |
| `--use_transformer` | `False` | 使用 Transformer policy |
| `--teacher_exptid` | `mimic` | teacher 实验名 |
| `--teacher_proj_name` | `g1_priv_mimic` | teacher 项目名 |
| `--teacher_checkpoint` | `-1` | teacher checkpoint |
| `--unlock_upper_body` | `False` | 解锁上肢 |
| `--eval_student` | `False` | student-only（禁 teacher 蒸馏） |
| `--motion_file` | `g1_zhen_phc` | motion 数据配置名 |
| `--cmd_vx` | `None` | CMG 前向速度命令 |
| `--cmd_vy` | `None` | CMG 侧向速度命令 |
| `--cmd_yaw` | `None` | CMG 偏航角速度命令 |
| `--cmd_yaw_deg` | `False` | 将 `cmd_yaw` 按度/秒解释 |

参数映射逻辑路径：`helpers.py:update_cfg_from_args(...)`

---

## 4. 配置文件参数（需改代码）

## 4.1 基础层：`HumanoidCharCfg`

路径：`legged_gym/legged_gym/envs/base/humanoid_char_config.py`

### 4.1.1 `env`（环境规模/观测）

常用：
- `num_envs`：并行环境数
- `episode_length_s`：单回合时长
- `normalize_obs`：是否归一化观测
- `randomize_start_pos` / `randomize_start_yaw` / `rand_yaw_range`：初始状态随机化
- `obs_type`、`num_observations`、`num_privileged_obs`：观测维度与类型

### 4.1.2 `terrain`（地形）

常用：
- `mesh_type`：`plane/heightfield/trimesh`
- `num_rows` / `num_cols`：地形课程网格
- `horizontal_scale` / `vertical_scale`：地形分辨率
- `height` / `gap_size` / `stepping_stone_distance`：障碍强度
- `terrain_dict`：各地形采样权重

### 4.1.3 `control`（控制器）

常用：
- `control_type`：`P/V/T`
- `stiffness` / `damping`：PD 参数
- `action_scale`：动作缩放
- `decimation`：控制降采样比

### 4.1.4 `domain_rand`（域随机化）

常用：
- `domain_rand_general`（总开关）
- `randomize_gravity` / `gravity_range`
- `randomize_friction` / `friction_range`
- `randomize_base_mass` / `added_mass_range`
- `randomize_base_com` / `added_com_range`
- `push_robots` / `push_interval_s` / `max_push_vel_xy`
- `action_delay` / `action_buf_len`

### 4.1.5 `rewards`（奖励）

常用：
- `regularization_names` / `regularization_scale`
- `tracking_sigma` / `tracking_sigma_ang`
- `termination_height` / `termination_roll` / `termination_pitch`
- `scales.*`：各奖励项权重

### 4.1.6 `noise`（观测噪声）

常用：
- `add_noise` / `noise_level` / `noise_increasing_steps`
- `noise_scales.{dof_pos,dof_vel,lin_vel,ang_vel,gravity,imu}`

### 4.1.7 `commands`（速度命令）

常用：
- `resampling_time`
- `lin_vel_clip` / `ang_vel_clip`
- `ranges.lin_vel_x / lin_vel_y / ang_vel_yaw`

### 4.1.8 `motion`（参考运动 / CMG）

常用：
- `motion_file`
- `motion_curriculum` / `motion_curriculum_gamma`
- `key_bodies` / `upper_key_bodies`
- `use_cmg`
- `cmg_model_path` / `cmg_data_path`
- `cmg_dt` / `cmg_vx_range` / `cmg_vy_range` / `cmg_yaw_range`

## 4.2 Mimic 层：`HumanoidMimicCfg` / `HumanoidMimicCfgPPO`

路径：`legged_gym/legged_gym/envs/base/humanoid_mimic_config.py`

- `env.enable_early_termination`
- `env.pose_termination` / `pose_termination_dist`
- `env.root_tracking_termination_dist`
- `env.tar_obs_steps`
- `env.rand_reset`
- `env.track_root`
- `env.global_obs`
- `env.dof_err_w`

PPO 侧关键：
- `policy.actor_hidden_dims` / `critic_hidden_dims`
- `policy.activation` / `fix_action_std`
- `algorithm.learning_rate` / `entropy_coef` / `clip_param`
- `algorithm.num_learning_epochs` / `num_mini_batches`
- `algorithm.desired_kl` / `max_grad_norm`
- `runner.num_steps_per_env` / `max_iterations` / `save_interval`

## 4.3 任务层：`g1_mimic_distill_config.py`

路径：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`

### 4.3.1 任务注册路径

`legged_gym/legged_gym/envs/__init__.py`

当前常用任务：
- Student: `g1_stu_rl_cmg_slow/medium/fast/fast_low/fast_narrow/fast_low_cmdswitch`
- Teacher: `g1_cmg_slow/medium/fast`

### 4.3.2 CMG student 关键参数（`G1MimicStuRLCMGBaseCfg`）

- `motion.cmg_vx_range/cmg_vy_range/cmg_yaw_range`：命令范围
- `motion.enable_cmd_switch`：是否启用 `cmd_A->cmd_B`
- `motion.cmd_switch_time_range_s`：切换时间范围（秒）
- `motion.cmd_switch_once_per_episode`：每回合切换次数策略
- `motion.cmd_switch_min_delta`：A/B 最小差异
- `motion.cmd_switch_post_window_s`：切换后统计窗口
- `rewards.scales.tracking_cmd_vel/tracking_cmd_yaw`：命令跟踪权重

### 4.3.3 速度档位范围

- `Slow`: `vx [0.5,1.5]`
- `Medium`: `vx [1.5,2.5]`
- `Fast`: `vx [2.5,3.5]`
- `FastLow`: `vx [1.0,3.5]`
- `FastNarrow`: `vx [2.2,3.0], vy [-0.3,0.3], yaw [-0.25,0.25]`

### 4.3.4 DAgger 参数（`G1MimicStuRLCfgDAgger*`）

- `runner.eval_student`：是否纯 student（不蒸馏）
- `runner.teacher_experiment_name/teacher_proj_name/teacher_checkpoint`
- `algorithm.dagger_coef`：teacher KL 权重
- `algorithm.dagger_coef_min`：退火下限
- `algorithm.dagger_coef_anneal_steps`：退火步数
- `algorithm.learning_rate/clip_param/desired_kl/entropy_coef`

## 4.4 CMG 运行时参数：`CMGMotionLib`

路径：`pose/pose/utils/cmg_motion_lib.py`

构造参数（`__init__`）：
- `cmg_model_path`：CMG 模型路径
- `cmg_data_path`：CMG 训练数据统计路径
- `urdf_path`：机器人 URDF
- `device` / `num_envs` / `episode_length_s`
- `dt`：CMG 时间步
- `vx_range` / `vy_range` / `yaw_range`
- `enable_cmd_switch`
- `switch_time_range`
- `switch_once_per_episode`
- `cmd_switch_min_delta`
- `root_height`

切换相关运行时状态可观测接口：
- `get_cmd_switch_status(...)`
- `get_commands()`
- `set_commands(...)`

---

## 5. 播放逻辑里可调（代码硬编码）

路径：`legged_gym/legged_gym/scripts/play.py`

`set_play_cfg(...)` 内固定可调项（需改代码）：
- `env_cfg.env.num_envs = 2`
- `env_cfg.env.episode_length_s = 60`
- `terrain.num_rows/num_cols = 5`
- `terrain.curriculum = False`
- `noise.add_noise = False`
- `deterministic=False` 时的随机化项（摩擦、push 等）

CMG 指令注入逻辑：
- `--cmd_vx/--cmd_vy/--cmd_yaw` 会覆盖当前命令并 `reset` 轨迹

---

## 6. 常见调参目标 -> 对应参数路径

1. 提升训练稳定性（防 collapse）
- 路径：`g1_mimic_distill_config.py :: *DAgger.algorithm`
- 参数：`learning_rate`, `clip_param`, `max_grad_norm`, `desired_kl`, `entropy_coef`

2. 提升突变命令响应速度
- 路径：`G1MimicStuRLCMGFastLowCmdSwitchCfg.motion`
- 参数：`enable_cmd_switch`, `cmd_switch_time_range_s`, `cmd_switch_min_delta`
- 路径：`...rewards.scales`
- 参数：`tracking_cmd_vel`, `tracking_cmd_yaw`

3. 降低摔倒率
- 路径：`...rewards.scales`
- 参数：`feet_stumble`, `dof_pos_limits`, `dof_torque_limits`, `action_rate`
- 路径：`...domain_rand`
- 参数：`push_interval_s`, `max_push_vel_xy`, `randomize_*`

4. 控制命令分布
- 路径：`...motion`
- 参数：`cmg_vx_range`, `cmg_vy_range`, `cmg_yaw_range`

5. 加快训练速度/降显存
- 路径：脚本参数
- 参数：`num_envs`, `max_iterations`, `device`

---

## 7. 建议

- 先用脚本参数调（可复现实验成本低），再改配置类。
- 每次只改一组参数（例如只改 reward 或只改 domain rand）。
- 把每次实验对应命令和改动路径记录到同一个 run 日志目录中。

---

## 8. 配置类全索引（按文件）

用于“全量查找可调参数”的入口索引（从这些类向下看每个字段即可）。

### 8.1 `g1_mimic_distill_config.py`

路径：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`

- `G1MimicPrivCfg`
- `G1MimicStuCfg`
- `G1MimicStuRLCfg`
- `G1MimicStuRLCMGBaseCfg`
- `G1MimicStuRLCMGSlowCfg`
- `G1MimicStuRLCMGMediumCfg`
- `G1MimicStuRLCMGFastCfg`
- `G1MimicStuRLCMGFastLowCfg`
- `G1MimicStuRLCMGFastLowCmdSwitchCfg`
- `G1MimicStuRLCMGFastNarrowCfg`
- `G1MimicPrivCfgPPO`
- `G1MimicStuRLCfgDAgger`
- `G1MimicStuRLCMGCfgDAgger`
- `G1MimicStuRLCMGFastCfgDAgger`
- `G1MimicStuRLCMGFastLowCfgDAgger`
- `G1MimicCMGBaseCfg`
- `G1MimicCMGSlowCfg`
- `G1MimicCMGMediumCfg`
- `G1MimicCMGFastCfg`
- `G1MimicCMGSlowCfgPPO`
- `G1MimicCMGMediumCfgPPO`
- `G1MimicCMGFastCfgPPO`

### 8.2 Base 配置

路径：`legged_gym/legged_gym/envs/base/humanoid_char_config.py`

- `HumanoidCharCfg`
- `HumanoidCharCfgPPO`

路径：`legged_gym/legged_gym/envs/base/humanoid_mimic_config.py`

- `HumanoidMimicCfg`
- `HumanoidMimicCfgPPO`

---

## 9. 参数自检命令（建议保存）

在仓库根目录执行，可快速核对“当前代码实际可调参数”：

```bash
# 1) 列出 train/play 脚本参数定义
rg -n \"Usage:|speed_mode|teacher_exptid|curriculum_stage|cmdswitch|checkpoint|num_envs|max_iterations\" \
  train_teacher_cmg.sh train_student_cmg.sh play_teacher.sh play_student.sh test_cmg.sh

# 2) 列出 Python CLI 参数（helpers.py）
nl -ba legged_gym/legged_gym/gym_utils/helpers.py | sed -n '190,252p'

# 3) 列出 g1 配置类（任务级参数入口）
rg -n \"^class .*Cfg\" legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

# 4) 列出 CMG 运行时切换参数
rg -n \"enable_cmd_switch|switch_time_range|cmd_switch_min_delta|get_cmd_switch_status\" \
  pose/pose/utils/cmg_motion_lib.py
```
