# CMG 训练脚本说明

本文档说明两个脚本的用法：
- `train_teacher_cmg.sh`
- `train_student_cmg.sh`
## 0. 目前的计划
bash play_student.sh cmg_student_fast_v2 cmg_teacher_fast_v1 g1_cmg_fast --cmd_vx 2.0 --cmd_vy 0.0 --cmd_yaw 0.0
bash play_teacher.sh cmg_teacher_fast_v1 g1_cmg_fast --cmd_vx 3.0 --cmd_vy 0.0 --cmd_yaw 0.0

1. 进入环境

  cd /root/TWIST-jiarui
  conda activate twist

  2. 阶段1：低速适配（不蒸馏，不加载 teacher）

  - 从 cmg_student_fast_v2/model_7000.pt 续训
  - 用 fast_low 任务扩展到低速

  bash train_student_cmg.sh fast cmg_student_fast_v2_low cmg_teacher_fast_v1 cuda:0 -1 cmg_student_fast_v2 7000 2048 8200 unlock low g1_stu_rl_cmg_fast 1 1

  bash train_student_cmg.sh fast cmg_student_fast_v2_low cmg_teacher_fast_v1 cuda:0 -1 cmg_student_fast_v2_low -1 2048 2500 unlock low g1_stu_rl_cmg_fast 1 1

  ```bash
  1. fast：speed_mode，速度档位。会走 fast 分支。
  2. cmg_student_fast_v2_low：student_exptid，本次 student 训练/日志 run 名。
  3. cmg_teacher_fast_v1：teacher_exptid，teacher run 名（用于定位老师模型）。
  4. cuda:0：device，训练设备。
  5. -1：teacher_checkpoint，老师 checkpoint 取最新。
  6. cmg_student_fast_v2_low：resumeid，从这个 student run 继续训练。
  7. -1：checkpoint，student 从该 run 的最新 checkpoint 恢复。
  8. 2048：num_envs，并行环境数。
  9. 2500：max_iterations，本次最多训练 2500 iter。
  10. unlock：unlock_upper_body，会加 --unlock_upper_body（解锁上肢/全身跟踪）。
  11. low：curriculum_stage，fast 下课程阶段；这里会把任务切到 g1_stu_rl_cmg_fast_low。
  12. g1_stu_rl_cmg_fast：resume_proj_name，恢复时从这个项目目录找 run。
  13. 1：startup_check，开启 30 秒启动体检。
  14. 1：eval_student，开启 student-only 模式（不加载 teacher 做蒸馏，teacher KL 置 0）。
  ```

  最后一个 1 就是 eval_student=1（关蒸馏）。

  3. 阶段2：回补高速（开蒸馏）

  bash train_student_cmg.sh fast cmg_student_fast_v2_low cmg_teacher_fast_v1 cuda:0 -1
  cmg_student_fast_v2_low -1 2048 9000 unlock full g1_stu_rl_cmg_fast 1 0

  ```
    1. fast：speed_mode，速度档位。
  2. cmg_student_fast_v2_low：student_exptid，student 训练 run 名。
  3. cmg_teacher_fast_v1：teacher_exptid，teacher run 名。
  4. cuda:0：device，训练设备。
  5. 2000：teacher_checkpoint，固定加载 teacher 的 model_2000.pt（不是最新）。
  6. cmg_student_fast_v2_low：resumeid，从这个 student run 恢复。
  7. -1：checkpoint，恢复 student 时取该 run 最新 checkpoint。
  8. 2048：num_envs，并行环境数。
  9. 9000：max_iterations，本次训练迭代数设为 9000。
  10. unlock：unlock_upper_body，开启上肢解锁（会传 --unlock_upper_body）。
  11. full：curriculum_stage，fast 任务下使用 full 课程。
  12. g1_stu_rl_cmg_fast：resume_proj_name，恢复 checkpoint 时从这个项目目录找。
  13. 1：startup_check，开启 30 秒启动自检。
  14. 0：eval_student，关闭 student-only 模式，即启用 teacher 蒸馏（会加载 teacher）。
  ```

  最后一个 0 是 eval_student=0（开蒸馏）。
  这里我建议 teacher checkpoint 固定用 2000，避免 -1 读到波动版本。

  4. 快速验证

  bash play_student.sh cmg_student_fast_v2_low cmg_teacher_fast_v1 g1_cmg_fast --cmd_vx 1.5 --cmd_vy 0.0 --cmd_yaw 0.0
  bash play_student.sh cmg_student_fast_v2_low cmg_teacher_fast_v1 g1_cmg_fast --cmd_vx 3.0 --cmd_vy 0.0 --cmd_yaw 0.0



## 1. `train_teacher_cmg.sh`

### 1.1 功能
训练 CMG Teacher（按速度模式：`slow / medium / fast`）。

### 1.2 命令格式
```bash
bash train_teacher_cmg.sh <speed_mode> <exptid> <device> [resumeid] [checkpoint] [num_envs] [max_iterations]
```

### 1.3 参数说明
| 参数 | 必填 | 含义 | 示例 | 默认值 |
|---|---|---|---|---|
| `speed_mode` | 是 | 速度模式，只能是 `slow`/`medium`/`fast` | `medium` | `slow` |
| `exptid` | 是 | 本次训练实验名（日志目录名） | `cmg_teacher_medium_v1` | `cmg_<speed_mode>_v1` |
| `device` | 是 | 训练设备 | `cuda:0` | `cuda:0` |
| `resumeid` | 否 | 恢复训练的 run 名（位于 `legged_gym/logs/<proj_name>/` 下） | `cmg_teacher_medium_v1` | 空 |
| `checkpoint` | 否 | 恢复的 checkpoint 编号 | `12000` | 空（若配合 resume，通常表示 latest） |
| `num_envs` | 否 | 并行环境数 | `2048` | 配置文件默认 |
| `max_iterations` | 否 | 最大迭代数 | `6001` | 配置文件默认 |

### 1.4 speed_mode 映射
| `speed_mode` | `task_name` | `proj_name` |
|---|---|---|
| `slow` | `g1_cmg_slow` | `g1_cmg_slow` |
| `medium` | `g1_cmg_medium` | `g1_cmg_medium` |
| `fast` | `g1_cmg_fast` | `g1_cmg_fast` |

### 1.5 常见示例
1. 从头训练 medium teacher：
```bash
bash train_teacher_cmg.sh medium cmg_teacher_medium_v1 cuda:0
```

2. 指定并行环境数（你之前用过的形式）：
```bash
bash train_teacher_cmg.sh medium cmg_teacher_medium_v1 cuda:0 "" "" 2048
```
说明：`"" ""` 用来跳过 `resumeid` 和 `checkpoint`，直接传第 6 个参数 `num_envs`。

3. 从旧 run 恢复训练并指定 checkpoint：
```bash
bash train_teacher_cmg.sh medium cmg_teacher_medium_v1 cuda:0 cmg_teacher_medium_v1 12000 2048 8000
```

4. 从该 run 下最后一个（最新/最大）checkpoint 续训：
```bash
bash train_teacher_cmg.sh medium cmg_teacher_medium_v1 cuda:0 cmg_teacher_medium_v1 -1
```
5. 从最大的checkpoint，且环境数是2048
```
bash train_teacher_cmg.sh medium cmg_teacher_medium_v1 cuda:0 cmg_teacher_medium_v1 -1 2048

bash train_teacher_cmg.sh fast cmg_teacher_fast_v1 cuda:0 cmg_teacher_fast_v1 -1 2048
```
说明：`checkpoint=-1` 表示自动选择最新 checkpoint。

### 1.6 输出位置
以 `medium + cmg_teacher_medium_v1` 为例：
- 日志目录：`legged_gym/logs/g1_cmg_medium/cmg_teacher_medium_v1`
- 模型 checkpoint：在上面目录内按训练过程保存

---

## 2. `train_student_cmg.sh`

### 2.1 功能
训练 CMG Student（DAgger: RL+BC），并从对应 CMG Teacher checkpoint 蒸馏。

### 2.2 命令格式
```bash
bash train_student_cmg.sh <speed_mode> <student_exptid> <teacher_exptid> <device> [teacher_checkpoint] [resumeid] [checkpoint] [num_envs] [max_iterations]
```

### 2.3 参数说明
| 参数 | 必填 | 含义 | 示例 | 默认值 |
|---|---|---|---|---|
| `speed_mode` | 是 | 速度模式，只能是 `slow`/`medium`/`fast` | `medium` | `medium` |
| `student_exptid` | 是 | Student 实验名 | `cmg_student_medium_v1` | `cmg_student_<speed_mode>_v1` |
| `teacher_exptid` | 是 | Teacher 实验名（用于定位老师模型） | `cmg_teacher_medium_v1` | `cmg_teacher_<speed_mode>_v1` |
| `device` | 是 | 训练设备 | `cuda:0` | `cuda:0` |
| `teacher_checkpoint` | 否 | 读取老师的 checkpoint 编号，`-1` 表示最新 | `12000` | `-1` |
| `resumeid` | 否 | 恢复 Student run 名 | `cmg_student_medium_v1` | 空 |
| `checkpoint` | 否 | 恢复 Student 的 checkpoint 编号 | `8000` | 空 |
| `num_envs` | 否 | 并行环境数 | `1024` | 配置文件默认 |
| `max_iterations` | 否 | 最大迭代数 | `30000` | 配置文件默认 |

### 2.4 speed_mode 映射（Teacher 项目目录）
`train_student_cmg.sh` 会自动设置 `teacher_proj_name`：

| `speed_mode` | `teacher_proj_name` |
|---|---|
| `slow` | `g1_cmg_slow` |
| `medium` | `g1_cmg_medium` |
| `fast` | `g1_cmg_fast` |

所以 student 会从：
- `legged_gym/logs/<teacher_proj_name>/<teacher_exptid>`
读取老师模型。

同时，student 训练任务与日志目录会自动对齐到 CMG student 任务：

| `speed_mode` | `task_name` | `proj_name` |
|---|---|---|
| `slow` | `g1_stu_rl_cmg_slow` | `g1_stu_rl_cmg_slow` |
| `medium` | `g1_stu_rl_cmg_medium` | `g1_stu_rl_cmg_medium` |
| `fast` | `g1_stu_rl_cmg_fast` | `g1_stu_rl_cmg_fast` |

### 2.5 常见示例
1. 用 medium teacher 训练 medium student（最常用）：
```bash
bash train_student_cmg.sh medium cmg_student_medium_v1 cmg_teacher_medium_v1 cuda:0
```

2. 指定老师 checkpoint（不是 latest）：
```bash
bash train_student_cmg.sh medium cmg_student_medium_v1 cmg_teacher_medium_v1 cuda:0 12000
```

3. 恢复 student 自己的训练：
```bash
bash train_student_cmg.sh medium cmg_student_medium_v1 cmg_teacher_medium_v1 cuda:0 -1 cmg_student_medium_v1 8000
```

4. 同时限制环境数和迭代数（降低显存压力时常用）：
```bash
bash train_student_cmg.sh fast cmg_student_medium_v1 cmg_teacher_medium_v1 cuda:0 -1 "" "" 1024 6000
```

5. Student 从自己最后一个 checkpoint 续训，同时 Teacher 也读取最后一个 checkpoint：
```bash
bash train_student_cmg.sh medium cmg_student_medium_v1 cmg_teacher_medium_v1 cuda:0 -1 cmg_student_medium_v1 -1
```
说明：第 5 个参数 `-1` 是 teacher 最新 checkpoint；第 7 个参数 `-1` 是 student 最新 checkpoint。

### 2.6 输出位置
- Student 日志目录：`legged_gym/logs/g1_stu_rl_cmg_<speed_mode>/<student_exptid>`
- Teacher 加载目录：`legged_gym/logs/g1_cmg_<speed_mode>/<teacher_exptid>`

---

## 3. 运行前检查建议
1. 确认 teacher 目录存在（以 medium 为例）：
```bash
ls legged_gym/logs/g1_cmg_medium/cmg_teacher_medium_v1
```

2. 如果遇到 CUDA OOM，可优先降低 `num_envs`（如 `2048 -> 1024 -> 512`）。

3. 两个脚本都已设置：
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
用于降低显存碎片带来的 OOM 概率。

---

## 4. `play_teacher.sh` 命令示例（随机/固定指令）

`play_teacher.sh` 会调用 `legged_gym/legged_gym/scripts/play.py`。  
对于 CMG 任务（`g1_cmg_*`）：
- 不传 `--cmd_vx/--cmd_vy/--cmd_yaw` 时，命令是随机采样的。
- 传了这些参数时，会使用你指定的固定命令。

### 4.1 随机命令（CMG）
1. medium teacher，使用随机命令：
```bash
bash play_teacher.sh cmg_teacher_medium_v1
```
说明：如果不手动指定 `proj_name`，脚本会根据 `exptid` 自动映射到 `g1_cmg_medium`。

2. 显式指定项目名（效果等价）：
```bash
bash play_teacher.sh cmg_teacher_medium_v1 g1_cmg_medium
```

### 4.2 固定命令（CMG）
如果你想固定速度命令，可直接在 `play_teacher.sh` 后追加参数：

1. 固定前进 2.0 m/s，侧向 0，偏航 0：
```bash
bash play_teacher.sh cmg_teacher_medium_v1 g1_cmg_medium --cmd_vx 2.0 --cmd_vy 0.0 --cmd_yaw 0.0
```

2. 固定前进 1.5 m/s，左移 0.2 m/s，左转 0.3 rad/s：
```bash
bash play_teacher.sh cmg_teacher_medium_v1 g1_cmg_medium --cmd_vx 1.5 --cmd_vy 0.2 --cmd_yaw 0.3
```

3. 用角度指定偏航速度（度/秒）：
```bash
bash play_teacher.sh cmg_teacher_medium_v1 g1_cmg_medium --cmd_vx 1.5 --cmd_vy 0.0 --cmd_yaw 30 --cmd_yaw_deg
```
说明：`--cmd_yaw_deg` 会把 `--cmd_yaw` 按度数转换为 rad/s。

---

## 5. `play_student.sh`（fast + 最新 checkpoint）

`play_student.sh` 会调用 `play.py`，而 `play.py` 的 `--checkpoint` 默认是 `-1`，即自动选择该 run 下最新（最大编号）checkpoint。

```bash
bash play_student.sh cmg_student_fast_v1 cmg_teacher_fast_v1 g1_cmg_fast
```

手动指定固定命令（cmd）可直接在后面追加：
```bash
bash play_student.sh cmg_student_fast_v1 cmg_teacher_fast_v1 g1_cmg_fast --cmd_vx 2.0 --cmd_vy 0.0 --cmd_yaw 0.0
```

当前仓库里 `legged_gym/logs/g1_stu_rl/cmg_student_fast_v1` 的最新模型是（旧路径示例）：
- `model_10000.pt`

---

## 6. Teacher / Student 一键对齐对比（CMG）

仓库根目录提供了对比脚本：

```bash
bash compare_cmg_teacher_student.sh <speed_mode> <student_exptid> <teacher_exptid> [student_checkpoint] [teacher_checkpoint] [cmd_vx] [cmd_vy] [cmd_yaw]
```

示例（fast 档，固定 3 m/s 直行）：

```bash
bash compare_cmg_teacher_student.sh fast cmg_student_fast_v2 cmg_teacher_fast_v1 -1 -1 3.0 0.0 0.0
```

说明：
- 脚本会按同一组 `vx/vy/yaw` 连续运行 teacher 和 student。
- `play.py` 默认使用确定性评测配置（禁用 domain randomization），便于公平对比。
- 如需恢复随机化播放，可在 `play_teacher.sh` 或 `play_student.sh` 里追加 `--eval_randomized`。


# skill
直接在对话里点名：$isaac-rl，或自然描述 Isaac RL 训练/调参/排障需求。
  3. 常用提问示例：
      - $isaac-rl 帮我分析训练不收敛的原因，给最小改动方案
      - $isaac-rl 基于这个任务配置给我 PPO 调参优先级
      - $isaac-rl 比较 logs 目录下最近几次训练结果并建议下一组实验
  4. 你也可以直接用内置脚本汇总日志：
      - python /root/.codex/skills/isaac-rl/scripts/extract_metrics.py <log文件或目录>
      - python /root/.codex/skills/isaac-rl/scripts/extract_metrics.py <log文件或目录> --json