# cmg_student_fast_v2 续训与收敛修复计划

## 1. 目标
- 让 `g1_stu_rl_cmg_fast/cmg_student_fast_v2` 从“低回报+短episode”进入稳定上升区间。
- 先修复训练流程中的确定性问题（代码/配置），再做超参与课程策略优化。
- 形成可复现实验流程，避免再次出现“参数错位/OOM/中断后无效续训”。

---

## 2. 当前状态（基于本地 wandb + 日志）

### 2.1 有效训练与失败训练
- 多个 run 在环境创建或初次 reset 即失败，主要是 **4096 env OOM**。
- 真正完成了连续训练的核心 run：
  - `run-20260227_000033-8uluvzwu`（迭代到 342 后手动中断）
  - `run-20260227_013559-pwqxb7hq`（继续到 453）

### 2.2 收敛表现
- `cmg_student_fast_v2` 在 0-453 迭代内，`mean_reward` 从负数升到约 `0.14~0.20`，说明并非完全不学。
- 但 `mean_episode_length` 仍仅约 `7~11`，和 teacher 差距巨大，属于“慢收敛/受限收敛”。

### 2.3 teacher 参考上限
- `cmg_teacher_fast_v1` 在对应区间显著更高（同量级迭代下回报和 episode 长度远高于 student）。
- 说明主要不是 teacher checkpoint 损坏，而是 student 训练路径本身存在障碍。

---

## 3. 已定位的高优先级问题

## P0-1: DaggerRunner 中 critic 归一化使用不一致（代码问题）
- 文件：`rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`
- 现象：
  - 代码创建了 `critic_normalizer`，但训练时 `critic_obs` 实际用的是 `teacher_normalizer`。
  - 导致 student critic/value 学习可能被 teacher 统计分布“绑死”，分布偏移时不稳定。

## P0-2: 迭代计数与保存逻辑不一致（代码问题）
- 文件：`rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`
- 现象：
  - `current_learning_iteration` 只在保存点更新。
  - 日志分母出现 `30002 -> 30402` 这类跳变，且中断时可能保存到落后 checkpoint。
  - 续训可追踪性下降，也容易误判训练进度。

## P0-3: student 初始步数计数被硬置为大值（配置/代码问题）
- 文件：`legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`
- 现象：
  - `obs_type == 'student'` 时，`total_env_steps_counter/global_counter` 被设为 `24*100000`。
  - 噪声增益与若干 schedule 相当于一开始就进入“后期状态”，对 fast 任务早期学习不利。

## P0-4: 训练过程稳定性不足（运行问题）
- 4096 env 多次 OOM，导致大量 run 几秒退出。
- 中断频繁，实际有效数据量不够，无法判断长期收敛趋势。

---

## 4. 执行计划（按优先级）

## 阶段 A：先修确定性问题（必须先做）
1. 修复 DaggerRunner 的 critic 归一化路径  
   - 让 `critic_obs` 与 `critic_normalizer` 对齐（与 `OnPolicyRunnerMimic` 行为一致）。
   - teacher normalizer 只用于 teacher 推理支路，不混入 student critic 输入归一化。

2. 修复迭代计数与保存逻辑  
   - 每个迭代结束都更新 `current_learning_iteration = it`（或 `it+1`，全局一致即可）。
   - 末尾保存时与真实迭代一致，避免“看似到 453 但只存到 400”的误差。

3. 调整 student 初始计数器策略  
   - 删除/改小 `24*100000` 的硬编码，恢复从 0 起步或与 resume checkpoint 对齐。

交付物：
- 一个可编译可运行的 patch（仅改上述三点，不混入其他改动）。

---

## 阶段 B：稳定训练流程（防止再次“无效 run”）
1. 固定默认续训命令模板，避免参数错位：
   - 一律单行命令，显式给 `resumeid/checkpoint/num_envs`。
2. 先用可稳定跑通的环境数：
   - 优先 `2048`（必要时 `1536`/`1024`），避免 4096 触发 OOM。
3. 每次启动后 30 秒内检查：
   - 是否打印正确 `Num Envs`
   - 是否正确加载 teacher/student checkpoint
   - 是否开始输出 iteration 日志

---

## 阶段 C：收敛策略优化（在 A/B 后进行）
1. 保留 CMG DAgger 强监督设置  
   - `dagger_coef=0.1 -> 0.03`（现有 CMG DAgger 配置），先不降。

2. 减少早期任务难度（fast任务专用）
   - 方案 C1（推荐）：先在 `fast` 中临时收窄命令范围（例如先收窄 vx/yaw），待 episode 长度上来后再放开。
   - 方案 C2：先做 `medium` student 蒸馏到可控长度，再迁移到 fast。

3. 加入清晰的训练闸门（gate）
   - Gate-1（前 200 iter）：`mean_reward` 持续正值，且非下降趋势。
   - Gate-2（前 600 iter）：`mean_episode_length` 必须明显高于当前 7~11 区间。
   - Gate-3（前 1000 iter）：`tracking_cmd_vel`、`tracking_cmd_yaw` 不能停滞在当前低位。

---

## 5. 实验矩阵（建议最小集）

## Exp-A（代码修复后基线）
- 环境数：2048
- 其余保持当前配置
- 目标：验证修复后曲线是否明显优于现状

## Exp-B（命令难度收窄）
- 在 Exp-A 基础上收窄 fast 的 `cmg_vx_range/cmg_yaw_range`
- 目标：先拉长 episode，再追求高速度精度

## Exp-C（若 A/B 仍慢）
- 先 medium student 预训练，再切回 fast 续蒸馏
- 目标：减少 fast 初期探索难度

---

## 6. 结果判定标准
- 若 1000 iter 内仍长期停留在 `mean_episode_length < 15` 且 `mean_reward` 近零波动：
  - 判定当前配置仍不适配 fast，进入 C 阶段迁移方案。
- 若 `mean_episode_length` 持续上升且 reward 同步上升：
  - 保持当前配置，延长训练时长，不频繁改超参。

---

## 7. 下一步实施顺序（可直接执行）
1. 先提交“仅修代码确定性问题”的 patch（阶段 A）。
2. 用 2048 env 跑一个连续 1000 iter 的稳定实验（阶段 B）。
3. 根据 gate 决定是否进入命令范围课程或 medium->fast 迁移（阶段 C）。

