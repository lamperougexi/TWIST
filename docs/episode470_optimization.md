# 针对 Episode≈470 平台期的专项优化说明

## 1. 470 的情况是什么
- 对象 run：`g1_stu_rl_cmg_fast/cmg_student_fast_v2_low_cmdswitch`
- 关键现象（优化前）：
  - `Mean episode length = 470.01`（见 `train_20260311_125440.log:46030`）
  - `Average_episode_length = 483.5703`（见 `train_20260311_125440.log:46056`）
  - 任务是 `cmdswitch`，且命令切换开启（见 `train_20260311_125440.log:39`）
- 解释：
  - 在原配置 `episode_length_s=10`、`dt=0.02` 下，理论上限是 500 step。
  - 均值长期在 470~490 一带，说明已经接近上限，但仍有一部分 episode 在后段提前失败（不是纯粹全部顶满 500）。

## 2. 优化总思路
- 目标不是“只拉高上限”，而是同时处理三类问题：
  - 上限分辨率不足（500 太容易顶格）
  - cmd switch 冲击过大（切换太早、delta 偏大）
  - 扰动与切换叠加导致后段跌倒
- 你当前反馈“优化后有提升”，说明这套改动方向与当前 run 痛点匹配。

## 3. 每个情况对应的改动

| 情况 | 改动原因 | 改动内容 | 预期结果 |
|---|---|---|---|
| 情况 A：长度接近 500 上限，checkpoint 差距被压缩 | 470~490 区间已接近 cap，指标分辨率不足 | 在 `G1MimicStuRLCMGFastLowCmdSwitchCfg.env` 中将 `episode_length_s: 10 -> 12`（见 `g1_mimic_distill_config.py:533-535`） | 上限从 500 提到 600 step，长度指标可分辨性更好，后段稳定性差异更容易体现 |
| 情况 B：切换太早，策略尚未稳定就发生 cmd_A->cmd_B | 减少“起步阶段就切换”导致的早期冲击 | `cmd_switch_time_range_s: [0.1, 1.0] -> [0.2, 1.2]`（见 `g1_mimic_distill_config.py:545`） | 提高切换前稳定时间，减少早期切换触发的异常终止 |
| 情况 C：切换跳变量偏大，瞬时扰动过强 | 保留切换有效性，同时降低冲击 | `cmd_switch_min_delta: [0.3, 0.15, 0.15] -> [0.25, 0.12, 0.12]`（见 `g1_mimic_distill_config.py:548`） | 切换仍有挑战性，但降低一次切换造成的大幅失稳 |
| 情况 D：高速边缘命令占比偏高，容易触发后段掉落 | 当前平台期更需要“先稳住再放开” | 在 cmdswitch 任务轻度收窄命令范围：`vx [1.0,3.5]->[1.0,3.4]`、`vy/yaw [-0.5,0.5]->[-0.45,0.45]`（见 `g1_mimic_distill_config.py:539-541`） | 降低极端命令尾部导致的失败率，提升 episode 长度稳定性 |
| 情况 E：随机推扰与切换叠加，放大后段摔倒风险 | 优先评估“切换鲁棒性”而非“强推扰抗性” | cmdswitch 任务内减弱推扰：`push_interval_s 6->8`、`max_push_vel_xy 0.6->0.45`、`max_push_force_end_effector 12->10`（见 `g1_mimic_distill_config.py:551-556`） | 下降由随机推扰造成的噪声失败，episode 长度更稳定 |
| 情况 F：后期优化步长/探索噪声仍偏“躁” | 平台期需要更稳的策略更新 | 新增 `G1MimicStuRLCMGFastLowCmdSwitchCfgDAgger`：`desired_kl 0.003->0.0025`、`entropy_coef 0.004->0.0035`（见 `g1_mimic_distill_config.py:703-709`） | 降低策略抖动，减少后期随机跌倒，提高长度曲线平滑性 |
| 情况 G：任务注册仍指向旧 DAgger 配置 | 需要确保新超参真正生效 | `g1_stu_rl_cmg_fast_low_cmdswitch` 绑定到新 DAgger 配置（见 `envs/__init__.py:66`） | 启动同名任务即可自动使用优化后的算法配置 |

## 4. 本次改动清单（代码入口）
- 配置改动文件：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
- 任务注册改动文件：`legged_gym/legged_gym/envs/__init__.py`

## 5. 预期收益与观察指标
- 预期直接收益：
  - `Train/mean_episode_length` 不再长期卡在 470 左右，后段波动减小
  - `Average_episode_length` 向更高区间移动
- 建议重点跟踪：
  - `Train/mean_episode_length`
  - `Episode_rew_metrics/metric_cmd_vel_err_post_switch`
  - `Episode_rew_metrics/metric_cmd_yaw_err_post_switch`
  - `Episode_rew_metrics/metric_cmd_switch_trigger_rate`

## 6. 备注
- 本优化是“平台期稳定化版本”，优先提升稳定与可分辨性，不追求一次性把任务难度拉满。
- 当长度和切换误差稳定后，可再逐步把命令范围和扰动强度放回原设定做回归验证。
