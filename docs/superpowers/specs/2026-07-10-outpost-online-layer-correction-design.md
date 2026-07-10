# 前哨站锁层后在线纠错设计

## 目标

解决前哨站已经锁层后，运行过程中低板被单帧关联错误地当成高板，并把错误层号直接写入 EKF 的问题。

保持现有快速初锁、前哨开火窗口、普通自瞄和打符行为不变。纠错只在已锁层的前哨站目标上运行，所有修改只保存在本地仓库，不上传 GitHub。

## 根因

当前 `Tracker::update_target` 为每个观测和三个预测层计算综合分数，然后把单帧最低分层号作为 `forced_id` 传给 `Target::update`。`Target` 不复核该层号，直接用它更新 EKF。

当低板的姿态或角度解算偏差使高板综合分数更低时，低板会以高板层号更新。由于高低板相差 0.2 m，错误更新随后会拉动 EKF 基准高度，使错误逐渐自洽。现有 `outpost_layer_residual` 只记录调试值，不参与纠错。

## 纠错证据

对 Tracker 已选中的同一观测，分别计算它到三个预测层的高度残差：

```text
z_residual[id] = abs(observed_z - predicted_layer_z[id])
height_id = argmin(z_residual)
z_improvement = z_residual[raw_id] - z_residual[height_id]
```

- `raw_id`：现有综合评分选出的层号。
- `height_id`：只看高度残差最小的层号。
- 只有 `height_id != raw_id` 且 `z_improvement >= 0.06 m` 时，才产生一票纠错证据。
- 该 0.06 m 是“改层后高度拟合至少改善多少”，不是观测绝对高度误差上限。

## 三帧确认

Tracker 保存前哨纠错候选层号和连续帧数：

```text
pending_layer = -1
pending_count = 0
required_frames = 3
```

处理规则：

1. 原始层号与高度层号一致，清空纠错状态并按原始层号正常更新。
2. 高度改善不足 0.06 m，清空纠错状态并按原始层号正常更新。
3. 高度证据与上一帧指向同一纠错层，连续帧数加一。
4. 高度证据改变、目标丢失或观测中断，重新从一帧开始。
5. 前两帧确认期间不调用 `Target::update`，返回未找到，使现有 Tracker 进入或保持 `temp_lost`；因此 EKF 不会被错误层号污染，Shooter 也会因非 tracking 自动停火。
6. 第三帧确认后，用 `height_id` 作为修正层号调用 `Target::update`，清空纠错状态并恢复 tracking。

纠错期间继续保留 Target 的预测输出，避免直接清空目标模型和红框。

## 配置

在 `configs/sentry.yaml` 增加：

```yaml
outpost_layer_correction_enabled: true
outpost_layer_correction_frames: 3
outpost_layer_correction_z_gate: 0.06
```

- `outpost_layer_correction_enabled`：只关闭在线纠错，不改变其他前哨逻辑。
- `outpost_layer_correction_frames`：连续确认帧数，最小值限制为 1。
- `outpost_layer_correction_z_gate`：改层后相对原始层号至少改善的高度残差，单位 m，最小值限制为 0。

## 数据流与边界

在线纠错状态归 Tracker 所有，因为候选层综合评分和多观测选择发生在 Tracker。Target 继续只负责按照最终层号更新 EKF。

仅在以下条件同时满足时启用纠错：

- 当前目标名称为 `ArmorName::outpost`。
- Target 已完成层级锁定。
- 三个预测层均可用。
- 已找到现有综合评分的最佳观测和原始层号。

普通目标、未锁层前哨、全向相机目标切换和打符不进入该分支。

## 调试与内录

通过 Target 的 EKF 调试数据和 `ovsentry_omni_mpc` 内录增加：

```text
outpost_layer_raw_id
outpost_layer_height_id
outpost_layer_raw_z_residual
outpost_layer_best_z_residual
outpost_layer_z_improvement
outpost_layer_correction_count
outpost_layer_correction_pending
outpost_layer_correction_applied
```

发生低板误判时，预期先看到 `raw_id=2`、`height_id=0`、连续计数 1/2，第三帧 `applied=1`，随后 `outpost_selected_id=0`。

## 测试

自动测试覆盖：

- 原始层号和高度层号一致时立即更新，不进入纠错。
- 低板被原始评分判为高板时，前两帧不更新，第三帧修正为低板。
- 纠错候选在低板和中板之间交替时不会累计到三帧。
- 观测中断会重置连续计数。
- 高度改善小于门限时保留原始综合评分层号。
- 未锁层前哨和普通目标不受在线纠错影响。
- 纠错待确认期间 Tracker 非 tracking，从而保持停火。

本机 Windows 缺少 CMake，代码完成后执行静态检查；最终逻辑测试和 `ovsentry_omni_mpc` 编译在车上 Ubuntu 完成。
