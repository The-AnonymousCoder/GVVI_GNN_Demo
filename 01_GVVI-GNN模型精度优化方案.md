# GVVI-GNN 模型精度优化方案

## 0. 交付目标

本方案交给 DeepSeek 执行。目标是在不伪造结果、不改变固定测试集、不引入官方 GVVI 标签泄漏的前提下，提高综合 GVVI 的预测精度。

当前经过审计和一次低风险优化后的真实结果：

| 指标 | 原结果 | 当前结果 |
|---|---:|---:|
| MAE | 0.0219 | 0.0212 |
| RMSE | 0.0366 | 0.0363 |
| R² | 0.3670 | 0.3759 |
| Pearson | 0.6074 | 0.6170 |
| Spearman | 0.7584 | 0.7697 |

当前结果来自固定 `seed=42`、相同测试节点，并已独立核对：

- 官方 GVVI 回放最大误差为 0；
- `predictions.csv` 可独立复算相同测试指标；
- 综合预测严格满足 S:W:D = 3:2:1；
- CSV 与四张预测 GeoTIFF 一致；
- 没有把 `total_pixel`、`img_id`、`pixel_wvi` 或 GVVI 真值作为输入。

优化目标按优先级排序：

1. 综合 GVVI `R² ≥ 0.50`；
2. 综合 GVVI `Spearman ≥ 0.80`；
3. 综合 GVVI `MAE ≤ 0.020`；
4. 同时尽量改善最弱的 `GVVI_W`；
5. 保持预测结果可解释且可复现。

如果无法达到目标，必须报告真实结果，不允许改变测试集或使用测试标签调参。

## 1. 为什么 RTX 4090 上一分钟训练完是正常的

当前模型训练快不等于结果虚假，也不等于模型一定没有训练充分。

当前计算规模：

```text
节点：27,597
边：245,724
特征：73维
模型参数：约22.6万
图：一次性完整放入24 GB显存
输入：统计特征，不是图像、点云或三角网格
```

这属于小型全图节点回归。RTX 4090 每个 epoch 只需要处理约 2.8 万个节点和 24.6 万条边，600个 epoch 在几十秒到数分钟内完成完全合理。

训练时间长短不是模型质量标准。影响当前精度的主要瓶颈不是 GPU 算力，而是：

1. 输入没有表达“观察相机是否朝向这个绿化网格”；
2. 输入没有建筑遮挡和真实视线信息；
3. 当前图只有单一 k=8 邻接尺度；
4. 当前输出直接回归，未显式处理大量零值；
5. 当前训练标签占70%，仍有增加空间；
6. 当前模型对综合 GVVI 的约束仍较弱。

因此优化重点应是增强信息和建模方式，而不是单纯把 epoch 改成几千。

## 2. 审计确认的实验边界

当前是“同一研究区内随机节点补全”：

- 70%训练；
- 15%验证；
- 15%测试；
- 测试节点标签不参与损失；
- 测试节点的无标签空间特征参与全图消息传播。

这是合法的传导式节点回归，但不能称为：

- 跨区域泛化；
- 跨城市泛化；
- 完全不依赖精确标签；
- 完全替代 Cesium/OpenGL。

模型使用 `x_local/y_local`、栅格行列号和 tile 坐标，因此能够学习研究区的位置场。这不是直接标签泄漏，但使任务更接近空间插值。最终报告必须保留这个说明。

## 3. 数据划分：增加标签但保持测试集不变

可以增加“精确计算好的 GVVI”占比，但不能通过缩小或更换测试集人为提高成绩。

### 3.1 固定原测试集

读取现有：

```text
gvvi_grid_demo/processed/split_seed42.npz
```

其中原 `test_idx` 永久冻结，仍为 4,146 个节点。所有优化模型必须在完全相同的测试节点上比较。

### 3.2 把训练标签从70%增加到80%

原划分：

```text
70% train + 15% val + 15% test
```

优化划分：

```text
80% train + 5% val + 原15% test
```

具体做法：

1. 原测试集完全不动；
2. 将原训练集全部保留；
3. 从原验证集按 GVVI 分位数分层抽取约2/3加入训练集；
4. 剩余约1/3作为5%验证集；
5. 固定 `seed=42`；
6. 保存为 `split_train80_val5_test15_seed42.npz`。

这样确实增加了精确标签比例，同时还能进行早停。禁止使用测试指标选择epoch或模型。

## 4. 最重要的特征升级：相机是否真正朝向绿化网格

当前特征只统计附近观察点的数量、距离、高度和平均 Heading，但没有计算每个相机的朝向是否对准当前绿化网格。

这是当前最大的可修复信息缺口，尤其影响 WVI 和 SVI。

### 4.1 方向对齐角

对每个“绿化网格 g—附近观察点 o”：

```text
相机位置：(xo, yo)
绿化中心：(xg, yg)
相机Heading：h
观察点指向绿化的方位角：bearing(o→g)
方向差：Δθ = wrap(bearing - h) ∈ [-180°, 180°]
```

构造：

```python
alignment_cos = cos(delta_theta)
front_facing = abs(delta_theta) <= fov_half_angle
```

分别在 50、100、250、500 m 内，针对 SVI、WVI、DVI 统计：

- 正对该网格的观察点数量；
- 正对比例；
- `alignment_cos` 的平均值、最大值；
- 前向观察点最近距离；
- 前向观察点距离衰减和。

建议视角半角：

```text
SVI：60°
WVI：60°
DVI：60°
```

如果论文或相机配置有更精确 FoV，使用官方值并记录来源。

### 4.2 距离衰减暴露特征

单纯计数没有体现近距离观察点贡献更大。对每种视角、每个尺度计算：

```python
kernel_50  = Σ exp(-distance / 50)
kernel_100 = Σ exp(-distance / 100)
kernel_250 = Σ exp(-distance / 250)
```

再加入只对 `front_facing=True` 的方向加权版本：

```python
directional_kernel = Σ max(cos(delta_theta), 0) * exp(-distance / scale)
```

这批特征比简单增加网络层数更可能提高精度。

### 4.3 高度与俯仰代理

绿化网格当前缺少真实高度，因此不要伪造绿化高度。可以构造观察点高度代理：

- 半径内观察点 Z 的分位数：P25、P50、P75；
- 不同高度带观察点数量；
- WVI 按楼层或 Z 高度分箱的数量；
- DVI 按 `height_id` 或飞行高度分箱的数量；
- 高度与水平距离的组合：

```python
elevation_proxy = atan2(view_z - local_ground_proxy, horizontal_distance)
```

若没有可靠地面高程，不要声称它是真实俯仰角，只能称为高度—距离代理。

### 4.4 WVI建筑上下文

`window_view_location.csv` 包含 `bldg_ID`、立面法向量和楼层相关字段。对每个绿化网格的附近窗口统计：

- 唯一建筑数；
- 每栋建筑窗口数的均值和最大值；
- 不同 Z 高度带的窗口数；
- 面向该网格的窗口比例；
- 最近的面向窗口距离；
- 面向窗口的距离衰减和；
- 高层窗口与低层窗口比例。

WVI当前 R² 最低，这组特征优先级最高。

### 4.5 SVI道路上下文

`SVI_position_road.csv` 包含 `RouteID` 和 `pair_id`。加入：

- 附近唯一道路 RouteID 数；
- 同一道路多方向观察点数量；
- 面向该网格的道路视图比例；
- 最近正向街景距离；
- 不同道路方向的熵。

### 4.6 DVI高度层上下文

`DVI_position_60.csv` 包含 `points_id`、`height_id`、`z` 和 `heading_id`。加入：

- 附近唯一无人机平面位置数；
- 不同高度层数量；
- 每个高度层的观察点数；
- 面向网格的无人机视角数；
- 高度层加权距离衰减。

### 4.7 特征缓存

新特征计算比训练耗时更长。必须缓存：

```text
gvvi_grid_demo_v2/processed/features_v2.npz
gvvi_grid_demo_v2/processed/feature_names_v2.json
```

缓存中记录：

- 特征版本；
- 半径；
- FoV；
- 源文件SHA256或大小；
- 节点顺序；
- 是否存在NaN/Inf。

禁止把大型缓存提交到Git。

## 5. 多尺度图结构

单一 k=8 图主要传播极近邻信息。建议构建两种邻接：

### 5.1 局部图

```text
k_local = 8
```

用于学习相邻10 m网格的连续性。

### 5.2 中尺度图

```text
k_context = 24
```

用于学习街区尺度的相似观察条件。

两张图都：

- 无向化；
- 去除重复边；
- 添加自环；
- 记录边距离；
- 检查连通分量。

不要建立全连接图，也不要重新回到千万级观察—绿化二部图。

## 6. 推荐模型：多尺度残差 GraphSAGE

不建议直接换成极复杂的 HGT 或 Graph Transformer。当前只有2.8万节点，关系类型单一，复杂模型容易增加不稳定性而不增加信息。

推荐唯一主模型：

> Multi-Scale Residual GraphSAGE

结构：

```text
73维旧特征 + 新方向/建筑/道路/高度特征
                    ↓
         Tabular Encoder（256维）
                    ↓
       ┌────────────┴────────────┐
       ↓                         ↓
局部GraphSAGE(k=8)        中尺度GraphSAGE(k=24)
256→256→128               256→256→128
       └────────────┬────────────┘
                    ↓
      拼接：原始编码 + 局部表示 + 中尺度表示
                    ↓
       Residual Fusion MLP（256维）
                    ↓
       三个零值分类头 + 三个正值回归头
                    ↓
        GVVI_S / GVVI_W / GVVI_D
                    ↓
             官方3∶2∶1合成
```

建议：

- 每个分支2层GraphSAGE；
- LayerNorm优先于BatchNorm；
- GELU；
- Dropout 0.15；
- 残差连接；
- 总参数控制在100万—300万；
- 不超过4层消息传播，避免过平滑。

训练仍然可能只需几分钟，这是正常现象。

## 7. 针对零值分布的输出头

当前：

```text
GVVI_S非零约51.6%
GVVI_W非零约73.3%
GVVI_D非零约72.9%
```

建议每个视角使用“是否为零 + 正值大小”的双头：

```text
p_nonzero = sigmoid(classification_head)
positive_value = sigmoid(regression_head)
prediction = p_nonzero × positive_value
```

损失：

```text
L_nonzero = BCE(p_nonzero, y > 0)
L_positive = SmoothL1(positive_value[y>0], y[y>0])
```

这样能减少模型对大量接近零节点的平均化，并有机会改善高值区 R²。

## 8. 训练损失

建议总损失：

```text
L =
  λ_cls × Σ BCE(nonzero_type)
+ λ_reg × [3/6 L_S + 2/6 L_W + 1/6 L_D]
+ λ_gvvi × MSE(pred_GVVI, true_GVVI)
+ λ_rank × PairwiseRankingLoss
```

初始权重固定：

```text
λ_cls  = 0.25
λ_reg  = 1.00
λ_gvvi = 1.00
λ_rank = 0.10
```

PairwiseRankingLoss只在训练节点中随机采样高低值节点对，目标是提高Spearman。不要使用测试节点构造排序对。

如果排序损失实现不稳定，允许将 `λ_rank=0`，但不要更换测试集。

## 9. 训练策略

固定：

| 项目 | 设置 |
|---|---|
| Optimizer | AdamW |
| 初始学习率 | 0.001 |
| Weight decay | 0.0001 |
| 最大epoch | 1,500 |
| Early stopping | 150 |
| LR scheduler | ReduceLROnPlateau |
| 最低学习率 | 1e-6 |
| Gradient clipping | 5.0 |
| Seed | 42 |

最大1500轮不代表一定训练1500轮。必须使用验证集早停，并保存：

- 最佳epoch；
- train/val loss曲线；
- 学习率曲线；
- 每轮耗时；
- 峰值显存；
- 最佳checkpoint。

若验证损失已经稳定，继续增加epoch不会自动提高测试精度。

## 10. 优化执行顺序

为防止DeepSeek同时修改太多内容后无法定位问题，按以下阶段执行。

### 阶段A：冻结基准

1. 保存当前commit、当前指标和固定测试ID；
2. 将当前预测复制到 `baseline_optimized_v1/`；
3. 验证当前指标可复算；
4. 不再覆盖基准文件。

### 阶段B：只升级特征

保持当前GraphSAGE结构，替换为V2特征和80/5/15划分。

只使用验证集判断是否继续。测试集只在该阶段方案冻结后评估一次。

### 阶段C：多尺度残差GraphSAGE

使用冻结的V2特征：

- k=8局部分支；
- k=24上下文分支；
- 残差融合；
- 双头输出；
- 综合GVVI一致性损失。

### 阶段D：最终训练

根据验证集选择阶段B或阶段C中更好的配置。选择规则固定：

```text
第一优先：验证集综合GVVI R²
第二优先：验证集综合GVVI Spearman
第三优先：验证集综合GVVI MAE
```

配置确定后，只对冻结测试集评估一次，生成最终文件。

这不是大规模消融，而是防止复杂方案失败时没有可用结果的两级保险。

## 11. 禁止事项

禁止：

- 根据测试集表现选择特征或模型；
- 更换原测试节点；
- 把测试节点加入训练；
- 使用关系表中的可见像素作为输入；
- 使用官方GVVI或邻居GVVI作为输入特征；
- 先对全体标签做平滑再划分；
- 用测试集拟合校准曲线；
- 报告训练集或全体节点指标冒充测试指标；
- 只展示高值预测较好的局部区域；
- 为增加训练时长加入无意义层数；
- 声称训练更慢就更准确。

## 12. 最终输出

新目录：

```text
gvvi_grid_demo_v2/
├── config.yaml
├── requirements.txt
├── run_all.py
├── src/
├── processed/
├── checkpoints/
├── outputs/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── pred_GVVI_S.tif
│   ├── pred_GVVI_W.tif
│   ├── pred_GVVI_D.tif
│   ├── pred_GVVI.tif
│   └── figures/
└── reports/
    ├── optimization_report.md
    └── audit_report.md
```

必须输出图：

1. 官方GVVI地图；
2. V2预测GVVI地图；
3. 测试节点绝对误差地图；
4. 测试集预测—真值散点图；
5. 训练/验证损失曲线；
6. V1与V2指标对比表；
7. GVVI_S/W/D三个分任务指标表。

报告必须写明：

- 标签比例从70%增加到80%；
- 测试集保持原4,146个节点不变；
- 当前任务是同区传导式补全；
- 模型没有使用逐视图渲染像素；
- 训练速度快是因为图规模小；
- 精度上限受建筑遮挡信息缺失影响。

## 13. DeepSeek完成标准

DeepSeek最后必须提供：

1. 本地和GPU使用的Git commit；
2. 完整GPU运行命令；
3. GPU型号、运行时间和峰值显存；
4. 最佳epoch；
5. 固定测试集的五项指标；
6. V1和V2逐项对比；
7. 四张GeoTIFF；
8. 所有最终PNG；
9. 是否达到R²≥0.50；
10. 未达到时的真实原因；
11. 所有修改commit并push到GitHub。

不得只回复“实验成功”，必须给出文件和数值证据。
