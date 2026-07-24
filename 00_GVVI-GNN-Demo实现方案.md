# GVVI-GNN Demo：明天中午前完成的单实验实施方案

## 0. 目标、边界和截止时间

目标是在 2026 年论文官方数据上完成一个真实、可运行、可展示的 GNN Demo：

> 使用不依赖逐视图渲染的空间特征和绿化网格邻接关系，预测每个绿化网格的 `GVVI_S`、`GVVI_W`、`GVVI_D`，再严格按官方权重合成综合 GVVI。

本轮只做一个模型、一次固定数据划分和一套最终结果，不做：

- 消融实验；
- 主动学习；
- GAT、HGT、R-GCN 等模型比较；
- XGBoost、MLP 等基线；
- 超参数搜索；
- 跨城市泛化；
- OBJ、点云、Cesium 视图的重新渲染；
- 观察视图—绿化网格千万级候选边预测。

截止目标：明天中午前形成可汇报结果。优先级依次为：

1. 数据与官方 GVVI 真值对齐；
2. 模型能稳定训练并预测四张 GVVI 图；
3. 输出可信的测试指标和对比图；
4. 再考虑界面或动画。

## 1. DeepSeek 现有工作的审计结论

现有代码位于 `gvvi_graph_demo/`。已经完成：

- 三类官方关系表的数据审计；
- `positive_edges.parquet`，约 47 MiB；
- `candidate_edges.parquet`，约 636 MiB；
- 零绿化视图 ID 统计；
- 候选边、空间划分、特征、GraphSAGE、基线和 GVVI 适配器的代码骨架；
- `data_audit.md` 和 `candidate_recall.json`。

官方数据连接正确：

| 视角 | 关系中唯一视图ID | 位置表ID | 成功连接 |
|---|---:|---:|---:|
| SVI | 12,095 | 18,096 | 12,095 |
| WVI | 43,341 | 84,936 | 43,341 |
| DVI | 14,879 | 17,424 | 14,879 |

但现有边级路线不能直接继续。当前候选边召回率：

| 视角 | Candidate Recall |
|---|---:|
| SVI | 60.07% |
| WVI | 28.97% |
| DVI | 0.29% |
| 综合 | 6.13% |

候选图漏掉约 1,445 万条真实正边。模型不可能预测候选图中不存在的关系。若继续扩大半径、视锥和每视图邻居数，图规模及负样本会迅速膨胀，无法保证明天中午前完成。

因此：

- 保留现有代码和审计结果，作为探索记录；
- 本轮不读取 `candidate_edges.parquet`；
- 本轮不训练现有边级 GraphSAGE；
- 新实验放在 `gvvi_grid_demo/`，不要继续堆叠在旧流水线上。

## 2. 最容易成功且不伪造结果的实验

### 2.1 任务定义

把论文中的 10 m 绿化网格作为图节点：

```text
每个绿化网格 = 一个节点
空间相邻或距离最近的绿化网格 = 图边
```

节点输入只使用不依赖逐视图渲染的廉价信息：

- 绿化网格位置；
- 周围街道、窗口、无人机观察点的数量和距离分布；
- 周围观察点的方向、高度和视角类型统计；
- 相邻绿化网格上下文。

节点监督目标来自官方栅格：

```text
yS = 官方 GVVI_S 的归一化值
yW = 官方 GVVI_W 的归一化值
yD = 官方 GVVI_D 的归一化值
```

模型输出三项后，按照官方代码合成：

```text
pred_GVVI = 3/6 × pred_GVVI_S
          + 2/6 × pred_GVVI_W
          + 1/6 × pred_GVVI_D
```

这是“网格节点多任务回归”，不是语义分割、目标检测或边分类。

### 2.2 为什么这条路线最适合当前时间

- 只有 27,597 个节点，单张消费级 GPU 足够；
- kNN 图约 22 万条有向边，远小于千万级二部图；
- 官方四张 GeoTIFF 已经提供真值；
- 不需要重新渲染图像；
- 三个分视角任务共享空间表示，符合多视角 GVVI 逻辑；
- 两层 GNN 能利用邻域绿化暴露的空间连续性；
- 可以直接输出地图，适合汇报展示。

### 2.3 必须诚实说明的边界

本 Demo 验证的是：

> 在同一城市研究区内，只精确计算部分绿化网格后，GNN 能否利用廉价空间特征和邻域信息补全其余网格的 GVVI。

本 Demo 不能证明：

- 已经完全替代 Cesium/OpenGL；
- 可以直接跨城市使用；
- 未见过任何精确标签时也能预测；
- 比所有传统模型更好；
- 能恢复每一条具体观察—绿化可见关系。

汇报中称为“研究区内 GVVI 快速补全可行性 Demo”最准确。

## 3. 只使用这些官方文件

```text
GVVI-GNN-Demo/
├── greenery/
│   ├── df_grid_ply.csv
│   └── veg_raster_buffer_100.tif
├── viewpoints/
│   ├── SVI_position_road.csv
│   ├── window_view_location.csv
│   └── DVI_position_60.csv
├── reference/
│   ├── GVVI_S.tif
│   ├── GVVI_W.tif
│   ├── GVVI_D.tif
│   └── GVVI.tif
├── official_code/
│   ├── 03_GVVI.py
│   └── utils.py
└── paper/
    ├── 2026_Multi-perspective_GVVI_CIM_author_manuscript.pdf
    └── 2026_GVVI_official_appendix.pdf
```

本实验不需要读取：

- `relations/SVI.csv`、`WVI.csv`、`DVI.csv`；
- `positive_edges.parquet`；
- `candidate_edges.parquet`；
- 原始视图图片、OBJ 或点云。

关系表保留在仓库中用于溯源，但本轮模型不使用，避免将真实可见像素信息作为输入造成标签泄漏。

## 4. 数据预处理

### 4.1 建立绿化节点

读取 `greenery/df_grid_ply.csv`：

```text
greenery_id = grid_idx_x + "_" + grid_idx_y
center_x = (grid_coords_min_x + grid_coords_max_x) / 2
center_y = (grid_coords_min_y + grid_coords_max_y) / 2
```

保留 27,597 个唯一绿化网格。使用 `grid_idx_x`、`grid_idx_y` 与官方栅格数组索引连接，不要用经纬度反推像元。

### 4.2 读取监督真值

分别读取 `reference/GVVI_S.tif`、`GVVI_W.tif`、`GVVI_D.tif` 和 `GVVI.tif`。

对每个绿化节点取：

```python
target_s = gvvi_s[grid_idx_x, grid_idx_y]
target_w = gvvi_w[grid_idx_x, grid_idx_y]
target_d = gvvi_d[grid_idx_x, grid_idx_y]
target_gvvi = gvvi[grid_idx_x, grid_idx_y]
```

先检查四张栅格均为 `247 × 245`，且索引不越界。

需要确认 `GVVI_S/W/D.tif` 是官方 `03_GVVI.py` 读取的分视角栅格。训练目标统一转换为 `[0,1]`：

```python
normalize(x) = (x - x.min()) / (x.max() - x.min())
```

必须复用 `official_code/utils.py` 的 `normalize_raster` 逻辑。

无模型回放检查：

```text
official_replay =
3/6 × normalize(GVVI_S)
+ 2/6 × normalize(GVVI_W)
+ 1/6 × normalize(GVVI_D)
```

`official_replay` 必须与 `reference/GVVI.tif` 在浮点容差内一致。若不一致，停止训练并先修复索引、掩膜或归一化。

### 4.3 构造廉价节点特征

所有坐标统一使用香港 1980 Grid（EPSG:2326）的米制坐标。对每类观察点分别计算以下统计：

#### 距离邻域统计

半径固定为：

```text
50 m、100 m、250 m、500 m
```

每类视角、每个半径计算：

- 观察点数量；
- 最近距离；
- 平均距离；
- 距离标准差。

三类视角共 `3 × 4 × 4 = 48` 个特征。

#### 高度统计

在 250 m 范围内分别计算：

- 平均 Z；
- 最大 Z；
- 最小 Z；
- 绿化网格相对于观察点平均高度差。

三类视角共 12 个特征。

#### 方向统计

在 250 m 范围内，把 Heading 编码为：

```python
mean_sin_heading
mean_cos_heading
```

三类视角共 6 个特征。

#### 绿化网格自身特征

- 局部坐标 `x_local, y_local`；
- 归一化行列号；
- 所属 OBJ tile 的 one-hot 或频率编码；
- 相邻绿化网格密度；
- 到研究区边界的距离。

禁止使用：

- `R/G/B` 唯一颜色编码；
- `total_pixel`；
- `count`；
- `img_id`；
- `pixel_wvi`；
- 任意官方 GVVI 值作为输入；
- 从测试节点真值计算出的统计量。

最终特征控制在约 65—80 维。缺少邻域观察点时，数量置 0，距离置为半径上限或统一缺失值，并增加 `has_neighbor` 标记。

### 4.4 建立绿化网格图

使用网格中心坐标建立无向 kNN 图：

```text
k = 8
```

要求：

- 删除自环后再由框架显式添加自环；
- 邻接关系对称化；
- 边特征只保留 `dx, dy, distance`；
- 保存为 `processed/grid_graph.pt`；
- 输出节点数、边数、连通分量和平均度。

第一版固定 k=8，不测试其他 k。

## 5. 固定数据划分

为了明天前得到稳定结果，本轮采用固定、可复现的节点随机划分：

```text
train = 70%
validation = 15%
test = 15%
seed = 42
```

划分时按综合 GVVI 分位数组合分层，确保高、中、低值在三个子集中比例接近。

必须保存：

```text
processed/split_seed42.npz
```

同一份划分用于全部训练、早停和最终测试。测试集只在训练结束后评估一次。

说明：随机节点划分衡量同一研究区内的空间补全能力，邻居可能跨越训练/测试掩膜，但测试节点标签不得参与训练。汇报中不得把它表述为跨区域泛化。

## 6. 唯一模型：两层 GraphSAGE

模型固定为：

```text
输入特征
  ↓ Linear + LayerNorm + ReLU
128维
  ↓ GraphSAGE层 + ReLU + Dropout(0.2)
128维
  ↓ GraphSAGE层 + ReLU
64维
  ↓ MLP回归头
3维
  ↓ Sigmoid
GVVI_S、GVVI_W、GVVI_D
```

固定训练参数：

| 参数 | 数值 |
|---|---:|
| 隐藏维度 | 128 → 64 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| 最大 epoch | 300 |
| Early stopping patience | 30 |
| 随机种子 | 42 |
| 损失 | 三个分指标加权 SmoothL1 |

损失：

```text
L = 3/6 × SmoothL1(pred_S, true_S)
  + 2/6 × SmoothL1(pred_W, true_W)
  + 1/6 × SmoothL1(pred_D, true_D)
```

只在训练节点掩膜上计算损失。验证集用于早停。保存验证损失最低的唯一 checkpoint。

图只有约 2.8 万节点，优先尝试全图训练。若显存不足，再改为 NeighborLoader；不要同时维护两套训练路径。

设备选择：

```python
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

## 7. 官方 GVVI 合成

模型输出三个 `[0,1]` 分指标后，严格执行：

```python
pred_gvvi = (
    3.0 / 6.0 * pred_s
    + 2.0 / 6.0 * pred_w
    + 1.0 / 6.0 * pred_d
)
```

将预测写回与官方相同的 `247 × 245` 栅格：

- `pred_GVVI_S.tif`
- `pred_GVVI_W.tif`
- `pred_GVVI_D.tif`
- `pred_GVVI.tif`

GeoTIFF 的 transform、CRS、宽度和高度直接继承 `veg_raster_buffer_100.tif`。

## 8. 只报告这些结果

对测试节点分别报告 `GVVI_S/W/D` 和综合 GVVI：

- MAE；
- RMSE；
- R²；
- Pearson；
- Spearman。

汇报重点使用综合 GVVI：

1. 测试集散点图：官方值 vs GNN预测；
2. 官方 GVVI 地图；
3. GNN预测 GVVI 地图；
4. 绝对误差地图；
5. 一张指标表；
6. 训练时间与单次全图推理时间。

不做统计显著性、不做多随机种子、不做消融、不与其他模型比较。

## 9. 最低成功标准

这是时间受限的可行性 Demo，不预设论文级性能。最低可展示标准：

```text
综合 GVVI：
Spearman ≥ 0.70
R² ≥ 0.50
MAE ≤ 0.15
```

如果达到：

- 可以表述为“GNN具有同一研究区内快速补全 GVVI 的初步可行性”。

如果部分达到：

- 如排序相关高但绝对误差偏大，表述为“适合热点排序和方案预筛选，不替代精确计算”。

如果未达到：

- 不篡改数据划分；
- 不把训练集指标当测试集指标；
- 展示真实结果并说明廉价几何特征不足，后续需加入建筑遮挡或少量渲染特征。

## 10. DeepSeek 必须创建的目录与文件

```text
GVVI-GNN-Demo/
└── gvvi_grid_demo/
    ├── requirements.txt
    ├── run_all.py
    ├── config.yaml
    ├── src/
    │   ├── audit_targets.py
    │   ├── build_grid_features.py
    │   ├── build_grid_graph.py
    │   ├── split_nodes.py
    │   ├── model.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── export_geotiff.py
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
        └── final_demo_report.md
```

`run_all.py` 必须支持：

```bash
python run_all.py --data-root .. --device cuda
```

并按顺序完成：

```text
目标回放检查
→ 特征构建
→ 图构建
→ 固定划分
→ 训练
→ 测试评估
→ GeoTIFF导出
→ 图表和报告
```

## 11. 明天中午前的执行检查点

### 检查点 A：代码写完后立即本地冒烟测试

- 只读取前 500 个网格；
- 跑 3 个 epoch；
- 检查没有 NaN、索引错误和 CRS 错误；
- 提交并推送 GitHub。

### 检查点 B：GPU 全量首次运行

- 拉取最新代码；
- 运行 `nvidia-smi`；
- 安装依赖；
- 全量构建特征和图；
- 训练最多 300 epoch；
- 推送 `metrics.json`、报告和小型图表；
- checkpoint 和大体积中间文件不提交 Git。

### 检查点 C：结果验收

必须检查：

- 测试指标不是训练指标；
- 预测范围在 `[0,1]`；
- 官方回放一致；
- 四张预测栅格可打开；
- 地图没有转置、上下颠倒或行列交换；
- 报告包含运行硬件和真实耗时；
- 任何未运行结果不得写成已完成。

## 12. Git与GPU交互规范

GitHub 远程：

```text
git@github.com:The-AnonymousCoder/GVVI_GNN_Demo.git
```

GPU 服务器：

```text
host: connect.nmb1.seetacloud.com
port: 22754
user: root
```

密码只保存在仓库根目录的本地 `.gpu.env` 中，并由 `.gitignore` 排除。禁止把密码、token、私钥写入方案、代码、提交记录或远程仓库。

交互方式：

```text
本地修改并push
→ GPU服务器git pull
→ GPU运行
→ 只将指标、报告和图表commit并push
→ 本地git pull查看结果
```

不要提交：

- `.gpu.env`；
- Python缓存；
- 模型 checkpoint；
- 636 MiB 的旧候选边；
- 47 MiB 的旧正边；
- 新生成的大型 Parquet；
- 临时日志。

## 13. 最终汇报的一句话

如果实验达到最低标准：

> 我使用论文公开的三类观察点和官方 GVVI 真值，将城市绿化划分为 10 m 对象节点，以周围街道、窗口和无人机观察条件作为廉价特征，并利用两层 GraphSAGE传播相邻绿化单元的空间上下文，实现了研究区内未计算网格的多视角 GVVI 快速补全。

