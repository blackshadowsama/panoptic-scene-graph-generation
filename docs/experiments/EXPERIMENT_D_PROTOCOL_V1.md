# 实验D正式预注册方案

## DSFormer-FloodHN-FIBE：洪涝交互边界编码实验协议 V1.0

本方案汇总前面对实验D的多轮审查与修正。后续代码实现、训练、验证、消融和外部测试均严格遵循本协议。除非发现明确实现错误，否则不得在观察验证或测试结果后临时修改实验定义。

---

# 一、实验D的研究定位

实验D不是继续加强困难负样本采样，而是在实验C的基础上回答一个新的问题：

> 在模型已经充分接触洪涝困难负样本后，显式编码对象与水体之间的局部边界、软拓扑和垂直交互，能否进一步区分“视觉共现”和“真实洪涝关系”？

A、B、C、D分别回答不同问题：

| 实验                      | 核心变化            | 研究问题                     |
| ----------------------- | --------------- | ------------------------ |
| A：DSFormer-Original     | 原始均匀负采样         | 原DSFormer在FloodPSG上的基础表现 |
| B：DSFormer-Uniform-All  | 零关系图像参与、全图均匀负采样 | 完整负样本覆盖是否有效              |
| C：DSFormer-FloodHN      | 洪涝困难负样本定向采样     | 困难负采样是否优于普通均匀采样          |
| D：DSFormer-FloodHN-FIBE | C基础上增加边界表示      | 局部交互表示是否提供额外贡献           |

核心消融关系固定为：

[
B-A=\text{完整负样本覆盖的贡献}
]

[
C-B=\text{洪涝困难负采样的净贡献}
]

[
D-C=\text{洪涝交互边界表示的净贡献}
]

实验D不得改变实验C中的采样比例、训练图像、负样本预算、损失函数和模型选择规则。

---

# 二、实验D主要解决的问题

DSFormer现有mask编码主要表达：

* 整体mask覆盖哪些patch；
* 对象整体形状；
* subject/object位置；
* 类别语义；
* 全局相对空间关系。

但洪涝关系是否成立，通常取决于对象与水体之间的**局部交互区域**，例如：

| 困难场景    | 全局共现证据           | 真正需要判断的局部证据   |
| ------- | ---------------- | ------------- |
| 人站在河岸边  | person与water接近   | 脚部是否进入水体      |
| 人在洪水中涉水 | person与water重叠   | 下半部是否被水覆盖     |
| 车辆停在积水旁 | vehicle与water邻近  | 轮胎或底盘是否接触水    |
| 车辆被淹    | vehicle与water重叠  | 水体覆盖车体的高度     |
| 建筑邻水    | building与water邻近 | 建筑底部是否真正受水影响  |
| 桥梁跨河    | bridge与water共现   | 水是否只位于桥下      |
| 道路旁积水   | road与water邻近     | 水是否覆盖道路表面     |
| 人在游泳池   | person与water接触   | 场景是否为洪灾而非普通水体 |

因此，D主要解决：

1. 邻近但不接触；
2. 局部接触但不浸没；
3. 二维投影重叠但不存在真实洪涝交互；
4. 水体只位于目标侧面或背景；
5. 全局mask相似但下部交互不同；
6. DSFormer因类别共现而产生的关系过预测。

D不能单独解决：

* 游泳池与洪水的场景语义区别；
* 长尾关系类别；
* 错误目标检测和严重错误mask；
* 单目图像中的完整三维恢复；
* 候选对象对过多；
* 跨地域图像域偏移；
* 文本与图像语义冲突。

---

# 三、实验D的分阶段结构

实验D不一次性加入所有模块，而是分三级实施。

## D1：FIBE-Scalar

正式名称：

> DSFormer-FloodHN-FIBE-Scalar

结构：

```text
实验C
+
图像空间软拓扑标量特征
+
轻量边界MLP
+
受限门控残差融合
```

D1回答：

> 不增加局部CNN和深度模型，仅使用可解释的边界几何特征，是否能进一步降低困难负样本误报？

D1是当前必须完成的主实验。

---

## D2：FIBE-LocalMap

结构：

```text
最佳D1
+
4通道64×64局部交互图
+
轻量CNN
```

D2回答：

> 在标量特征之外，局部交互形状是否提供额外增益？

只有D1通过预先规定的验证门槛后，才允许开发D2。

---

## D3：FIBE-Depth

结构：

```text
最佳D2
+
单目相对深度或显式透视特征
```

D3回答：

> 外部相对深度线索是否进一步缓解投影重叠问题？

D3属于后续扩展，不是当前论文阶段必须完成的内容。

---

# 四、D1的核心理论定义

D1不声称恢复真实三维距离，也不声称从二维mask严格恢复真实地理拓扑。

正式使用术语：

> Soft Image-space Topological Descriptors
> 图像空间软拓扑描述子

这些特征描述：

* 二维图像中的软接触；
* 局部重叠；
* 边界邻近；
* 下部交互；
* 相对垂直位置；
* 表观水体边界。

论文中必须明确：

> 图像空间距离不等于真实地表距离；二维mask拓扑不等于真实三维地理拓扑；表观水体边界不一定是真实自由水面线。

---

# 五、表观水体边界的正式定义

原来的`waterline`统一改称：

> apparent water boundary
> 表观水体边界

原因是SAM水体mask的上边界可能表示：

* 水—道路交界；
* 水—建筑遮挡边界；
* 水—车辆遮挡边界；
* 水—背景交界；
* 分割模型产生的断裂边缘。

它不一定是真正的自由水面线。

正式特征命名为：

* `apparent_boundary_relative_height`
* `apparent_boundary_orientation_sin`
* `apparent_boundary_orientation_cos`
* `apparent_boundary_fit_residual`
* `apparent_boundary_valid`

表观边界方向应解释为：

> 局部透视、相机姿态、遮挡和水体空间状态共同作用下的图像代理线索。

不能解释为：

* 真实水面坡度；
* 相机roll的精确估计；
* 真实洪水深度；
* 水体固有物理形态。

---

# 六、Mask数据来源

## 1. 边界特征必须使用独立二值mask

D1优先使用现有的原始对象二值mask：

```text
overlapping binary masks
```

而不是只使用非重叠panoptic PNG。

原因是panoptic渲染会强制每个像素只属于一个对象，可能丢失：

* 水体覆盖道路的重叠；
* 水体覆盖车辆底部的区域；
* 建筑和水体的实际交互；
* 被遮挡对象的原始mask范围。

数据用途固定为：

| 数据            | 用途               |
| ------------- | ---------------- |
| 独立binary mask | D1边界和重叠特征        |
| panoptic PNG  | 原DSFormer训练和标准评估 |
| annotation关系  | 正负关系监督           |
| HN索引          | 实验C和D的采样         |

如果独立binary mask缺失，预处理脚本必须直接报错，不得静默回退到panoptic mask。

---

# 七、目标—危险源方向定义

FIBE内部将对象对统一转换为：

[
(\text{target},\text{hazard})
]

其中：

* target：人、车辆、道路、建筑、桥梁、排水设施、船等；
* hazard：water、mud、debris、barricade等。

方向必须根据**对象类别体系**确定，不能使用GT关系标签确定，否则会产生标签泄漏。

规则：

1. 对象对中恰好一个对象属于hazard类别时，自动规范为target–hazard；
2. 两个对象均不是hazard或均为hazard时，标记`fibe_pair_valid=0`；
3. 无效对象对的FIBE残差严格为0；
4. 原DSFormer的subject/object方向保持不变；
5. 规范化方向仅用于边界特征计算。

对于无效对象对：

[
g=0,\qquad r_b=0
]

因此模型退化为实验C，不会凭空增加噪声。

---

# 八、D1冻结特征体系

D1-v1固定为：

* 18个连续特征；
* 3个二进制质量标志；
* 总输入维度 (F=21)。

## 1. 重叠和软拓扑特征

| 编号 | 特征                                   |
| -: | ------------------------------------ |
|  1 | target overlap ratio                 |
|  2 | hazard overlap ratio                 |
|  3 | interior–interior overlap ratio      |
|  4 | boundary soft-contact ratio at (r_1) |
|  5 | boundary soft-contact ratio at (r_2) |
|  6 | target boundary inside hazard ratio  |
|  7 | hazard boundary inside target ratio  |

这些特征用于区分：

```text
Disjoint
→ Near
→ Soft Meet
→ Partial Overlap
→ Large Overlap
```

多尺度膨胀结果只能称为软接触或邻近，不能称为真实物理接触。

---

## 2. 距离特征

| 编号 | 特征                                |
| -: | --------------------------------- |
|  8 | minimum boundary distance         |
|  9 | boundary-distance 10th percentile |
| 10 | target-bottom-to-hazard distance  |

距离统一除以对象对union bbox对角线或target高度，不能使用原始像素距离。

这些特征仅解释为：

> normalized image-space distance

不能解释为真实三维距离。

---

## 3. 下部交互特征

| 编号 | 特征                            |
| -: | ----------------------------- |
| 11 | lower-25% overlap             |
| 12 | lower-50% overlap             |
| 13 | bottom-boundary contact ratio |
| 14 | contact relative height       |

其中：

[
\text{contact relative height}
==============================

\frac{
y_{\mathrm{contact}}-y_{\mathrm{target-top}}
}{
h_{\mathrm{target}}
}
]

它用于区分：

* 人脚部接触水体；
* 车辆底盘接触水体；
* 建筑底部受水影响；
* 水体仅位于目标上方或背景。

---

## 4. 表观水体边界特征

| 编号 | 特征                                   |
| -: | ------------------------------------ |
| 15 | apparent boundary relative height    |
| 16 | apparent boundary orientation sine   |
| 17 | apparent boundary orientation cosine |
| 18 | apparent boundary fitting residual   |

表观边界仅在局部边界像素数量和跨度足够时拟合。

拟合无效时：

```text
对应连续特征 = 0
apparent_boundary_valid = 0
```

---

## 5. 质量和有效性标志

| 编号 | 标志                        |
| -: | ------------------------- |
| 19 | apparent boundary valid   |
| 20 | mask/crop truncation flag |
| 21 | FIBE pair valid           |

`mask/crop truncation flag`用于指示目标或危险源是否接触图像边缘或局部crop边界。

---

# 九、软接触半径的确定

软接触半径不得依据验证指标或最终测试结果调整。

先在训练集执行边界质量人工审计，估计SAM边界误差：

* 中位误差；
* 90%分位误差；
* 相对于union bbox对角线的归一化误差。

然后固定：

[
r_1=\text{训练集边界误差中位数}
]

[
r_2=\text{训练集边界误差90%分位数}
]

实际像素半径需要裁剪到合理范围，例如：

[
2\le r_1\le 8
]

[
4\le r_2\le 16
]

一旦生成正式特征缓存，(r_1,r_2)不得再修改。

---

# 十、边界质量人工审计

在正式提取特征前，必须抽样人工检查独立binary mask的接触边界质量。

建议抽样约240个对象对，覆盖：

* person–water；
* vehicle–water；
* road–water；
* building–water；
* bridge–water；
* drain–water；
* boat–water；
* debris/mud–road；
* 正关系；
* 困难负关系；
* 大、中、小对象；
* 近邻、接触、重叠和遮挡场景。

长尾类别数量不足时，应审计全部可用样本。

人工记录：

```text
边界准确
轻微偏差
明显偏差
无法判断
```

并记录近似误差范围：

```text
0–2 px
3–5 px
6–10 px
>10 px
```

正式输出：

```text
fibe_mask_boundary_audit_v1.csv
fibe_mask_boundary_audit_summary_v1.json
```

边界质量审计只用于：

* 确定软接触容差；
* 识别特征适用性；
* 报告数据局限。

不得依据后续模型预测结果定向修正测试集mask。

---

# 十一、特征预处理

所有连续特征只使用训练集计算归一化统计量。

## 1. 稳健缩放

连续变量采用：

[
x'=
\frac{x-\operatorname{median}(x)}
{\operatorname{IQR}(x)+\epsilon}
]

然后裁剪：

[
x'\in[-5,5]
]

验证集、旧测试集和广西数据均使用训练集统计量，不得重新拟合。

## 2. 特征类型处理

比例特征：

```text
限制在[0,1]
```

距离特征：

```text
先按对象尺度归一化，再进行RobustScaler
```

角度特征：

```text
使用sin和cos，不直接使用角度值
```

无效特征：

```text
数值设0，同时设置valid flag
```

不得将NaN或Inf输入模型。

## 3. 不使用BatchNorm作为输入标准化

D1不依赖batch统计，因为每个batch的：

* 对象类别；
* 正负比例；
* HN比例；
* 对象尺度；

均可能变化。

使用固定训练集统计量与LayerNorm更加稳定。

---

# 十二、离线特征预处理流程

D1所有形态学和几何计算必须离线完成。

禁止在`Dataset.__getitem__()`中实时执行：

* 膨胀和腐蚀；
* 边界提取；
* 距离变换；
* 连通域分析；
* 表观水边界拟合；
* 多边形求交；
* 重复读取同一mask。

## 正确处理单位

预处理单位是“图像”，不是“对象对”。

对于每张图像：

```text
读取一次全部对象mask
        ↓
每个对象只计算一次bbox、边界、下部区域和距离图
        ↓
缓存对象级中间结果
        ↓
遍历该图所有合法有向对象对
        ↓
写入对象对特征张量
```

同一水体mask的距离变换只能计算一次，并被多个target复用。

---

# 十三、特征缓存格式

建议目录：

```text
data/floodpsg/features/fibe_scalar_v1/
├── train_features.pt
├── validation_features.pt
├── feature_schema.json
├── normalization_stats.json
├── extraction_config.json
├── extraction_audit.json
├── mask_source_sha256.txt
└── feature_cache_sha256.txt
```

每张图像保存：

```python
features_by_image[image_id] = {
    "features": Tensor[num_objects, num_objects, 21],
    "valid_pairs": BoolTensor[num_objects, num_objects],
}
```

训练时按：

```text
image_id
subject_index
object_index
```

直接O(1)索引。

缓存使用`float32`，不使用`float16`压缩D1标量特征。

训练启动时将全部D1缓存载入128GB系统内存，不从机械硬盘逐对象对读取。

---

# 十四、D1网络结构

## 1. 边界标量编码器

输入：

[
f_b\in\mathbb{R}^{21}
]

结构：

```text
21
→ Linear(21, 64)
→ GELU
→ LayerNorm(64)
→ Linear(64, 128)
→ GELU
→ LayerNorm(128)
```

得到：

[
z_b\in\mathbb{R}^{128}
]

FIBE分支不输入：

* target类别embedding；
* hazard类别embedding；
* predicate标签；
* 困难负样本标签；
* GT关系类型。

这样避免边界分支退化为类别先验放大器。

---

## 2. 边界残差投影

```text
128
→ Linear(128, 384)
→ LayerNorm(384)
```

得到：

[
r_b=\operatorname{LN}(P(z_b))
]

必须先标准化边界残差，再由gate和alpha控制注入强度。

不得使用：

[
\operatorname{LN}(\alpha r_b)
]

因为LayerNorm会削弱(\alpha)的尺度控制作用。

---

## 3. 门控

第一版使用标量gate：

```text
128
→ Linear(128, 1)
→ Sigmoid
```

[
g=\sigma(W_gz_b+b_g)
]

Gate只接收纯几何边界特征，不接收：

* 类别embedding；
* predicate；
* 完整DSFormer语义表示。

这样可以减少类别捷径学习。

初始化：

```python
nn.init.zeros_(boundary_gate.weight)
nn.init.constant_(boundary_gate.bias, -3.0)
```

初始：

[
g\approx\sigma(-3)\approx0.047
]

---

## 4. 受限残差缩放

定义：

[
\alpha=0.2\sigma(\text{raw_alpha})
]

初始化：

```python
raw_alpha = nn.Parameter(
    torch.tensor(-1.0986)
)
```

初始：

[
\alpha\approx0.05
]

融合公式：

[
\boxed{
z_{\mathrm{final}}
==================

z_{\mathrm{ds}}
+
\alpha g r_b
}
]

初始边界注入强度约为：

[
0.05\times0.047\approx0.00235
]

因此D1训练开始时接近实验C。

如果FIBE对象对无效：

[
g=0
]

模型严格退化为C。

除非原DSFormer对应位置本来存在LayerNorm，否则融合后不额外增加新的LayerNorm。

---

# 十五、类别感知设计

D1主实验不使用类别条件边界编码。

后续可选消融：

> D1-GC：类别乘性调制

先计算纯几何特征：

[
z_b=\phi(f_b)
]

类别只能产生乘性缩放：

[
\gamma=\tanh(W_\gamma[e_t,e_h])
]

[
\tilde z_b=
z_b\odot(1+\lambda\gamma)
]

限制：

* (W_\gamma)零初始化；
* (\lambda\le0.2)；
* 不允许类别生成加性偏置；
* 类别不得进入gate。

D1-GC只有在D1-G完成后才允许实施，不属于D1主实验。

---

# 十六、参数容量控制

为排除“D1只是因为增加参数而变好”的质疑，设置轻量容量对照：

> C-Adapter

C-Adapter：

* 使用与D1相近参数量的残差MLP；
* 输入只来自DSFormer已有对象对特征；
* 不输入任何边界几何；
* 使用相同的受限alpha和残差融合形式；
* 新增参数量与D1控制在±10%以内。

比较关系：

[
D1-C=\text{边界模块总体效果}
]

[
D1-C\text{-Adapter}
=\text{边界新信息相对单纯增容的贡献}
]

如果时间受限，至少必须报告：

* D1新增参数量；
* 总参数增长比例；
* 推理速度；
* 峰值显存；
* 每秒对象对数。

---

# 十七、训练初始化公平性

增加FIBE模块可能改变随机数消耗顺序，从而导致DSFormer主干初始化与C不同。

必须执行初始化审计：

1. 使用seed 3407分别实例化C和D1；
2. 确保先构造DSFormer主干，再构造FIBE；
3. 比较所有共享参数；
4. 共享参数最大绝对差必须为0；
5. 保存共享初始化审计结果。

输出：

```text
D1_SHARED_INITIALIZATION_AUDIT.json
```

若共享参数不一致，不得开始正式训练。

---

# 十八、采样公平性

D1必须完全复用C的FloodHN-target50采样逻辑。

固定：

* 相同训练图像；
* 相同正关系；
* 相同合法负对象对池；
* 相同负样本预算；
* 相同HN比例；
* 相同零关系图像处理；
* 相同seed；
* 相同epoch数。

优先复用C冻结的每轮负样本manifest。

若C没有保存完整manifest，则必须：

* 为sampler单独设置随机数生成器；
* 为DataLoader单独设置随机数生成器；
* 防止FIBE初始化消耗采样随机数；
* 审计前3个epoch中C与D1采样的对象对ID完全一致。

输出：

```text
D1_C_SAMPLING_EQUIVALENCE_AUDIT.json
```

D1与C唯一允许不同的因素是：

> D1增加了FIBE边界特征分支。

---

# 十九、正式训练协议

D1必须从头训练，不能从C的checkpoint继续微调。

| 项目             | 固定设置              |
| -------------- | ----------------- |
| 数据划分           | group-strict，与C一致 |
| 训练图像           | 1500              |
| 验证图像           | 173               |
| epoch          | 40                |
| seed           | 3407              |
| batch size     | 8                 |
| rels per batch | 128               |
| optimizer      | 与C一致              |
| learning rate  | 与C一致              |
| weight decay   | 与C一致              |
| node loss      | 与C一致              |
| relation loss  | 与C一致              |
| FloodHN比例      | target50          |
| checkpoint规则   | 验证mR@50最高         |
| 精度模式           | 与C一致              |
| 唯一新增           | FIBE-Scalar       |

不得单独为D1：

* 修改学习率；
* 增加epoch；
* 修改loss权重；
* 修改batch size；
* 修改HN比例；
* 修改每图负样本数；
* 改变checkpoint选择指标；
* 根据旧最终测试结果调整任何参数。

---

# 二十、工程与设备方案

当前设备：

* NVIDIA RTX 4090 D；
* 24GB专用显存；
* 128GB系统内存；
* C盘为NVMe SSD；
* D盘为SATA机械硬盘；
* WSL2环境。

## 1. 热数据位置

以下内容必须放在WSL ext4或NVMe：

* FIBE特征缓存；
* 当前annotation；
* 当前训练mask；
* 当前checkpoint；
* TensorBoard日志；
* 当前代码。

禁止训练时从：

```text
/mnt/d/...
```

逐对象对读取mask或局部特征。

D盘仅用于：

* 原始数据备份；
* 历史checkpoint；
* 最终结果归档；
* 大型不频繁访问文件。

## 2. DataLoader

初始配置：

```python
num_workers = 4
pin_memory = True
persistent_workers = True
prefetch_factor = 2
```

数据传输：

```python
tensor.to(
    device,
    non_blocking=True,
)
```

环境限制：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

不得默认使用16个worker。离线缓存后，worker过多只会增加调度开销。

## 3. GPU监测

使用：

```bash
watch -n 1 nvidia-smi
```

或：

```bash
nvidia-smi dmon -s pucm -d 1
```

重点观察：

* GPU-Util；
* 专用显存；
* 功耗；
* 温度；
* PCIe传输。

Windows任务管理器默认3D曲线不能代表CUDA训练利用率。

共享GPU内存不能视为额外CUDA显存。

---

# 二十一、工程Smoke Test

正式40轮训练前先运行1 epoch smoke test。

必须记录：

* 平均data time；
* 平均forward/backward time；
* 平均step time；
* GPU利用率；
* 峰值显存；
* 每秒对象对数；
* 特征缓存miss；
* NaN/Inf数量；
* 初始和最终gate均值；
* alpha值；
* FIBE梯度范数；
* DSFormer主干梯度范数。

通过条件：

```text
feature cache miss = 0
NaN/Inf = 0
peak VRAM < 22 GB
data_time / step_time < 15%
D1 step time相对C增加 < 10%
所有FIBE参数有有效梯度
共享初始化审计PASS
采样等价审计PASS
```

若`data_time / step_time > 20%`，先解决数据流问题，不得直接运行40轮。

---

# 二十二、验证指标与选择规则

实验D开发阶段只使用：

* 训练集；
* 验证集；
* 冻结验证困难负样本。

旧最终测试集不得参与D的结构和参数选择。

## 1. 标准关系指标

报告：

* R@20、R@50；
* mR@20、mR@50；
* NgR@20、NgR@50；
* mNgR@20、mNgR@50；
* 8类关系Recall；
* 候选级F1和ROC-AUC作为辅助指标。

## 2. 困难负样本指标

报告：

* FPR@0.5；
* FPR@0.7；
* FPR@0.9；
* mean NONE score；
* mean max-positive score；
* argmax relation rate；
* high-confidence FP count；
* pair-family FPR；
* scene-family FPR；
* 图像聚类bootstrap 95% CI；
* 精确McNemar检验。

## 3. 主要选择约束

D1必须满足：

[
mR@50_{D1}
\ge
mR@50_C-0.02
]

即标准mR@50下降不得超过2个百分点。

同时要求：

1. 水相关HN FPR@0.5低于C；
2. FPR@0.7和FPR@0.9方向一致；
3. 图像聚类bootstrap区间支持改善；
4. vehicle Recall@20不得出现明显额外下降；
5. human和building关系不得出现系统性退化。

---

# 二十三、D1进入D2的冻结门槛

只有满足以下全部条件，才进入D2：

1. 验证mR@50相对C下降不超过0.02；
2. 水相关HN FPR@0.5至少下降5个百分点；
3. FPR@0.7和FPR@0.9同方向下降；
4. 图像聚类bootstrap 95% CI上界低于0；
5. 几何置零后性能明显回退；
6. 分层几何错配后性能明显回退；
7. 改善不能仅由少量图像或单一类别贡献；
8. D1优于或至少不弱于C-Adapter容量对照。

D1未通过时：

```text
停止D2开发
→ 分析边界特征质量
→ 报告D1负结果
```

不得通过临时加入CNN、深度模型或修改loss掩盖D1无效。

---

# 二十四、反事实捷径学习审计

## 1. 几何置零

将验证集FIBE特征全部置零：

```text
正常D1
vs
boundary features = 0
```

预期：

* 标准关系指标可能轻微变化；
* HN FPR应明显回升。

若几乎无变化，说明FIBE未被模型实际使用。

---

## 2. 分层几何错配

不得在任意对象对之间完全随机交换。

交换层级固定为：

```text
相同对象对类别族
+
相同target尺度箱
+
相同union尺度箱
+
相同数据来源或事件
```

尺度依据：

* target bbox面积占图像比例；
* union bbox面积占图像比例；
* target高度占图像比例。

使用训练集或验证集分位数建立固定尺度箱。

两种错配方式：

### 分布保持错配

相同类别、尺度和事件内部任意置换几何特征。

用途：

> 检查模型是否依赖当前对象对自身的几何信息。

### 同标签错配

在相同类别、尺度、事件和正负标签内部置换。

用途：

> 排除因交换正负几何模式导致的简单分布破坏。

---

## 3. 同类别困难负样本评估

分别比较：

* person–water正关系与HN；
* vehicle–water正关系与HN；
* building–water正关系与HN；
* road–water正关系与HN；
* bridge–water正关系与HN。

正负对象对类别完全一致，因此类别先验无法完成判别。

D1若在这些子集中降低FPR，才能说明边界几何具有真正增量价值。

---

## 4. Gate审计

报告：

* 全体gate均值和分布；
* 正关系gate分布；
* HN gate分布；
* Near-No-Contact gate；
* Bottom-Contact gate；
* Large-Overlap gate；
* 无效FIBE pair gate；
* alpha训练轨迹。

不得仅凭gate较大或较小直接解释因果，必须结合置零和错配结果。

---

# 二十五、困难负样本空间类型分组

验证HN至少划分为：

| 类型                  | 定义                 |
| ------------------- | ------------------ |
| Near-No-Contact     | 距离近但无软接触和重叠        |
| Soft-Meet           | 存在边界软接触但内部重叠极小     |
| Side-Contact        | 接触集中在目标侧边          |
| Bottom-Contact      | 接触集中在目标底部          |
| Partial-Overlap     | 存在局部内部重叠           |
| Large-Overlap       | target较大比例进入hazard |
| Projection-Like     | 二维重叠但垂直与下部证据较弱     |
| Truncated/Uncertain | mask被图像或crop边缘截断   |

比较C和D1在各类型上的FPR。

D1应主要改善：

* Near-No-Contact；
* Soft-Meet；
* Projection-Like；
* Side-Contact。

若仅在Large-Overlap上改善，不能充分证明它解决了边界歧义。

---

# 二十六、D2的预注册设计

D2只有在D1通过后实施。

局部交互图固定为4通道：

1. target mask；
2. hazard mask；
3. target–hazard intersection；
4. normalized distance map。

处理：

```text
原始binary mask
→ union bbox裁剪
→ 适度扩边
→ 统一缩放到64×64
→ 轻量CNN
→ 128维局部交互向量
```

不直接把1～2像素边界压缩到8×8 patch。

D2缓存不得创建“一对象对一个文件”，应使用：

* 大块`.pt`分片；
* LMDB；
* Zarr；
* 或按图像分组缓存。

局部图必须离线生成。

---

# 二十七、D3的预注册限制

D3引入单目相对深度后，必须单独说明：

* 使用的外部预训练模型；
* 是否冻结；
* 深度模型训练数据；
* 参数量；
* 计算开销；
* 相对深度的尺度不确定性；
* 图像域偏移风险。

D3不得与D2同时首次加入，否则无法分离深度与LocalMap贡献。

---

# 二十八、最终测试协议

## 1. 原175张测试集

原测试集已经被用于A、B、C结果分析，因此对D只能作为：

> post-hoc supplementary benchmark
> 事后补充对比基准

不得用于：

* D1特征选择；
* 软接触半径选择；
* 网络结构选择；
* checkpoint选择；
* alpha调整；
* D2/D3进入决策。

## 2. 新外部测试集

D的严格最终泛化结论使用：

> 广西洪灾抖音/其他跨事件数据

外部测试必须在以下内容全部冻结后一次性运行：

* D版本；
* checkpoint；
* 归一化统计量；
* FIBE特征定义；
* HN阈值；
* 标准评估脚本；
* 关系类别；
* pair-family定义。

外部测试不得根据预测结果重新标注或定向修正mask。

---

# 二十九、代码与目录规划

建议建立独立分支：

```text
exp/dsformer-fibe-d1-v1
```

建议新增文件：

```text
scripts/audit_fibe_mask_boundaries_v1.py
scripts/build_fibe_scalar_features_v1.py
scripts/audit_fibe_feature_cache_v1.py
scripts/audit_fibe_initialization_v1.py
scripts/audit_fibe_sampling_equivalence_v1.py
scripts/evaluate_fibe_counterfactuals_v1.py
scripts/evaluate_fibe_validation_hn_v1.py

fair_psgg/data/fibe_features.py
fair_psgg/models/fibe_scalar.py
```

正式输出目录：

```text
outputs/
flood_groupstrict_coarse8_floodhn_fibe_scalar_seed3407_40epoch_v1
```

统计目录：

```text
data/floodpsg/stats/dsformer_fibe_d1_v1/
```

协议文件：

```text
EXPERIMENT_D_PROTOCOL_V1.md
```

---

# 三十、正式执行顺序

## 阶段0：冻结实验协议

```text
写入EXPERIMENT_D_PROTOCOL_V1.md
记录Git commit
记录A/B/C结果哈希
创建D独立分支
```

## 阶段1：Mask边界质量审计

```text
抽样约240对
人工核验边界误差
确定r1和r2
冻结审计结果
```

## 阶段2：20张图特征单元测试

检查：

* 特征方向；
* 数值范围；
* target/hazard规范化；
* 无效pair处理；
* 表观边界拟合；
* NaN/Inf；
* 可视化叠加。

## 阶段3：完整离线特征提取

```text
1500张训练图
173张验证图
按图像处理
生成train/validation缓存
计算训练集归一化统计量
保存manifest与SHA256
```

## 阶段4：数据和初始化审计

```text
缓存命中率审计
对象对索引审计
共享主干初始化审计
C/D1采样等价审计
```

## 阶段5：1 epoch smoke test

检查：

* 显存；
* GPU利用率；
* data time；
* step time；
* 梯度；
* gate；
* alpha；
* loss；
* NaN/Inf。

## 阶段6：D1正式训练

```text
seed 3407
40 epoch
从头训练
使用验证mR@50选择checkpoint
```

## 阶段7：D1验证

比较：

```text
C
C-Adapter
D1
```

报告标准关系指标与冻结HN指标。

## 阶段8：反事实审计

执行：

* 几何置零；
* 分层几何错配；
* 同标签错配；
* 同类别HN评估；
* gate与alpha审计。

## 阶段9：D1决策

```text
通过门槛
→ 进入D2

未通过
→ 停止结构叠加并分析原因
```

## 阶段10：外部跨事件测试

仅在最终模型与参数完全冻结后运行一次。

---

# 三十一、禁止事项

实验D期间禁止：

1. 覆盖A、B、C原模型和结果；
2. 从C checkpoint继续微调D；
3. 修改FloodHN-target50；
4. 修改负样本总预算；
5. 根据旧最终测试结果调整D；
6. 将类别embedding直接输入gate；
7. 在Dataset中实时做形态学运算；
8. 从D盘机械硬盘逐pair读取特征；
9. 创建十万个对象对小文件；
10. 使用验证集重新拟合归一化统计量；
11. 使用GT predicate确定target/hazard方向；
12. D1未通过就直接增加D2、D3；
13. 同时改变学习率、loss和网络结构；
14. 根据测试失败案例定向修改测试mask；
15. 将二维图像距离写成真实三维距离；
16. 将表观水体边界写成真实水深或真实水面坡度。

---

# 三十二、实验D最终成功标准

D1被认为有效，需要同时满足：

## 标准关系性能

[
\Delta mR@50 \ge -0.02
]

并且R@20、mR@20不存在明显系统性退化。

## 困难负样本性能

* 水相关HN FPR@0.5至少降低5个百分点；
* FPR@0.7和FPR@0.9同方向改善；
* bootstrap置信区间支持改善；
* high-confidence FP明显减少。

## 机制有效性

* 几何置零后性能回退；
* 分层错配后性能回退；
* 同类别HN中仍存在改善；
* 改善不能由类别先验解释；
* D1优于或不弱于容量控制。

## 工程性能

* 无缓存miss；
* 无NaN/Inf；
* 峰值显存低于22GB；
* DataLoader等待占比低于15%；
* 训练速度相对C下降不超过10%。

---

# 三十三、实验D的最终论文表述

实验D的方法贡献应表述为：

> 在洪涝感知困难负样本训练的基础上，本文进一步提出洪涝交互边界编码器FIBE。该模块从独立二值mask中提取图像空间软拓扑、边界邻近、局部重叠和目标下部交互特征，并通过受限门控残差机制注入DSFormer关系表示。FIBE不恢复真实三维距离或严格地理拓扑，而是显式建模二维图像中与洪涝关系判别相关的局部交互证据。

实验D的研究假设固定为：

> 当对象类别相同且全局共现模式相似时，局部边界、下部接触和软拓扑特征能够帮助模型区分真实洪涝关系与视觉共现困难负关系。

---

本协议自此冻结为：

```text
EXPERIMENT_D_PROTOCOL_V1
```

后续任何方法性修改必须在正式训练前形成新版本：

```text
EXPERIMENT_D_PROTOCOL_V2
```

并说明修改原因、修改时间和受影响实验。不得在观察最终测试结果后追溯修改V1协议。

