# 训练动力学 · Training Dynamics

> 理解深度网络在训练过程中参数、表示、损失如何随步数演化。

---

## 目录

- [背景：为什么需要训练动力学](#背景为什么需要训练动力学)
- [核心文献解读](#核心文献解读)
  - [1. Visualizing the Loss Landscape of Neural Nets (Li et al., 2018)](#1-visualizing-the-loss-landscape-of-neural-nets-li-et-al-2018)
- [文献索引](#文献索引)
- [阅读路线](#阅读路线)

---

## 背景：为什么需要训练动力学

深度学习在工程上大获成功，但理论理解严重滞后。以下三个矛盾驱动了这个领域的研究：

### 矛盾 1：理论说不可能，实际很好训

```
理论上：神经网络损失面 → 高维非凸 → 鞍点和小极小无穷多 → SGD 不应该收敛
实际上：SGD 每次都收敛到零损失，而且在测试集上泛化得不错
```

### 矛盾 2：Sharp vs Flat 极小值之争

小 batch SGD 倾向于找到 "flat" 极小值，泛化好；大 batch 则找到 "sharp" 极小值，泛化差（Keskar et al., 2017）。但 Dinh et al. (2017) 指出，由于网络的**尺度不变性**（scale invariance），sharp/flat 的测量本身就有问题——看起来 sharp 的极小值换个角度看就是 flat。

那 sharp 和泛化之间的关系到底是真是假？

### 矛盾 3：跳过连接为什么有效

ResNet 加上跳过连接就能训到 1000+ 层，普通 CNN 过了几十层梯度就炸了。简单的"梯度更好传"解释太粗糙，它对损失面的**几何形状**到底做了什么？

**训练动力学**就是为回答这些问题而生的——它不关心"训没训好"，而是研究"中间发生了什么、为什么"。

---

## 核心文献解读

### 1. Visualizing the Loss Landscape of Neural Nets (Li et al., 2018)

**Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein**  
*NIPS 2018* | [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)

#### 要回答的问题

损失面的几何形状如何影响泛化？跳过连接到底对损失面做了什么？

#### 核心困难：高维不能直接画

神经网络参数是几百万维的，画图必须降到 2D：

```
f(α, β) = L(θ* + α·d₁ + β·d₂)
```

经典的"取两个随机方向画等高线"方法有严重缺陷——**尺度不变性**让不同网络之间的比较毫无意义：

```
两个等价的网络 A 和 B：
  A: filter 权重很大，B: filter 权重很小
  (但因为 ReLU + BN，它们数学上等价)

在同一个随机方向 d 上扰动 1 个单位：
  A 看起来变化不大（因为它的权重以 10 为单位）
  B 看起来暴跳（因为它的权重以 0.1 为单位）

→ 不是 B 更 sharp，只是你选的扰动方向对 B 来说"步子"太大了
```

#### 核心方法：Filter Normalization

在画图之前，对随机方向向量做**按 filter 归一化**：

```
对每个 filter：
  d_filter ← d_filter × ||θ_filter|| / ||d_filter||

即：让扰动方向和该 filter 的实际权重同尺度
```

只有这样，不同网络、不同 minimizer 之间的 sharpness 才能做有意义比较。

**验证：** 归一化后 sharpness 与泛化误差强相关（R² > 0.9）；不归一化则是噪声。

#### 关键发现

**1. 跳过连接让损失面光滑**

```
Plain Net（无跳过连接）：
  10 层以内 → 近似凸
  20 层以上 → 突然变为"混沌"（chaotic）→ 训不动

ResNet（有跳过连接）：
  无论多深 → 损失面保持光滑
```

这是这篇论文最重要的发现之一。普通网络存在一个从"近似凸"到"混沌"的**相变点**，跳过连接把这个相变推到了极深处。

**2. Sharp minimizer 泛化差（用正确的测量方法）**

- 使用 filter normalization 后，sharp 极小值确实泛化更差
- Hessian 的负特征值（沟壑）数量与泛化误差正相关
- 大 batch 更容易收敛到 sharp 极小值

**3. SGD 轨迹落在低维空间**

- 虽然参数空间是百万维，但 SGD 的实际轨迹只需要约 5 个有效维度就能描述
- 解释了为什么 2D 投影能捕捉到真实结构

#### 对实践的启示

| 场景 | 启示 |
|------|------|
| 设计 U-Net 做生成 | 保留跳跃连接——它不只是梯度通道，还让损失面几何更友好 |
| 对比不同训练方法 | 用 filter normalization 画 loss landscape，客观比较 |
| 泛化差 | 量一下 Hessian 最大特征值，sharp 则考虑 SWA 平滑 |

---

## 文献索引

| # | 论文 | 年份 | 类别 |
|---|------|------|------|
| 1 | **Visualizing the Loss Landscape of Neural Nets** — Li et al. | 2018 | 入门可视化 |
| 2 | **Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability** — Cohen et al. | 2021 | 核心认知 |
| 3 | **Neural Tangent Kernel: Convergence and Generalization in Neural Networks** — Jacot et al. | 2018 | 理论基础 |
| 4 | **A Mathematical Framework for Learning** — Roberts et al. | 2022 | 全景框架 |
| 5 | **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets** — Power et al. | 2022 | 相变现象 |
| 6 | **Tensor Programs Vb: Scaling Limits and Wide Neural Networks (μP)** — Yang & Hu | 2021 | 实用工具 |
| 7 | **Stochastic Weight Averaging** — Izmailov et al. | 2018 | 工程技巧 |
| 8 | **Model Soups** — Wortsman et al. | 2022 | 工程技巧 |
| 9 | **Opening the Black Box of Deep Neural Networks via Information** — Tishby & Zaslavsky | 2017 | 信息论视角 |

---

## 阅读路线

```
入门 → 先看 Li et al. 的 PDF（可视化，最直观）
  ↓
核心 → Cohen et al. 的 Edge of Stability（训练动力学的里程碑）
  ↓
理论 → Jacot et al. 的 NTK（无限宽度极限，理解特征学习与记忆的边界）
  ↓
深入 → Power et al. 的 Grokking（训练中的相变现象）
  ↓
实用 → Izmailov et al. 的 SWA（权重平均免费提点）
  ↓
进阶 → Yang & Hu 的 μP（超参数跨规模迁移）
```

---

*Created: 2026-06-05*
