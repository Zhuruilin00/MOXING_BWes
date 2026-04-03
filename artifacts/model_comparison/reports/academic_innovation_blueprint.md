# 学术创新投稿原型（面向 benchmark-thoughtful）

## 1. 研究问题与创新目标

在科创板短周期方向预测中，核心困难是：

- 标签噪声高（大量接近 0 的未来收益）
- 分布强非平稳（风格切换快）
- 样本有效窗口短（旧数据易失真）

本文目标不是提出更复杂模型，而是提出一种可迁移、可解释、可验证的稳健训练机制：

- 在不增加模型结构复杂度前提下，提高概率排序质量与高置信信号纯度。

## 2. 方法定义：RDSW（Robust Denoising + Short-window Weighting）

RDSW 由四个可复用模块组成，可插接于树模型或其他二分类器。

### 2.1 横截面去噪标签

对每个交易日 $t$、股票 $i$，定义未来 $h$ 日收益为 $r_{i,t}^{(h)}$。

先计算当日横截面绝对收益分位阈值：

$$
\tau_t(q)=\text{Quantile}_q\big(|r_{1,t}^{(h)}|,\dots,|r_{N_t,t}^{(h)}|\big)
$$

只保留 $|r_{i,t}^{(h)}|\ge \tau_t(q)$ 的样本，再定义方向标签：

$$
y_{i,t}=\mathbb{I}(r_{i,t}^{(h)}>0)
$$

其中 $q$ 为去噪强度（当前实证较优值约 $q=0.985$）。

### 2.2 时间衰减样本权重

设样本日期距离训练末端的天数为 $\Delta t$，半衰期为 $H$，时间权重：

$$
w_{\text{time}}=2^{-\Delta t/H}
$$

再叠加收益振幅权重（强调更有信息量样本）：

$$
w_{\text{amp}}=1+\min(\alpha\cdot |r_{i,t}^{(h)}|,\,b_{\max})
$$

最终权重：

$$
w_{i,t}=w_{\text{time}}\cdot w_{\text{amp}}
$$

### 2.3 近期训练窗截断

训练时仅使用最近 $W$ 个交易日样本，以减弱历史旧阶段对当前阶段的干扰。

### 2.4 阈值后校准

对验证期概率 $p_{i,t}$ 不直接用 0.5 固定阈值，而在网格上联合搜索：

- 决策阈值 $\theta_d$（分类）
- 置信阈值 $\theta_c$（高置信筛选）

最终按统一目标分数进行模型与阈值选择。

## 3. 可检验创新点（论文写法）

可将创新点写为三条：

1. 提出面向高噪声横截面方向预测的强去噪标签机制（按日分位裁剪）。
2. 提出面向非平稳市场的“时间衰减 + 近期窗截断”联合训练机制。
3. 提出“概率排序 + 高置信纯度”双目标后校准流程，统一于同一滚动验证框架中。

## 4. 与现有代码的对应关系

- 比较入口：[compare_models.py](../../compare_models.py)
- 同条件对比框架：[src/star_predictor/benchmark.py](../../src/star_predictor/benchmark.py)
- 估计器与权重、阈值逻辑：[src/star_predictor/model.py](../../src/star_predictor/model.py)
- 预设与训练配置：[src/star_predictor/pipeline.py](../../src/star_predictor/pipeline.py)

## 5. 必做实验设计（投稿最低配置）

### 5.1 消融实验

固定其余条件，仅改变一个模块：

- 去掉横截面去噪（或降低 $q$）
- 去掉时间衰减（$H\to\infty$）
- 去掉近期窗截断（$W=\text{all}$）
- 去掉阈值后校准（固定 0.5 / 0.6）

报告每项对以下指标的影响：

- accuracy
- balanced_accuracy
- AUC
- calibrated_high_conf_accuracy
- calibrated_coverage
- selection_score

### 5.2 统计显著性

基于折级指标做配对检验（建议 Wilcoxon 或 paired t-test），对比：

- RDSW vs RandomForest
- RDSW vs HistGradientBoosting
- RDSW vs LogisticRegression
- RDSW full vs RDSW ablation

给出 $p$ 值和效应方向。

### 5.3 稳健性

至少三组稳健性：

- 不同预测周期（$h=3,5$）
- 不同时间段切分（早期/中期/近期）
- 不同股票池规模（如 30 与更大样本）

## 6. 经济意义验证（建议）

用最小可交易检验增强说服力：

- 仅对高置信样本交易
- 固定换手约束与单边成本
- 报告净胜率、方向命中率与覆盖率变化

说明：即便非完整回测，也能显著提高论文可信度。

## 7. 论文结构建议

- 引言：高噪声+非平稳场景下，复杂模型不一定优
- 方法：RDSW 四模块（含公式）
- 实验：同条件对比 + 消融 + 显著性 + 稳健性
- 讨论：为何高置信纯度提升对实务更关键
- 结论：低复杂度模型通过训练机制创新可达竞争性能

## 8. 你当前结果的学术定位

基于现有结果文件 [summaries/benchmark_thoughtful/summary_h5.csv](../summaries/benchmark_thoughtful/summary_h5.csv)，你已经具备“方法雏形 + 明确实证增益”。

要达到“学术创新更稳投稿”，关键不在继续调 1-2 个百分点，而在补齐：

- 机制化消融证据
- 显著性证据
- 稳健性证据

完成这三项后，论文定位可从“工程优化报告”升级到“方法型实证研究”。
