# 科创板短周期方向预测模型

这是一个面向科创板股票的短周期方向预测项目，目标是在真实市场数据上预测未来 3 天或 5 天的上涨/下跌方向，并尽量在高噪声、风格切换频繁、样本长度不均衡的环境里保持可解释、可复现、可迭代的效果。

项目的核心不是堆叠复杂模型，而是围绕科创板本身的交易特征，构建一套更适合这个市场的样本组织方式、标签去噪方式、横截面特征表达和滚动验证流程。当前代码已经固化了四组经过验证的强配置：

- `accuracy-over65`：优先提高平均 accuracy。
- `accuracy-balanced`：在保持较高 accuracy 的同时尽量抬高 balanced accuracy。
- `balanced-over65`：优先提高平均 balanced accuracy。
- `benchmark-best`：用于和热门模型对比时的当前最强折中配置。

此外，当前仓库还保留了一组更适合主结论展示的低复杂度最终配置：

- `benchmark-thoughtful`：当前推荐的单模型主结论配置，通过更强去噪、近期样本重权和更克制的树正则化，取得了目前最高的同条件 `selection_score`。

## 0. 项目使用说明（必读）

下面是从零到复现的最短路径，按顺序执行即可。

### 0.1 环境准备

```bash
pip install -r requirements.txt
```

### 0.2 训练与预测（常规使用）

训练 5 日模型：

```bash
python train.py --horizon 5 --dataset-path artifacts/star_history.csv
```

预测最新信号：

```bash
python predict.py --model-path artifacts/star_direction_h5.joblib --dataset-path artifacts/star_history.csv --top-k 20
```

### 0.3 推荐对比实验（主结论）

```bash
python compare_models.py --preset benchmark-thoughtful --dataset-path artifacts/star_history.csv
```

对比图表生成：

```bash
python visualize_comparison.py \
	--summary-csv artifacts/model_comparison/summaries/benchmark_thoughtful/summary_h5.csv \
	--folds-csv artifacts/model_comparison/summaries/benchmark_thoughtful/folds_h5.csv \
	--output-dir artifacts/model_comparison/charts/benchmark_thoughtful \
	--notes-language bilingual \
	--title "Benchmark Thoughtful"
```

### 0.4 学术投稿证据包（一键）

```bash
python run_publication_pack.py --dataset-path artifacts/star_history.csv --root-output artifacts/model_comparison/publication_pack
```

执行后优先查看：

- `artifacts/model_comparison/publication_pack/publication_readiness_report.md`
- `artifacts/model_comparison/publication_pack/ablation_main/ablation_summary.csv`
- `artifacts/model_comparison/publication_pack/significance_table.csv`

### 0.5 三个新增脚本作用

- `run_ablation.py`：运行主方法消融。
- `run_significance.py`：对折级结果做配对显著性检验。
- `run_publication_pack.py`：一键串联消融、稳健性、显著性与评估报告。

## 0.6 模型创新点与意义

为避免“只堆模型复杂度”的常见问题，本项目把创新集中在训练机制与验证机制上，而不是更复杂的网络结构。

### 创新点 1：横截面强去噪标签

不是直接用“未来收益是否大于 0”打标签，而是按交易日横截面先过滤绝对收益过小样本，仅保留方向信息更明确的样本再做二分类。

意义：

- 降低短周期任务中接近随机扰动样本对模型的污染。
- 提高监督信号质量，使模型更聚焦于可学习的方向模式。

### 创新点 2：时间衰减 + 近期训练窗截断

训练样本随时间衰减赋权，同时限制最大训练窗口，减少过旧样本对当前市场阶段的干扰。

意义：

- 面向非平稳市场的实用稳健化设计。
- 不改变模型结构即可提升阶段适配能力。

### 创新点 3：后验阈值双层校准

在滚动验证概率输出上联合搜索决策阈值与高置信阈值，不固定使用 0.5。

意义：

- 同时兼顾整体分类效果与高置信信号纯度。
- 更贴近实务中“可交易信号筛选”的使用方式。

### 创新点 4：低复杂度前提下的系统化证据链

项目提供统一的同条件模型对比、消融实验、稳健性检查与显著性检验脚本。

意义：

- 把创新从“参数偶然最优”提升为“可复现方法结论”。
- 为论文写作提供完整证据路径（方法、实验、统计检验）。

### 综合意义

本项目的核心贡献可以概括为：

- 方法层面：提出一套面向高噪声、非平稳股票短周期任务的低复杂度稳健训练框架。
- 工程层面：实现从数据、训练、对比、可视化到投稿证据包的一体化可复现流程。
- 应用层面：在不引入复杂模型结构的前提下，获得有竞争力的综合指标与高置信信号质量。

## 1. 建模背景

科创板与主板、中小盘宽基数据集有明显差异，直接照搬常见选股模型往往效果不稳定，主要原因有四类：

1. 新股多、上市年龄差异大，单只股票历史长度不一致。
2. 波动率高，短周期收益噪声大，小幅涨跌经常接近随机扰动。
3. 题材轮动快，市场风格切换频繁，旧样本对新阶段经常失真。
4. 若按单票建模，样本量往往不足；若简单拼接全市场，又容易受量纲、价格区间和上市阶段差异干扰。

因此本项目采用的是“全市场共享样本 + 强去噪标签 + 横截面秩特征 + 时间滚动验证”的思路，而不是单票黑盒预测。

## 2. 模型解决什么问题

这个模型解决的是“未来若干个交易日方向判断”问题，而不是收益率点预测问题。它主要适合以下场景：

- 生成未来 3 日或 5 日的上涨/下跌方向信号。
- 作为选股排序器中的方向过滤层。
- 作为交易研究中的高置信度候选池生成器。
- 用于比较不同标签去噪强度、不同滚动窗口、不同科创因子组合对短周期稳定性的影响。

它不直接解决以下问题：

- 仓位管理。
- 交易成本与滑点控制。
- 涨跌停约束处理。
- 实盘级风控与组合优化。

## 3. 整体结构

当前项目结构如下：

- `train.py`：训练入口，支持普通训练与预设训练。
- `predict.py`：加载模型，对最新交易日生成方向信号。
- `src/star_predictor/data.py`：下载科创板股票池与历史行情。
- `src/star_predictor/features.py`：构建时序特征、板块特征、科创扩展特征与标签。
- `src/star_predictor/sci_factors.py`：构建研发、解禁、资金流、公告事件、分钟微观结构等科创扩展因子。
- `src/star_predictor/model.py`：定义 ExtraTrees 模型、滚动验证、样本权重、阈值校准和指标汇总。
- `src/star_predictor/pipeline.py`：组织训练流程、自动调参、模型保存与预测流程。
- `artifacts/`：数据缓存、验证结果、模型文件和实验输出。

## 4. 模型建构机制

### 4.1 数据来源

项目通过 `akshare` 抓取真实公开市场数据，主要包括：

- 科创板股票池：A 股股票代码与名称。
- 个股日线：前复权 OHLCV、成交额、换手率等行情。
- 财务与科创属性：研发费用、研发费用率、上市时长、解禁信息。
- 资金与事件：主力净流入、超大单/大单净流入、公告数量、公告风险词/正向词。
- 其他扩展项：研报代理、专利关键词代理、分钟级微观结构因子。

其中日线行情是主干数据，科创扩展因子是增强层。模型允许只用纯行情特征训练，也允许合并扩展因子训练。

### 4.2 标签定义

标签不是简单“未来涨跌是否大于 0”后直接训练，而是先做按日横截面去噪：

1. 计算未来 `horizon` 日收益率 `future_return`。
2. 对每个交易日，在横截面上计算未来收益绝对值的分位阈值。
3. 剔除绝对收益过小的样本，只保留更“有方向性”的样本。
4. 对保留样本按收益正负打 0/1 标签。

这一层是本项目最重要的稳健化处理之一。最近几轮实验也证明，若目标是提高平均 accuracy 或 balanced accuracy，继续增强去噪强度通常比继续加深树模型更有效。

### 4.3 特征体系

模型主要使用三大类特征。

#### 个股时序特征

- 1/2/3/5/10/15/20 日收益率。
- 隔夜跳空、日内收益、振幅、实体占振幅比。
- 成交量、成交额、换手率相对历史均值的放大倍数。
- 5/10 日波动率。
- 5/10/20 日均线偏离。
- 20 日回撤。
- 5/10 日趋势强度与上涨天数占比。

#### 板块环境特征

- 当日科创板横截面收益中位数。
- 5 日平均收益。
- 上涨家数占比。
- 横截面离散度。
- 板块换手中位数。
- 板块波动率中位数。

#### 科创扩展因子

- 上市时长。
- 研发费用、研发费用率。
- 下一次解禁剩余天数、解禁占总市值比例。
- 近 180 天研报数量、专利关键词代理值。
- 主力/超大单/大单净流入占比。
- 30 日公告数量。
- 90 日公告风险词强度、正向词强度。
- 分钟级微观结构因子。

其中大部分个股特征会进一步转成“按交易日做横截面分位秩”，这是本项目区别于很多直接用绝对技术指标建模的关键点。这样做有两个好处：

1. 降低不同价格区间、不同波动水平股票之间的量纲差异。
2. 让模型更关注相对强弱，而不是绝对数值。

### 4.4 模型主体

当前主模型是 `ExtraTreesClassifier`，并配有中位数缺失值填补。它不是深度网络，也不是堆叠集成，而是一种更偏稳健、样本需求更低的树模型方案。

为什么这里优先用 ExtraTrees：

- 对非线性特征和缺失值后的结构较友好。
- 对横截面秩特征适配度高。
- 不依赖特征标准化。
- 在中小样本、高噪声条件下，比复杂神经网络更容易稳定复现。
- 训练成本可控，便于频繁滚动重训和多组配置实验。

### 4.5 分层建模

模型支持“新股层”和“成熟股层”的分层训练逻辑：

- 若 `listing_age_days <= newborn_days`，样本归入新股层。
- 其余样本归入成熟股层。
- 若某层样本量不足，则回退到全样本模型。

这部分是针对科创板上市阶段差异设计的。因为新股和成熟股的行为模式往往不一致，统一模型容易被混淆。

### 4.6 时间权重与近期窗口

训练时会对近期样本赋予更高权重，权重随样本年龄按半衰期衰减。同时，自动调参阶段还支持限制只保留最近一段训练窗口，避免太早的数据拖累当前市场阶段。

这部分机制的目标是解决非平稳性：

- 老样本不是没用，但不应该和最新样本权重完全相同。
- 在风格切换显著时，短训练窗常常比全历史更有效。

### 4.7 阈值校准

训练完成前，模型会基于滚动验证产生的概率结果做阈值校准，而不是简单固定 `0.5` 决策阈值。

- `selection_objective=accuracy` 时，阈值按验证集 accuracy 搜索。
- `selection_objective=joint` 时，阈值按验证集 accuracy 与 balanced accuracy 的联合得分搜索。
- `selection_objective=balanced_accuracy` 时，阈值按验证集 balanced accuracy 搜索。
- 同时还会校准高置信度筛选阈值，用于生成更高质量的信号子集。

这意味着本项目的训练结果不是只有一个树模型，还包含一组经过验证期标定的判定规则。

## 5. 训练与预测流程

### 5.1 训练流程

训练流程可以概括为：

1. 下载或加载历史数据。
2. 构造科创扩展因子并与行情合并。
3. 生成特征与去噪标签。
4. 按时间滚动切分训练/验证窗口。
5. 对每组候选配置进行 walk-forward 验证。
6. 用验证期概率结果校准决策阈值与高置信阈值。
7. 选择最优配置，在全训练样本上拟合最终模型。
8. 输出模型、滚动验证指标、逐笔预测结果。

### 5.2 预测流程

预测时会：

1. 读取模型包。
2. 构造最新交易日的特征。
3. 输出上涨概率、方向预测、置信度。
4. 根据训练期校准得到的阈值筛选高置信信号。

## 6. 验证方法

本项目不使用随机切分，而是使用严格的时间滚动验证。原因很直接：

- 随机切分会产生未来信息泄漏。
- 短周期交易模型对时间顺序非常敏感。
- 真正需要验证的是“在过去训练、在未来验证”的稳定性。

当前验证方式是：

- 先用一段时间作为训练窗口。
- 留出 `gap_days` 防止近邻泄漏。
- 用后续一段时间作为验证窗口。
- 继续向前滚动，直到样本结束。

输出的核心验证文件包括：

- `cv_metrics_h*.csv`：每个滚动折的指标。
- `cv_predictions_h*.csv`：每条验证样本的预测概率与方向。

## 7. 当前达到的效果

下面是当前代码库里已经验证并保存下来的几组关键结果。为了可复现，这些结果都基于本地 `artifacts/star_history.csv` 和当前缓存因子文件运行得到。

### 7.1 原始基线

默认 5 日模型的历史基线大致为：

- mean accuracy：53.78%
- mean balanced accuracy：52.87%
- mean high-conf accuracy：55.92%
- mean coverage：25.81%

### 7.2 accuracy 优先配置

`accuracy-over65` 预设对应的是一组更激进的 accuracy 导向配置。当前已验证结果约为：

- mean accuracy：66.08%
- mean balanced accuracy：55.59%
- mean AUC：67.85%
- high-conf accuracy：70.87%
- coverage：71.56%

这一组更适合“尽量提高命中率”的目标。

### 7.3 accuracy / balanced 折中配置

`accuracy-balanced` 预设对应的是一组兼顾双指标的折中配置。当前已验证结果约为：

- mean accuracy：65.56%
- mean balanced accuracy：58.47%
- mean AUC：63.25%
- high-conf accuracy：69.77%
- coverage：68.62%

这一组更适合“accuracy 仍然要保持在 65% 左右，但不希望 balanced accuracy 压得太低”的目标。

### 7.4 balanced accuracy 优先配置

`balanced-over65` 预设对应的是一组更激进的平衡型配置。当前已验证结果约为：

- mean balanced accuracy：65.45%
- mean accuracy：63.63%
- mean AUC：70.39%
- high-conf accuracy：68.95%
- coverage：52.03%

这一组更适合“希望上涨和下跌两边都更均衡”的目标。

### 7.5 科创扩展因子的增益

在现有对照实验中，启用科创扩展因子相对于只用纯行情特征，带来的是小幅但稳定的提升，大致表现为：

- accuracy 提升约 1 个百分点。
- balanced accuracy 提升约 0.8 个百分点。

这说明科创因子不是决定性飞跃来源，但确实提供了增量信息，尤其是：

- 解禁相关特征。
- 公告正向词强度。
- 研发费用与研发费用率。
- 公告数量。

### 7.6 benchmark-thoughtful（当前推荐主结论）

`benchmark-thoughtful` 是当前最推荐用于主结论展示的低复杂度配置。它仍然是单个 `ExtraTreesClassifier`，没有使用深度模型，也没有使用复杂堆叠；提升主要来自三类克制但有效的改动：

- 更强一点的横截面去噪：`neutral_quantile=0.985`
- 更贴近当前阶段的时间权重：`sample_weight_halflife_days=240`，`max_train_days=260`
- 更克制的树复杂度控制：`520` 棵树、`max_depth=10`、`min_samples_split=50`、`min_samples_leaf=18`、`max_features=0.48`

基于最新保留文件 `artifacts/model_comparison/summaries/benchmark_thoughtful/summary_h5.csv`，当前 ExtraTrees（本项目）核心指标为：

- selection score：0.6598
- mean accuracy：62.92%
- mean balanced accuracy：64.12%
- mean AUC：70.24%
- calibrated high-conf accuracy：86.79%
- calibrated coverage：12.27%

这一组结果说明，当前最优提升并不是靠增加结构复杂度，而是靠更合理的标签强度、时间衰减和树正则化共同作用。

### 7.7 benchmark-best（上一版对比基准）

`benchmark-best` 是当前用于“和热门模型同条件对比”的统一配置。基于最新保留文件 `artifacts/model_comparison/summaries/benchmark_best/summary_h5.csv`，当前 ExtraTrees（本项目）核心指标为：

- selection score：0.6352
- mean accuracy：63.63%
- mean balanced accuracy：65.45%
- mean AUC：70.39%
- high-conf accuracy：68.95%
- coverage：52.03%

### 7.8 与热门模型的最新同条件对比

同一份滚动验证、同一份特征和阈值校准逻辑下（preset=`benchmark-thoughtful`）的最新结果：

- ExtraTrees (current preset)：accuracy 62.92%，balanced accuracy 64.12%，AUC 70.24%，selection score 0.6598
- RandomForestClassifier：accuracy 64.35%，balanced accuracy 64.65%，AUC 68.36%，selection score 0.6405
- HistGradientBoostingClassifier：accuracy 62.85%，balanced accuracy 59.29%，AUC 65.53%，selection score 0.6189
- LogisticRegression：accuracy 57.47%，balanced accuracy 58.71%，AUC 65.16%，selection score 0.5964

当前版本下，ExtraTrees 在 `selection_score` 和 `AUC` 上保持领先；RandomForest 的原始 `accuracy` 与 `balanced_accuracy` 略高，但综合阈值质量与高置信信号纯度后，最终仍是 `benchmark-thoughtful` 更强。

## 8. 与其他模型方案相比的优势

这里的“对比”分为两种：一种是结构性优势，一种是代码库内已经验证的经验优势。除结构比较外，仓库内已提供可重复执行的同条件模型对比脚本与可视化脚本，可直接复核结论。

### 8.1 相比单票模型

优势：

- 解决单票样本不足问题。
- 共享跨股票行为模式。
- 对上市时间短的新股更友好。

代价：

- 牺牲一部分个股特异性。

### 8.2 相比随机切分模型

优势：

- 验证更贴近真实未来预测场景。
- 避免未来信息泄漏导致的虚高结果。

代价：

- 指标往往更难看，但更可信。

### 8.3 相比简单逻辑回归或线性模型

优势：

- 能更自然处理非线性特征关系。
- 对横截面秩特征和事件型因子适配更好。
- 无需特征标准化。

代价：

- 可解释性不如纯线性权重直观。

### 8.4 相比深度学习或复杂堆叠模型

优势：

- 对样本量要求更低。
- 训练成本更低，易于频繁重训。
- 实验迭代速度更快。
- 更容易把增益归因到标签、特征和窗口，而不是黑盒结构本身。

代价：

- 理论上表达能力不如复杂神经网络。
- 在极大规模数据或高频序列场景下未必最优。

### 8.5 相比“只调模型不调标签”的方案

本项目的经验非常明确：在当前数据结构下，继续增强去噪通常比盲目加深树更有效。最近的 over65 结果也主要来自：

- 更激进的 `neutral_quantile`。
- 更贴近目标指标的选模目标。
- 更合理的验证阈值校准。

## 9. 如何直接复现这些效果

如果你的目标非常明确，可以直接按下面几组命令运行。它们分别对应不同的优化方向：

- `accuracy-over65`：尽量把平均 accuracy 推高。
- `accuracy-balanced`：在保持 accuracy 处于 65% 左右的同时，尽量抬高 balanced accuracy。
- `balanced-over65`：优先把平均 balanced accuracy 推到 65% 以上。
- `benchmark-thoughtful`：当前推荐的低复杂度主结论配置，优先兼顾同条件对比下的综合表现。

### 9.1 复现 accuracy-over65

```bash
python train.py --preset accuracy-over65 --dataset-path artifacts/star_history.csv
```

预期效果约为：

- mean accuracy：66.08%
- mean balanced accuracy：55.59%

### 9.2 复现 accuracy-balanced

```bash
python train.py --preset accuracy-balanced --dataset-path artifacts/star_history.csv
```

预期效果约为：

- mean accuracy：65.56%
- mean balanced accuracy：58.47%

### 9.3 复现 balanced-over65

```bash
python train.py --preset balanced-over65 --dataset-path artifacts/star_history.csv
```

预期效果约为：

- mean accuracy：63.63%
- mean balanced accuracy：65.45%

### 9.4 复现 benchmark-thoughtful

```bash
python compare_models.py --preset benchmark-thoughtful --dataset-path artifacts/star_history.csv
```

预期效果约为：

- selection score：0.6598
- mean accuracy：62.92%
- mean balanced accuracy：64.12%
- mean AUC：70.24%
- calibrated high-conf accuracy：86.79%

这些预设会：

- 使用已经验证过的参数组合。
- 复用当前 `artifacts/sci_factor_cache.csv`。
- 避免因为重新补刷不完整缓存而改变结果行为。
- 使用本地 `artifacts/star_history.csv` 以尽量复现当前结果。

## 10. 常用命令

安装依赖：

```bash
pip install -r requirements.txt
```

一键生成投稿证据包：

```bash
python run_publication_pack.py --dataset-path artifacts/star_history.csv --root-output artifacts/model_comparison/publication_pack
```

训练 5 日模型：

```bash
python train.py --horizon 5
```

训练 3 日模型：

```bash
python train.py --horizon 3
```

只使用纯行情特征：

```bash
python train.py --horizon 5 --disable-sci-factors
```

优先提高平均 accuracy：

```bash
python train.py --horizon 5 --selection-objective accuracy
```

兼顾 accuracy 与 balanced accuracy：

```bash
python train.py --horizon 5 --selection-objective joint
```

优先提高平均 balanced accuracy：

```bash
python train.py --horizon 5 --selection-objective balanced_accuracy
```

预测最新信号：

```bash
python predict.py --model-path artifacts/star_direction_h5.joblib --top-k 20
```

对比当前模型与热门模型效果：

```bash
python compare_models.py --preset accuracy-balanced --dataset-path artifacts/star_history.csv
```

对比 accuracy 导向配置与热门模型：

```bash
python compare_models.py --preset accuracy-over65 --dataset-path artifacts/star_history.csv
```

对比 balanced 导向配置与热门模型：

```bash
python compare_models.py --preset balanced-over65 --dataset-path artifacts/star_history.csv
```

对比 benchmark-best（推荐用于当前主结论）：

```bash
python compare_models.py --preset benchmark-best --dataset-path artifacts/star_history.csv
```

对比 benchmark-thoughtful（当前推荐用于主结论）：

```bash
python compare_models.py --preset benchmark-thoughtful --dataset-path artifacts/star_history.csv
```

把对比结果生成图表（柱状图 + 折线图）：

```bash
python visualize_comparison.py \
	--summary-csv artifacts/model_comparison/summaries/benchmark_best/summary_h5.csv \
	--folds-csv artifacts/model_comparison/summaries/benchmark_best/folds_h5.csv \
	--output-dir artifacts/model_comparison/charts \
	--notes-language bilingual \
	--title "Benchmark Best (H5)"
```

默认会同时生成 `comparison_chart_notes.md`，其中包含每张图的用途、解读重点和决策建议。
默认还会生成 `paper_figure_captions.md`，可直接作为论文图注初稿。
如只想导出图片可加 `--no-notes --no-paper-captions`。

为最终 thoughtful 结果生成单独图表目录：

```bash
python visualize_comparison.py \
	--summary-csv artifacts/model_comparison/summaries/benchmark_thoughtful/summary_h5.csv \
	--folds-csv artifacts/model_comparison/summaries/benchmark_thoughtful/folds_h5.csv \
	--output-dir artifacts/model_comparison/charts/benchmark_thoughtful \
	--notes-language bilingual \
	--title "Benchmark Thoughtful"
```

### 10.1 如何解读图表

生成图表后，建议按下面顺序阅读：

1. 先看 `comparison_main_metrics.png`：
	- 重点比较 `selection_score`、`accuracy`、`balanced_accuracy`、`auc` 四项。
	- 若你的目标是实盘方向命中优先，可先看 `accuracy` 与 `selection_score`。
	- 若你更关注涨跌两类样本的均衡识别，优先看 `balanced_accuracy`。
	- `auc` 反映概率排序能力，通常用于判断模型区分度是否稳定。

2. 再看 `comparison_thresholds_and_coverage.png`：
	- `decision_threshold` 越高，通常意味着更保守的“看涨判定”。
	- `confidence_threshold` 越高，高置信信号会更少但通常更纯。
	- `coverage` 表示高置信样本覆盖率，过低会导致可交易信号不足，过高则可能稀释质量。

3. 最后看 `comparison_fold_accuracy_trend.png`：
	- 观察各模型在不同时间折上的波动幅度。
	- 平均值接近时，优先选曲线更平滑、回撤更小的模型。
	- 若某模型在最新几个折明显走弱，说明它可能对当前市场阶段适配下降。

4. 实际使用时的一个简化准则：
	- 主模型优先级：先看 `selection_score`，再看 `balanced_accuracy` 与 `accuracy` 的共同水平。
	- 阈值策略优先级：在可接受的 `coverage` 下，优先选择能稳定抬高高置信准确率的阈值组合。

## 11. 输出文件说明

训练后常见输出包括：

- `artifacts/star_history.csv`：历史行情缓存。
- `artifacts/sci_factor_cache.csv`：科创扩展因子缓存。
- `artifacts/star_direction_h3.joblib` / `artifacts/star_direction_h5.joblib`：模型包。
- `artifacts/cv_metrics_h*.csv`：逐折指标。
- `artifacts/cv_predictions_h*.csv`：逐笔验证预测。
- `artifacts/archive/runs/accuracy_over65/`：accuracy 导向 over65 历史结果归档。
- `artifacts/archive/runs/accuracy_balanced/`：accuracy / balanced 折中历史结果归档。
- `artifacts/archive/runs/balanced_over65/`：balanced 导向 over65 历史结果归档。
- `artifacts/archive/runs/`：其余历史实验目录归档（含旧 benchmark 与实验网格结果）。
- `artifacts/archive/datasets/`：历史数据快照归档（如 `star_history_5.csv`、`star_history_30.csv`）。
- `artifacts/model_comparison/summaries/benchmark_best/`：benchmark-best 的汇总、逐折、逐笔结果。
- `artifacts/model_comparison/summaries/benchmark_thoughtful/`：benchmark-thoughtful 的汇总、逐折、逐笔结果。
- `artifacts/model_comparison/charts/benchmark_thoughtful/`：当前最终 thoughtful 结果对应的图表、双语说明和论文图注模板。
- `artifacts/model_comparison/reports/academic_innovation_blueprint.md`：学术创新投稿蓝图。
- `artifacts/model_comparison/reports/paper_method_section_draft.md`：论文方法章节初稿。
- `artifacts/model_comparison/reports/paper_experiment_template.md`：实验章节与表格模板。
- `artifacts/model_comparison/publication_pack/`：一键投稿证据包输出目录。

## 12. 注意事项

1. 这是方向预测模型，不是收益率点预测模型。
2. 股票市场是非平稳系统，结果会随重训时点、数据样本和缓存因子状态发生变化。
3. `accuracy-over65`、`accuracy-balanced`、`balanced-over65` 和 `benchmark-thoughtful` 是当前数据缓存条件下验证过的强配置，不代表未来任意时间段都能保持同样结果。
4. 若重新抓取最新数据或刷新科创因子缓存，结果可能与当前 README 中记录的数值不同。
5. 若用于实盘，必须额外引入交易成本、滑点、停牌、涨跌停约束和风险控制模块。