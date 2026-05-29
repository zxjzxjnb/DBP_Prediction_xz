# 数据结构如何改变模型预测驱动模式

## 核心结论

两个数据集出现完全不同的 SHAP 驱动模式，并不只是“模型换了”或“样本量变了”，而是输入数据本身的统计结构发生了根本变化。

更准确地说，模型学到的不是一个抽象、跨数据集不变的化学规律，而是在各自数据分布内最能解释目标波动的预测坐标。Tai Lake 小数据集里，Temperature 是最宽、最有季节性、最能代理反应动力学和运行季节变化的坐标；Dataset1 里，Temperature 的有效变异被压缩，而 TOC、Bromide、Cl2 dose 等更直接、更宽、更异质的变量接管了预测解释权。

因此，驱动模式的差异可以概括为：

- Tai Lake：模型主要沿着“季节/温度轴”预测 DBP。
- Dataset1：模型主要沿着“水源/前体物/加氯运行轴”预测 DBP。

## SHAP 证据对齐

已有 SHAP 结果显示，Tai Lake 5-common 条件 A' 中，Temperature 在 Random Forest 和 XGBoost 下对 THM4、DBCM、BDCM 都是 rank #1。

Dataset1 的对照实验则相反。即使把 Dataset1 下采样到和 Tai Lake 一样的 175 行，并且只保留 5 个共同输入，Temperature 仍然没有恢复到主导地位：

- Dataset1 5-common 下采样 D：THM4 主要由 TOC / UV254 / Bromide 驱动，Temperature 常在 rank 4-5。
- Dataset1 5-common 下采样 D：DBCM 和 BDCM 主要由 Bromide 驱动。
- Dataset1 6-feature 下采样 E：Cl2 dose 进入后成为 THM4 的主驱动，但 Temperature 在加入 Cl2 之前就已经不是主驱动。
- Dataset1 7-feature formal best：THM4 的 top drivers 是 Cl2 dose、TOC、Bromide；DBCM/BDCM 主要由 Bromide、Cl2 dose、UV254/TOC 驱动。

这说明样本量不是主因；Cl2 解释了一部分 THM4 排名变化，但不是 Temperature 失去主导地位的根因。根因更接近数据生成结构和输入分布结构的变化。

## 机制 1：Temperature 在 Tai Lake 中是强预测坐标，在 Dataset1 中被“范围压缩”

Temperature 是两个数据集结构差异最关键的例子。

Tai Lake 的 Temperature 分布更宽：std = 8.33，IQR = 14.00。Dataset1 的 Temperature 分布更窄：std = 3.44，IQR = 3.68，只有 Tai Lake std 的 41%。

这会直接改变模型能学到的东西。树模型或神经网络要利用一个变量，前提是这个变量在训练数据里能把目标值分开。Tai Lake 的温度跨度大，且通常对应季节、反应速率、原水状态、加氯强度等综合变化；模型很容易发现“沿温度轴切分后，DBP 目标明显变化”。所以 Temperature 变成 SHAP rank #1。

Dataset1 里，Temperature 的总体变异小，而且 within-tsid 中位数只有 0.79，within/overall std ratio = 0.229。这意味着很多同一 tsid 内部温度变化有限。模型在 Dataset1 中沿 Temperature 切分时，能解释的目标差异有限；相比之下，沿 Bromide、TOC、Cl2 dose 切分能得到更大的目标差异。

所以 Temperature 在 Dataset1 不是“化学上不重要”，而是“在这个数据结构里，可被模型利用的独立预测信息变少了”。

## 机制 2：Dataset1 有更直接的过程变量，削弱了 Temperature 的代理角色

Tai Lake 小数据集没有 Cl2 dose 和 contact time。模型如果想捕捉消毒过程强度，只能通过间接变量来代理，其中 Temperature 是最强的代理之一。温度不仅影响反应速率，也可能和季节性运行策略、原水水质、加氯需求共同变化。

Dataset1 直接包含 Cl2 dose 和 contact time。尤其是 Cl2 dose，它对 THM4 是非常直接的运行输入。SHAP 结果显示，在 Dataset1 6-feature 和 7-feature 中，Cl2 dose 对 THM4 经常成为 rank #1。

这说明在 Tai Lake 里由 Temperature 代理的一部分运行信息，在 Dataset1 中被 Cl2 dose 直接吸收了。模型会优先使用更直接、更接近生成机制的变量，而不是再依赖 Temperature 这个间接代理。

不过这不是全部原因，因为在 Dataset1 5-common 条件 D 中，没有 Cl2 dose 时 Temperature 仍然不是主驱动。这说明 Cl2 是“放大排名差异”的因素，不是最初让 Temperature 变弱的唯一原因。

## 机制 3：TOC 和 Bromide 在 Dataset1 中有更强的可学习异质性

Dataset1 的 TOC 和 Bromide 比 Tai Lake 更异质、更长尾。

TOC：

- Dataset1 std = 3.10，Tai Lake std = 0.80。
- Dataset1 std 约为 Tai Lake 的 3.85 倍。
- Dataset1 的 TOC 均值高于中位数，说明有明显右尾。

Bromide：

- 原始数值尺度需要进一步核对单位，但结构上非常明确：Dataset1 的 Bromide 更长尾、更异质。
- 若按 mg/L-equivalent 理解，Dataset1 的 Bromide 中心值和 Tai Lake 同量级，但变异性仍显著更强。
- Dataset1 within-tsid Bromide 中位 std 远低于 overall std，说明差异主要来自 tsid 之间的水源/工艺异质性。

这会改变模型驱动模式。DBCM 和 BDCM 是含溴副产物，Bromide 是更直接的前体物信号。当 Dataset1 中 Bromide 变异足够大，模型自然会把 Bromide 放到 DBCM/BDCM 的主导位置。Tai Lake 中 Bromide 范围较窄，Temperature 反而更能解释跨样本的目标变化。

THM4 则更容易被 TOC 和 Cl2 dose 驱动，因为 TOC 表示有机前体物供给，Cl2 dose 表示氧化/消毒强度。Dataset1 中这两个变量的结构性变异都比 Temperature 更有解释力，所以 THM4 的驱动模式从 Temperature 转向 TOC/Cl2 dose。

## 机制 4：Dataset1 的主要差异是 tsid 之间的结构差异，不是 tsid 内部动态

Dataset1 的很多输入都有一个共同结构：within-tsid 波动远小于整体波动。

例如：

- pH：within/overall std ratio = 0.197。
- UV254：within/overall std ratio = 0.062。
- TOC：within/overall std ratio = 0.107。
- Bromide：within/overall std ratio = 0.045。
- Cl2 dose：within/overall std ratio = 0.093。
- Contact time：within/overall std ratio = 0.003。

这说明 Dataset1 的大部分预测信息不是“同一个水厂/同一类运行序列内部随时间连续变化”，而是“不同 tsid 之间本来就处在不同的水源、工艺、加氯、前体物水平”。

模型面对这种数据时，会学习到跨 tsid 的结构性差异。例如某些 tsid 天然 Bromide 高、TOC 高、Cl2 dose 高、contact time 长，对应的 DBP 目标也高。于是模型预测更像是在识别“这条样本属于哪类水源/工艺状态”，而不是像 Tai Lake 那样主要沿着温度季节轴做预测。

这也是为什么 Dataset1 的 driver 更像“source/process chemistry drivers”，而 Tai Lake 的 driver 更像“seasonal temperature driver”。

## 机制 5：小数据集的输入空间更窄，Temperature 更容易成为总代理

Tai Lake 的很多非温度输入范围较窄：

- pH std = 0.16，IQR = 0.20。
- TOC std = 0.80。
- Bromide std = 0.031。
- UV254 std = 0.047。

这些变量不是没有化学意义，而是在这个小数据集里能提供的可分割空间较有限。模型在寻找最能降低误差的方向时，会优先选择变化幅度大、同时又和目标强相关的变量。Temperature 正好满足这两个条件。

所以 Tai Lake 中 Temperature 的 SHAP dominance 不应被解释成“温度永远是 DBP 的第一因子”，而应解释成：在这个单数据源、小范围化学输入、强季节变化的数据结构中，Temperature 是最强的综合代理变量。

## 机制 6：特征相关结构会改变 SHAP 归因

SHAP 排名不是纯粹的因果排序。它依赖模型在给定数据分布中如何分配解释权。

如果 Temperature 和加氯强度、原水季节、TOC、UV254 等变量高度共变，而 Cl2 dose 没有被观测到，那么模型可能把这些未观测机制的一部分归因到 Temperature。

如果 Dataset1 直接观测了 Cl2 dose，并且 Bromide/TOC 有更大的独立变异，那么原来可能被 Temperature 代理的解释权会被重新分配给更直接的变量。于是 SHAP 排名发生转移。

这就是两个数据集的“驱动模式”看起来完全不同的根本统计原因：不是一个模型突然改变了化学规律，而是两个数据集提供给模型的可学习协方差结构不同。

## 对模型预测的直接影响

在 Tai Lake 模型中，预测值会对 Temperature 特别敏感。高温样本往往推动 THM4、DBCM、BDCM 预测上升；Temperature 的变化同时携带反应动力学和季节运行信息。

在 Dataset1 模型中，预测值会对目标相关的直接化学/运行变量更敏感：

- THM4：Cl2 dose、TOC、Bromide 是主要预测轴。
- DBCM：Bromide 是最稳定的主驱动，Temperature 只在部分模型中保留次级作用。
- BDCM：Bromide 和 Cl2 dose 更重要，UV254/TOC 作为有机前体物信号参与解释。

因此，如果把 Tai Lake 模型拿去解释 Dataset1，容易过度强调 Temperature；如果把 Dataset1 模型拿去解释 Tai Lake，可能会期待 Cl2/Bromide/TOC 结构信号，但 Tai Lake 数据里这些变量的变异空间不足或变量缺失，导致解释不匹配。

## 可用于论文/汇报的表述

最稳妥的表述是：

> The difference in driver patterns is primarily a data-distribution effect. In the Tai Lake single-plant dataset, Temperature spans a much broader seasonal range and acts as a strong proxy for reaction kinetics and unobserved operational variation. In Dataset1, Temperature variation is more compressed, especially within tsid series, while precursor and process variables such as Bromide, TOC, and Cl2 dose vary more strongly across source/process contexts. As a result, the model reallocates predictive attribution from Temperature to more direct chemical and operational drivers.

中文可以写成：

> 两个数据集的 SHAP 驱动模式差异主要来自数据分布结构差异。Tai Lake 小数据集中的温度跨度大、季节性强，并且可能代理了未观测的加氯和原水状态变化，因此模型把 Temperature 学成主导特征。Dataset1 中温度变异被压缩，尤其在同一 tsid 内部波动较弱，而 Bromide、TOC、Cl2 dose 等前体物和运行变量在不同 tsid/水源/工艺之间差异更大、更直接对应 DBP 形成，因此模型把解释权转移到这些变量上。

## 需要明确的边界

这些结论是模型解释和数据结构层面的，不应直接写成单一因果结论。我们可以说“data-source / data-distribution effect 是最强支持的解释”，但不能说“只有水化学差异造成了全部变化”。

特别是 Bromide，需要进一步核对 Dataset1 的存储单位。如果 Dataset1 `br_in_avg` 是 ug/L，而 Tai Lake `Br_mg_L` 是 mg/L，则绝对比值必须先换算；但无论是否换算，Dataset1 Bromide 的长尾和异质性更强这一结构判断仍然成立。

