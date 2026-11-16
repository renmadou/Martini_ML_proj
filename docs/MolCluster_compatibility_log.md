# MolCluster 输入兼容性日志（第 3 周·周四，2026-11-16）

> 目的：确认 MolCluster 推理入口能接受与 Mapper 基线同一批 SMILES（`benchmarks/molecule_set_SMILES.csv` 冻结版），不调 γ，只做：①分子图构建 ②前向传播 ③预定义基团对保护覆盖 ④输出可读性。
> 运行方式：`conda run -n molcluster python pipeline/molcluster_compat.py /tmp/week3_mols.txt`（新脚本，固定 γ=1.0 单次前向，逐分子输出 JSON）。
> 模型：`MolCluster/MolCluster/script/model.pth`（`weights_only=False`，超参从 checkpoint 内 `args` 读取）。SMARTS 词表：`data/martini2_vocab-2.txt`（16 条）。

## 兼容性矩阵

| 分子 | 重原子/边数 | 图构建 | 前向传播 | 分组输出可读性 | SMARTS 保护覆盖 | 状态 |
|---|---|---|---|---|---|---|
| Folate | 32 / 68 | ✅ | ✅（cuda:0，无 OOM） | ✅ 32 原子逐一有组标签，γ=1.0 得 8 组 | benzene_ring×1、carboxylic_acid×2、amide×1、ketone×2、amine×2；**蝶呤稠环、芳香 NH₂ 无内置条目** | 可进入第 4 周 γ 校准 |
| BPDC | 18 / 38 | ✅ | ✅ | ✅ 18 原子有标签，γ=1.0 得 4 组 | benzene_ring×2、carboxylic_acid×2 — **全覆盖** | 可进入第 4 周 γ 校准 |
| alpha-CD | 66 / 144 | ✅ 30 元大环正常解析 | ✅ 无显存不足/截断（显存占用可忽略，前向 <1 s） | ✅ 66 原子有标签，γ=1.0 得 10 组 | alcohol×18、ether×12、diol×6、alkyl_4carbon×18；**无糖苷键专用条目（由通用 ether 覆盖）** | 可进入第 4 周 γ 校准（硬性边界预案未触发） |

无崩溃点，无截断，无算子不支持。第 3 周计划的"α-CD 大环完全无法建图则降级"硬性边界**未触发**。

## 预定义基团保护偏离度（γ=1.0 参考值，非校准结果）

用 `utils.count_predefined_group_departure` 统计"同一 SMARTS 匹配的原子被拆到不同组"的次数（偏离/匹配总数）：

| 分子 | 偏离/匹配 | 解读 |
|---|---|---|
| BPDC | 0 / 4 | 2 个苯环 + 2 个羧基全部完整成组，保护机制有效 |
| Folate | 3 / 8 | 8 处匹配中 3 处被拆分；γ=1.0 非校准值，第 4 周校准后需复测 |
| alpha-CD | 20 / 54 | 大环多羟基体系在 γ=1.0 下分组细碎，属预期；需第 4 周校准后复测 |

## 保护项缺口（记录为"需要手动扩展的缺失"，本周不修改损失函数）

1. **蝶呤环/嘌呤样稠合杂环**：词表只有单环芳杂环（pyrrole、pyridine），无蝶啶/蝶呤/嘌呤条目——叶酸蝶呤环上的内酰胺 C=O 被 ketone 通用匹配，芳香 NH₂（连在芳碳上）不匹配 `amine`（其 SMARTS 要求脂肪碳 `[NX3;H1,H2][CX4]`）。
2. **糖苷键**：无专用条目，仅由通用 `ether` 覆盖；无法区分"糖苷氧应归属哪一侧"。
3. α-CD 的 `alkyl_4carbon` 匹配（18 处）为词表对长脂肪链的通用条目，在糖骨架上属误配——第 5 周分歧表候选条目。

## 输出可读性

分组结果为逐原子整数标签（RDKit 原子编号序），可直接与 Mapper `.txt` 的 `atoms_idx` 做第 5 周逐原子归属对比——第 2 周前置任务 2（统一 RDKit 原子编号）在工具层面已可行。

## 结论

三个分子在 MolCluster 侧**全部可进入第 4 周 γ 二分搜索**，无降级条目；唯一约束是保护词表对蝶呤环/糖苷键的缺口，列入第 5 周分歧表候选。
