# 第 3 周分子集冻结备忘录

日期：2026-11-16（实际执行于第 2 周周一，进度超前约 4 个工作日）
状态：第 3 周出口检查。分子集自此冻结，第 4 周 γ 校准不再回头修改分子集；任何新增分子须单独记为"第 4 周后增补"。

> 路径说明：本备忘录全部路径相对 `/share/home/johndou/Martini_ML_proj`。原计划中 `outputs/week3_mapper/` 按工作区约定改归档至 `benchmarks/week3_mapper/`（`benchmarks/mapper_outputs/` 为 gitignore 的临时区，正式基线需要可追溯，故单独建目录并入库）。

## 1. 最终 SMILES 与质子化态/互变异构体声明

定稿文件：`benchmarks/molecule_set_SMILES.csv`（2026-11-11 冻结，含来源 CID/URL、声明、原子数）。要点：

| 分子 | 来源 | 声明要点 |
|---|---|---|
| Folate | PubChem CID 6037（11-11 现场确认 CID 未变） | 中性；两个羧基质子化；蝶呤环固定为 canonical 酮式，不探索互变异构体空间；isomeric SMILES 保留 L-谷氨酸 α-碳手性 |
| BPDC | PubChem CID 13084 + 文献（Gu et al. AFM 2023, DOI: 10.1002/adfm.202311864 Fig.1a） | 中性二羧酸为 Mapper 输入惯例；阎云老师 11-12 确认实验投料为**二钠盐**——第 4 周 γ 校准/第 6 周配位测试须补 BPDC²⁻ 对比行 |
| alpha-CD | PubChem CID 444913 | 中性全羟基；手性标记已核对（30 个立体中心 = 6 葡萄糖 × 5）；C36H60O30，126 原子 |
| Folate–M²⁺ 片段 | 由 Folate 派生 | 不生成独立 SMILES（Martini 3 无标准金属珠子）；配位子结构定义（谷氨酸尾两羧基 C/O + 相邻酰胺 N）留第 6 周 |

## 2. Martini Mapper 支持状态矩阵

详表：`benchmarks/Mapper_baseline_summary.csv`；失败定位过程：`docs/Mapper_precheck_log.md`；正式基线输出：`benchmarks/week3_mapper/`（`--no-xtb`，与第 1 周苯基线格式一致，仅 `[bonds]` 默认参数，仅作拓扑基线）。

| 分子 | 状态 | 失败/存疑定位 | 判定 |
|---|---|---|---|
| BPDC | ✅ 支持 | 8 beads：2 苯环 ×3 TC5 + 2 羧基各 1 SP2 | 覆盖范围内；羧基各自独立成珠（双羧基合并假说的负面先验） |
| alpha-CD | ⚠️ 支持但存疑 | 24 beads（每葡萄糖 4，Martini 3 原生为 3）；糖苷氧与碳配对 TN4a；TC3(C,C) = 某葡萄糖 C4/C5 环碳对，化学上可疑 | 覆盖范围内，与 Martini 3 碳水化合物构建块不一致——第 5 周分歧表第一批条目 |
| Folate | ❌ 完全失败 | ①蝶呤嘧啶酮环（鸟嘌呤样内酰胺 C=O/NH₂ 在芳香稠环无匹配珠子）；②谷氨酸尾（α-氨基酸+远端第二羧基支化无法分配） | 均为**字典/算法未覆盖**，非珠子层面不可表达 |
| Folate–M²⁺ 片段 | 本周不运行 | — | 第 6 周处理 |

叶酸后续路径按预检日志预案：周三正式运行确认整体失败后，采用 (a) 碎片对比（PABA ✅ / 蝶呤 ❌ / 谷氨酸 ❌ 已于预检完成碎片级测试）并转入 MolCluster-only 记录；不手工拼接碎片强行生成拓扑（第 3 周硬性边界）。

## 3. MolCluster 输入兼容性矩阵

详表：`docs/MolCluster_compatibility_log.md`（新脚本 `pipeline/molcluster_compat.py`，固定 γ=1.0 单次前向）。

| 分子 | 图构建 | 前向传播 | 输出可读性 | 保护项覆盖 |
|---|---|---|---|---|
| Folate（32 原子/68 边） | ✅ | ✅ cuda:0 | ✅ 逐原子标签 | 羧基/苯环/酰胺覆盖；**蝶呤稠环、芳香 NH₂ 缺条目** |
| BPDC（18/38） | ✅ | ✅ | ✅ | 全覆盖（偏离 0/4） |
| alpha-CD（66/144） | ✅ 大环正常 | ✅ 无 OOM/截断 | ✅ | 羟基/醚/二醇覆盖；**无糖苷键专用条目** |

无崩溃、无截断；"α-CD 无法建图则降级"硬性边界**未触发**。

## 4. 第 4 周 γ 校准的输入约束

- **可正常进入 γ 二分搜索**：BPDC（N_target=8）、alpha-CD（N_target=24）。沿用二分初值 γ_init=[0.01,10]、ε=1（第 1 周备忘录口径）。
- **降级条目**：Folate —— Mapper 侧无基线珠子数，**不进入可比分辨率对比**；改为 MolCluster-only 记录（自由 γ 扫描 + 保护项偏离度），降级理由：Mapper 字典未覆盖蝶呤环与谷氨酸尾，无 N_target 可取。
- **α-CD 在 MolCluster 侧无降级**（与硬性边界预案相反，实测图构建/前向均正常）；但其 Mapper 珠子方案与 Martini 3 原生碳水构建块不一致，γ 校准时 N_target=24 仅对齐 Mapper 输出，不代表化学正确性。
- 校准后须对所有分子复测预定义基团偏离度（γ=1.0 参考值：BPDC 0/4、Folate 3/8、alpha-CD 20/54）。
- 第 5 周分歧表第一批候选：①α-CD 每葡萄糖 4 vs 3 beads；②糖苷氧归属；③TC3 糖骨架珠；④蝶呤环字典缺口（Mapper 失败 + MolCluster 保护缺口双侧条目）；⑤谷氨酸尾非环细分算法缺口。

## 本周产出清单核对

- [x] `benchmarks/molecule_set_SMILES.csv`（冻结版，11-11）
- [x] `docs/Mapper_precheck_log.md`（11-12）
- [x] `benchmarks/week3_mapper/`（BPDC、alpha-CD 的 .itp/.gro/.txt；Folate 无输出，失败记录在案）
- [x] `benchmarks/Mapper_baseline_summary.csv`
- [x] `docs/MolCluster_compatibility_log.md`（含新脚本 `pipeline/molcluster_compat.py`）
- [x] 本备忘录
