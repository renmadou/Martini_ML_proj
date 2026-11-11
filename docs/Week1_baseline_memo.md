# 第 1 周基线备忘录（Martini Mapper × MolCluster,苯）

日期：2026-11-11
状态：第 1 周出口检查。本文档 + 两个环境快照 + `Martini_Mapper/Benzene_comparison.md` + `MolCluster/MolCluster/script/Benzene_gamma_sweep.csv` 共 5 个文件，足以在不询问作者的情况下复现第 1 周全部步骤。

---

## 1. 环境配置路径与失败分支

### 1.1 主机与系统

- OS：Rocky Linux 8.10 (Green Obsidian)，内核 4.18.0-553.el8_10
- **glibc 2.28**（关键约束，见 1.4）
- GPU：NVIDIA RTX 5090（sm_120），驱动 580.65.06，CUDA 13.0

### 1.2 molcluster 环境（MolCluster 用）

- 路径：`/share/home/johndou/.conda/envs/molcluster`，Python 3.9.0
- 快照：`MolCluster/env_molcluster.txt`（`conda list --export`，2026-11-11 更新）
- 声明文件：`MolCluster/enviroment.yaml`、`MolCluster/requirements_py39.txt`（git commit `bac0fbc`）
- 关键栈：torch 2.7.1+cu128 / torchvision 0.22.1+cu128（`--extra-index-url https://download.pytorch.org/whl/cu128`），torch-geometric **2.6.1**
- 复现：`conda create -n molcluster python=3.9` 后 `pip install -r MolCluster/requirements_py39.txt`

### 1.3 martini_mapper 环境（Martini Mapper 用）

- 路径：`/share/home/johndou/.conda/envs/martini_mapper`，Python 3.11.18
- 快照：`Martini_Mapper/env_martini_mapper.txt`（`conda list --export`，2026-11-11 更新）
- **xtb 6.7.1 + xtb-python 22.1 已通过 conda-forge 安装成功**——周一预案中的"conda 装 xtb 失败则降级 pip / --no-xtb"两条失败分支均未触发，default（xtb）流程可用
- 复现：`conda create -n martini_mapper --file Martini_Mapper/env_martini_mapper.txt`，再 `pip install -e Martini_Mapper/`

### 1.4 已记录的失败/降级分支（molcluster 环境升级时实际踩到）

1. **PyG 2.7.0 在 py39 装不上**：`torch-geometric>=2.7` 要求 Python ≥3.10，py39 上限为 2.6.1。已固定 `torch-geometric==2.6.1` 并验证与 torch 2.7.1 兼容。
2. **PyG 扩展轮子与 glibc 冲突**：torch-scatter/sparse/cluster/spline-conv 的 `+pt27cu128` 预编译轮子要求 GLIBC_2.32，本机 2.28，import 直接 `OSError`。已卸载；项目只用 GCNConv/GATConv/GINConv/SAGEConv/NNConv 和 global pooling，PyG 有原生回退，功能不受影响。若日后需要这些扩展，须装 CUDA 12.8 工具链源码编译。
3. 验证：`MolCluster/MolCluster/minimal_test.py` 端到端通过（5 个分子 + checkpoint 推理，modularity 0.6381）。

## 2. --no-xtb 下 Martini Mapper 输出状态（苯）

对比目录：`Martini_Mapper/outputs/Benzene_default/` vs `Martini_Mapper/outputs/Benzene_noxtb/`，详见 `Martini_Mapper/Benzene_comparison.md` 第 2 节。

- **.gro：有可用坐标**。两个模式的 `Benzene.gro` 完全一致，3 个 bead 坐标为原子质心位置（来自 default 流程保存的几何）；分子映射拓扑（3 × TC5，归属 {0,1}{2,3}{4,5}）在两种模式下相同。
- **.itp：完整但精度降级**。分子类型、原子表、电荷/质量完整；差异在成键参数：
  - default：`[constraints]` 按 xtb 实际几何给 b0（≈0.288 nm），含 `[angles]`（≈60°，力常数由 Hessian 导出）
  - noxtb：退回 `[bonds]` 统一默认 b0=0.29103 nm、k=20000，**无 angles/dihedrals 参数**
- 结论：noxtb 可做映射拓扑预检，不宜直接用于生产 MD。xtb 已装好，第 2 周默认走 default 流程。

## 3. MolCluster 在苯上的 γ 响应与二分搜索收敛性

数据：`MolCluster/MolCluster/script/Benzene_gamma_sweep.csv`；脚本：`MolCluster/MolCluster/script/benzene_gamma_sweep.py`（git commit `8b0a331`）。固定随机种子下两次运行结果完全一致。

**γ → N_groups 响应（苯，6 原子）**：γ = 0.1 / 0.5 / 1.0 → N=3；γ = 2.0 → N=6。单调不减，但呈**宽平台阶梯**，N_groups 只能取 {3, 6}。

**二分搜索**：N_target=3（对齐 MARTINI 苯的 3 珠子），γ_init=[0.01, 10]，ε=0 → **3 次迭代收敛于 γ=1.258750**，N_groups=3，且原子归属与 Martini Mapper 完全一致（均为相邻碳两两成对）。

**明确判断**：二分搜索在苯上**收敛**，原因为 γ→N_groups 单调不减且 N_target=3 落在可达阶梯 {3, 6} 上。**风险预案（带入第 4 周）**：若复杂分子的 N_target 落在阶梯空隙（如苯的 N_target=4），|N−N_target|≤ε 永不满足，搜索会耗尽迭代——建议 ε 取 1，或改用 N_atom 比例型目标。

## 4. 第 2 周分子集准备的前置条件

1. **叶酸 SMILES 必须在周一前确认**。叶酸（C₁₉H₁₉N₇O₆）只含 C/H/N/O，**不含 S/P/卤素**，规避了 Martini Mapper 字典对这些元素的实验性支持风险；其余候选分子须逐一做同样的元素筛查，含 S/P/卤素的分子要么替换、要么标记为"字典支持待验证"。
2. 每个分子同时准备：Martini Mapper 输入（SMILES + 名称）与 MolCluster 输入（.smi 行），原子编号统一用 RDKit 顺序，便于归属对比。
3. MolCluster 侧 N_target 取 Martini Mapper 实际输出的珠子数；二分搜索初值沿用苯的参照（γ_init=[0.01, 10]，苯收敛于 γ≈1.26 / 3 次迭代），ε=1。
4. 环境冻结已完成，第 2 周任何人先核对 `env_molcluster.txt` / `env_martini_mapper.txt` 再开工；环境有变动必须重新导出快照并记录原因。

## 5. 复现索引（交付文件）

> 2026-11-11 起，两个项目已整合进统一工作区仓库 `Martini_ML_proj`（`/share/home/johndou/Martini_ML_proj`），以下路径均为该仓库内路径；两侧原始仓库保留不动。

| 文件 | 内容 |
|---|---|
| `envs/env_molcluster.txt` | molcluster 环境快照（conda list --export） |
| `envs/env_martini_mapper.txt` | martini_mapper 环境快照（conda list --export） |
| `docs/Benzene_comparison.md` | 苯：Martini Mapper vs MolCluster 并排表 + default/noxtb 差异 |
| `benchmarks/Benzene_gamma_sweep.csv` | γ 扫描、二分搜索迭代记录与收敛判断 |
| `pipeline/run_comparison.py` | 跨工具对比驱动（苯一行命令复跑：Mapper → MolCluster 二分 → 对比表） |

正文其他位置提到的 `MolCluster/...`、`Martini_Mapper/...` 相对路径在 `Martini_ML_proj` 仓库根目录下同样成立。
