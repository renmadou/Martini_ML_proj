# AGENTS.md — Martini_ML_proj 工作指南

> 本文档汇总原 `/share/home/johndou/MolCluster`、`/share/home/johndou/Martini_Mapper` 及 home 目录下历次会话积累的全部操作上下文。**请统一在 `/share/home/johndou/Martini_ML_proj` 目录下启动 kimi**，不要再从旧项目目录启动。

## 1. 项目是什么

MARTINI 粗粒化映射的统一工作区，整合两条技术路线：

- `Martini_Mapper/` — 规则映射工具（原 `renmadou/Martini_Mapper`，fork 自 `eliobaby`，分支 `main`）
- `MolCluster/` — GNN + Leiden 聚类映射（原 `renmadou/MolCluster`，分支 `master`）
- 两者以 read-tree 合并方式并入本仓库，完整 git 历史保留；本仓库 remote：`git@github.com:renmadou/Martini_ML_proj.git`（分支 `main`）
- 原两个独立仓库保留在 `/share/home/johndou/` 下，仅作 subtree 同步源（remote 名 `molcluster_src`、`mapper_src`）

## 2. 目录结构

```
Martini_ML_proj/
├── Martini_Mapper/       # 规则映射
├── MolCluster/           # ML 映射（代码在 MolCluster/MolCluster/ 子目录）
├── pipeline/             # 跨工具整合层（见 pipeline/README.md）
├── benchmarks/           # 对比实验产物（mapper_outputs/ 被 gitignore）
├── docs/                 # Week1_baseline_memo.md、Benzene_comparison.md
└── envs/                 # 两个 conda 环境的 conda list --export 快照
```

## 3. 双 conda 环境（不可合并，硬约束）

| 环境 | Python | 关键栈 | 用途 |
|---|---|---|---|
| `molcluster` | 3.9.0 | torch 2.7.1+cu128、torchvision 0.22.1+cu128、torch-geometric **2.6.1** | MolCluster 训练/推理 |
| `martini_mapper` | 3.11.18 | xtb 6.7.1、xtb-python 22.1、rdkit | Martini Mapper |

- 跨环境调用一律用 `conda run -n <env> python ...` 子进程（见 `pipeline/run_comparison.py`）
- molcluster 安装 torch 必须带 `--extra-index-url https://download.pytorch.org/whl/cu128`
- 声明文件：`MolCluster/enviroment.yaml`、`MolCluster/requirements_py39.txt`；冻结快照：`envs/*.txt`（环境变动后须重新导出 `conda list --export` 并提交）

## 4. 主机环境约束（踩坑记录，勿再踩）

- OS：Rocky Linux 8.10，**glibc 2.28**；GPU：RTX 5090（sm_120），驱动 580.65.06，CUDA 13.0
- **PyG ≥ 2.7 要求 Python ≥ 3.10**，py39 环境最高只能装 2.6.1
- **PyG 扩展轮子（torch-scatter/sparse/cluster/spline-conv 的 `+pt27cu128`）要求 GLIBC_2.32，本机无法加载**（`OSError: GLIBC_2.32 not found`）。不要安装；项目只用 GCNConv/GATConv/GINConv/SAGEConv/NNConv 和 global pooling，PyG 原生回退足够。若必须，需 CUDA 12.8 工具链源码编译
- `git subtree` 命令不可用（git 2.43.0 无 subtree 插件）；子树操作用 `git merge -s ours --no-commit` + `git read-tree --prefix=... -u` 等效实现
- 无 `gh` CLI、无 GitHub PAT；push 用 SSH（`git@github.com`，已认证为 `renmadou`）

## 5. 常用操作

```bash
# 交叉对比管线（任意环境，base 即可）
python pipeline/run_comparison.py molecules.txt --tag <tag> [--tol 1] [--no-xtb]
# molecules.txt 每行 "<Name> <SMILES>"；产物在 benchmarks/

# MolCluster 单分子二分搜索（molcluster 环境内）
conda run -n molcluster python pipeline/molcluster_bisect.py \
    --smiles c1ccccc1 --n-target 3 --tol 1
# stdout 最后一行 JSON：converged/gamma/n_groups/iterations/groups

# Martini Mapper 单分子（martini_mapper 环境内）
conda run -n martini_mapper python -m martini_mapper Benzene c1ccccc1 --out-dir <dir> [--no-xtb]
# 输出直接落在 <dir>/<Name>.{txt,gro,itp}（不会自动建子目录）

# MolCluster 冒烟测试
cd MolCluster/MolCluster && conda run -n molcluster python minimal_test.py
```

MolCluster 推理依赖 checkpoint：`MolCluster/MolCluster/script/model.pth`（checkpoint 内含 `args` 命名空间，模型超参从其中读取；`torch.load` 需 `weights_only=False`）。

## 6. 已有结论（苯基线，2026-11-11）

- 苯：Martini Mapper = 3 beads（3×TC5，归属 {0,1}{2,3}{4,5}）；MolCluster 二分搜索 N_target=3，γ_init=[0.01,10]、ε=0，**3 次迭代收敛于 γ=1.258750**，原子归属与 Mapper 完全一致
- γ→N_groups 单调不减但呈**阶梯平台**（苯只有 {3,6} 两档）；N_target 落阶梯空隙时二分搜索永不收敛 → 复杂分子建议 ε=1 或比例型目标（第 4 周风险预案）
- `--no-xtb`：.gro 坐标可用且与 default 一致；.itp 完整但键长退化为统一默认 0.29103 nm、无 angles/dihedrals——仅作拓扑预检；xtb 已装好，默认走 xtb 流程
- 详细记录：`docs/Week1_baseline_memo.md`、`docs/Benzene_comparison.md`、`benchmarks/Benzene_gamma_sweep.csv`

## 7. 第 2 周任务前置（来自 Week1 备忘录）

1. 周一前确认叶酸 SMILES；叶酸 C₁₉H₁₉N₇O₆ 只含 C/H/N/O，不触及字典对 S/P/卤素的实验性支持；其余候选分子逐一做元素筛查
2. 两工具统一用 RDKit 原子编号做归属对比
3. MolCluster N_target 取 Mapper 实际珠子数，二分初值沿用 γ_init=[0.01,10]、ε=1

## 8. 协作约定

- 提交信息用英文；`git commit` 前先把拟用信息给用户确认
- 交付物优先落盘到 `docs/`、`benchmarks/` 并提交推送
- 环境有任何变动：更新 `envs/` 快照 + 备忘录中记录原因
