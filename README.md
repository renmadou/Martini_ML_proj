# Martini_ML_proj

MARTINI 粗粒化映射的两条技术路线——规则映射（Martini Mapper）与机器学习（MolCluster）——的统一工作区。两个子项目以 git 子树形式并入，完整提交历史保留。

## 结构

```
Martini_ML_proj/
├── Martini_Mapper/   # 规则映射工具（原 renmadou/Martini_Mapper, main 分支）
├── MolCluster/       # GNN + Leiden 聚类映射（原 renmadou/MolCluster, master 分支）
├── pipeline/         # 跨工具整合层：Mapper 珠子数 → MolCluster 二分搜索 → 对比表
├── benchmarks/       # 跨工具对比实验产物（γ 扫描、对比表、Mapper 输出）
├── docs/             # 周基线备忘录与交叉对比文档
└── envs/             # 两个 conda 环境的冻结快照（conda list --export）
```

## 双环境说明

两个 conda 环境不可合并（依赖栈冲突），整合层通过 `conda run` 子进程桥接：

| 环境 | Python | 关键栈 | 快照 |
|---|---|---|---|
| molcluster | 3.9.0 | torch 2.7.1+cu128, torch-geometric 2.6.1 | `envs/env_molcluster.txt` |
| martini_mapper | 3.11.18 | xtb 6.7.1, rdkit | `envs/env_martini_mapper.txt` |

环境复现见 `docs/Week1_baseline_memo.md` 第 1 节。

## 快速开始

```bash
printf 'Benzene c1ccccc1\n' > molecules.txt
python pipeline/run_comparison.py molecules.txt --tag demo
cat benchmarks/demo_comparison.md
```

详见 `pipeline/README.md`。

## 文档索引

- `docs/Week1_baseline_memo.md` — 第 1 周基线备忘录（环境、失败分支、γ 摸底结论、第 2 周前置条件）
- `docs/Benzene_comparison.md` — 苯：两工具并排对比 + xtb/--no-xtb 差异
- `benchmarks/Benzene_gamma_sweep.csv` — MolCluster 苯上 γ→N_groups 响应与二分搜索记录
