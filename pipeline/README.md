# pipeline — 跨工具整合层

桥接两个 conda 环境（不合并环境本身）：

- `martini_mapper`（py3.11 + xtb）：SMILES → MARTINI 珠子数与原子归属
- `molcluster`（py3.9 + torch/cu128 + PyG 2.6.1）：Leiden γ 二分搜索，N_target 自动取 Mapper 珠子数

## 用法

```bash
# 分子清单：每行 "<Name> <SMILES>"
printf 'Benzene c1ccccc1\n' > molecules.txt

# 任意环境（base 即可）运行，--no-xtb 可跳过 xtb 流程
python pipeline/run_comparison.py molecules.txt --tag week2 [--tol 1] [--no-xtb]
```

产物：

- `benchmarks/<tag>_comparison.md` — 并排对比表（珠子数、N_groups、迭代数、收敛 γ、双方原子归属、结论）
- `benchmarks/mapper_outputs/` — Martini Mapper 的 .txt/.gro/.itp 原始输出

## 组件

- `run_comparison.py`：驱动脚本，在两个环境间调度（`conda run` 子进程），汇总报告
- `molcluster_bisect.py`：在 molcluster 环境内执行的单分子二分搜索，stdout 输出一行 JSON（收敛与否、γ、迭代数、分组）

依赖 MolCluster 的 checkpoint：`MolCluster/MolCluster/script/model.pth`。
