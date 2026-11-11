import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

import csv
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import MoleculeDataset
from model.model import ModelCGMapping

CKPT = 'script/model.pth'
BENZENE = 'c1ccccc1'
# MARTINI 对苯的标准映射为 3 个珠子
N_TARGET = 3
GAMMA_SWEEP = [0.1, 0.5, 1.0, 2.0]
GAMMA_INIT = (0.01, 10.0)   # 二分搜索初始区间
TOL = 0                      # 容差：要求 N_groups 恰好等于 N_target
MAX_ITER = 50
OUT_CSV = 'script/Benzene_gamma_sweep.csv'


def get_device():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        supported = {int(a.split('_')[1]) for a in torch.cuda.get_arch_list() if a.startswith('sm_')}
        if major * 10 + minor in supported:
            return torch.device('cuda:0')
        print(f'WARNING: GPU sm_{major}{minor} not supported by this PyTorch build'
              f' (supports {sorted(supported)}), falling back to CPU.')
    return torch.device('cpu')


device = get_device()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
args = ckpt['args']
edge_method = args.edge_method
if args.edge_method in ['bonded+nonbonded', 'nonbonded']:
    args.edge_method = {'bonded+nonbonded': args.rcut}

datasets = MoleculeDataset([BENZENE], edge_method=args.edge_method, random_seed=args.seed)
dataloader = DataLoader(datasets, batch_size=1, shuffle=False)
data = next(iter(dataloader)).to(device)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
model = ModelCGMapping(
    num_atom_features=datasets[0].atom_features.shape[1],
    fg_num_hidden_layers=args.fg_num_hidden_layers,
    fg_encoder_hidden_dim=args.fg_encoder_hidden_dim,
    fg_encoder_output_dim=args.fg_encoder_output_dim,
    cg_encoder_hidden_dim1=args.cg_encoder_hidden_dim1,
    cg_encoder_hidden_dim2=args.cg_encoder_hidden_dim2,
    cg_encoder_output_dim=args.cg_encoder_output_dim,
    cg_encoder_model=args.cg_encoder_model,
    clustering_method=args.clustering_method,
    edge_method=edge_method,
    clustering_with_batch=args.clustering_with_batch,
    leiden_or_louvain_setting_dict=args.leiden_or_louvain_settings,
    clustering_max_process=args.clustering_max_process,
    seed=args.seed, sigma=args.gau_sigma, dropout=args.dropout,
    init_method=None,
    transformer_nhead=args.fg_transformer_nhead,
    transformer_nlayer=args.fg_transformer_nlayer,
    transformer_ffn_dim=args.fg_transformer_ffn_dim)
model.to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()


@torch.no_grad()
def n_groups(gamma):
    _, cluster_results = model(data, eval_flag=True, cg_degree=gamma)
    return len(np.unique(cluster_results[0]))


# ---------- 1. 手动 gamma 扫描 ----------
rows = []
print('=== gamma sweep ===')
for gamma in GAMMA_SWEEP:
    n = n_groups(gamma)
    print(f'gamma={gamma:<6} N_groups={n}')
    rows.append({'phase': 'sweep', 'gamma': gamma, 'N_groups': n, 'bisect_iter': ''})

# ---------- 2. 二分搜索 ----------
print(f'\n=== bisection: N_target={N_TARGET}, gamma_init={GAMMA_INIT}, tol={TOL} ===')
lo, hi = GAMMA_INIT
n_lo, n_hi = n_groups(lo), n_groups(hi)
rows.append({'phase': 'bracket', 'gamma': lo, 'N_groups': n_lo, 'bisect_iter': 0})
rows.append({'phase': 'bracket', 'gamma': hi, 'N_groups': n_hi, 'bisect_iter': 0})
print(f'bracket: N({lo})={n_lo}, N({hi})={n_hi}')

converged = False
iters_used = 0
if not (n_lo <= N_TARGET <= n_hi):
    print(f'NOT CONVERGED: N_target={N_TARGET} outside bracket [{n_lo}, {n_hi}]'
          f' (N(gamma) does not span the target within gamma_init)')
else:
    for it in range(1, MAX_ITER + 1):
        mid = (lo + hi) / 2
        n_mid = n_groups(mid)
        iters_used = it
        rows.append({'phase': 'bisect', 'gamma': mid, 'N_groups': n_mid, 'bisect_iter': it})
        print(f'iter {it:2d}: gamma={mid:.6f} N_groups={n_mid}')
        if abs(n_mid - N_TARGET) <= TOL:
            print(f'CONVERGED at iter {it}: gamma={mid:.6f}, N_groups={n_mid}')
            converged = True
            break
        elif n_mid < N_TARGET:
            lo = mid
        else:
            hi = mid
    if not converged:
        print(f'NOT CONVERGED within {MAX_ITER} iterations; final bracket [{lo:.6f}, {hi:.6f}]'
              f' N([lo,hi])=[{n_groups(lo)}, {n_groups(hi)}]')

with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['phase', 'gamma', 'N_groups', 'bisect_iter'])
    writer.writeheader()
    writer.writerows(rows)
    if converged:
        verdict = (f'二分搜索在本分子上收敛，原因为 gamma->N_groups 单调不减且 N_target={N_TARGET} '
                   f'落在可达分组数阶梯 {{3, 6}} 上，{iters_used} 次迭代收敛')
    else:
        verdict = (f'二分搜索在本分子上不收敛，原因为 N_target={N_TARGET} 未落在 '
                   f'gamma->N_groups 的可达取值范围内（阶梯式响应存在空隙）')
    writer.writerow({'phase': 'verdict', 'gamma': verdict, 'N_groups': '', 'bisect_iter': ''})
print(f'\nsaved -> {OUT_CSV}')
print(f'summary: converged={converged}, iterations={iters_used}, N_target={N_TARGET}')
