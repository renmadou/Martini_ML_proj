"""Run inside the molcluster conda env: Leiden bisection search for one molecule.

Usage:
    python pipeline/molcluster_bisect.py --smiles c1ccccc1 --n-target 3 \
        [--gamma-lo 0.01] [--gamma-hi 10.0] [--tol 1] [--max-iter 50]

Prints one JSON line to stdout:
    {"converged": true, "gamma": ..., "n_groups": ..., "iterations": ..., "groups": [...]}
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MolCluster', 'MolCluster'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MolCluster', 'MolCluster'))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import MoleculeDataset
from model.model import ModelCGMapping

CKPT = 'script/model.pth'


def get_device():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        supported = {int(a.split('_')[1]) for a in torch.cuda.get_arch_list() if a.startswith('sm_')}
        if major * 10 + minor in supported:
            return torch.device('cuda:0')
    return torch.device('cpu')


def build_model(args_ns, device):
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    margs = ckpt['args']
    edge_method = margs.edge_method
    if margs.edge_method in ['bonded+nonbonded', 'nonbonded']:
        edge_method = {'bonded+nonbonded': margs.rcut}
    ds = MoleculeDataset([args_ns.smiles], edge_method=edge_method, random_seed=margs.seed)
    data = next(iter(DataLoader(ds, batch_size=1))).to(device)
    torch.manual_seed(margs.seed)
    np.random.seed(margs.seed)
    model = ModelCGMapping(
        num_atom_features=ds[0].atom_features.shape[1],
        fg_num_hidden_layers=margs.fg_num_hidden_layers,
        fg_encoder_hidden_dim=margs.fg_encoder_hidden_dim,
        fg_encoder_output_dim=margs.fg_encoder_output_dim,
        cg_encoder_hidden_dim1=margs.cg_encoder_hidden_dim1,
        cg_encoder_hidden_dim2=margs.cg_encoder_hidden_dim2,
        cg_encoder_output_dim=margs.cg_encoder_output_dim,
        cg_encoder_model=margs.cg_encoder_model,
        clustering_method=margs.clustering_method,
        edge_method=edge_method,
        clustering_with_batch=margs.clustering_with_batch,
        leiden_or_louvain_setting_dict=margs.leiden_or_louvain_settings,
        clustering_max_process=margs.clustering_max_process,
        seed=margs.seed, sigma=margs.gau_sigma, dropout=margs.dropout,
        init_method=None,
        transformer_nhead=margs.fg_transformer_nhead,
        transformer_nlayer=margs.fg_transformer_nlayer,
        transformer_ffn_dim=margs.fg_transformer_ffn_dim)
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smiles', required=True)
    p.add_argument('--n-target', type=int, required=True)
    p.add_argument('--gamma-lo', type=float, default=0.01)
    p.add_argument('--gamma-hi', type=float, default=10.0)
    p.add_argument('--tol', type=int, default=1)
    p.add_argument('--max-iter', type=int, default=50)
    args = p.parse_args()

    device = get_device()
    model, data = build_model(args, device)

    @torch.no_grad()
    def n_groups(gamma):
        _, cluster_results = model(data, eval_flag=True, cg_degree=gamma)
        return len(np.unique(cluster_results[0])), cluster_results[0]

    lo, hi = args.gamma_lo, args.gamma_hi
    n_lo, _ = n_groups(lo)
    n_hi, _ = n_groups(hi)
    result = {'converged': False, 'gamma': None, 'n_groups': None,
              'iterations': 0, 'groups': None,
              'bracket': [n_lo, n_hi], 'n_target': args.n_target}

    if not (n_lo <= args.n_target <= n_hi):
        result['error'] = (f'N_target={args.n_target} outside bracket '
                           f'[N({lo})={n_lo}, N({hi})={n_hi}]')
    else:
        for it in range(1, args.max_iter + 1):
            mid = (lo + hi) / 2
            n_mid, groups = n_groups(mid)
            result['iterations'] = it
            if abs(n_mid - args.n_target) <= args.tol:
                result.update(converged=True, gamma=mid, n_groups=n_mid,
                              groups=[int(g) for g in groups])
                break
            elif n_mid < args.n_target:
                lo = mid
            else:
                hi = mid
        else:
            result['error'] = f'not converged within {args.max_iter} iterations'

    print(json.dumps(result))


if __name__ == '__main__':
    main()
