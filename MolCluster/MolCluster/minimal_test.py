import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import MoleculeDataset, load_smiles_from_file
from model.model import ModelCGMapping
from utils import read_smarts_file, flatten_and_offset_atom_group_indexs, \
    compute_modularity_with_kernel, count_predefined_group_departure

CKPT = 'script/model.pth'                       # 自带的 checkpoint
SMILES_FILE = 'data/minimal_test.smi'           # 最小测试集
SMARTS_FILE = 'data/martini2_vocab-2.txt'  # 预定义官能团 SMARTS 文件；不需要可改成 None

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
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)  # 自有 checkpoint，含 args 命名空间
args = ckpt['args']
edge_method = args.edge_method
if args.edge_method in ['bonded+nonbonded', 'nonbonded']:
    args.edge_method = {'bonded+nonbonded': args.rcut}

smiles_list = load_smiles_from_file(SMILES_FILE)
datasets = MoleculeDataset(smiles_list, edge_method=args.edge_method, random_seed=args.seed)
num_atom_features = datasets[0].atom_features.shape[1]
dataloader = DataLoader(datasets, batch_size=len(datasets), shuffle=False)

model = ModelCGMapping(
    num_atom_features=num_atom_features,
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

smarts_groups = read_smarts_file(SMARTS_FILE) if SMARTS_FILE else []

all_mod, all_cond, all_dep, all_match = [], [], [], []
with torch.no_grad():
    for data in dataloader:
        data = data.to(device)
        fg_features, cluster_results = model(data, eval_flag=True)

        for smi, labels in zip(data.smiles, cluster_results):
            print(f'{smi} -> CG groups: {labels}')

        res = count_predefined_group_departure(smarts_groups, data.smiles, cluster_results)
        if res is not None:
            all_dep.append(res[0]); all_match.append(res[1])

        flat = flatten_and_offset_atom_group_indexs(cluster_results)
        mod = compute_modularity_with_kernel(flat, fg_features, data.edge_index, sigma=1.0).item()
        all_mod.append(mod)
        all_cond.append(np.mean(model.conductance_list))

print(f'\nmodularity: {np.mean(all_mod):.4f}')
print(f'conductance: {np.mean(all_cond):.4f}')
if all_match:
    print(f'preservation_rate: {1 - sum(all_dep) / sum(all_match):.4f}')
