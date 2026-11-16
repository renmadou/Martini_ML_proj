"""Run inside the molcluster conda env: Week-3 compatibility check (no gamma tuning).

For each "<Name> <SMILES>" line in the input file:
  1. build the molecular graph (record atom/edge counts)
  2. run a single forward pass at a fixed gamma (default 1.0) and record n_groups
  3. match the predefined SMARTS vocab (data/martini2_vocab-2.txt) against the molecule
  4. verify the group output is readable (one group label per atom index)

Usage:
    conda run -n molcluster python pipeline/molcluster_compat.py molecules.txt [--gamma 1.0]

Prints one JSON line per molecule to stdout.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MolCluster', 'MolCluster'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MolCluster', 'MolCluster'))

import numpy as np
import torch
from rdkit import Chem

from molcluster_bisect import build_model, get_device  # noqa: E402  (sys.path set above)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils import read_smarts_file  # noqa: E402

SMARTS_FILE = 'data/martini2_vocab-2.txt'


def smarts_coverage(smiles):
    mol = Chem.MolFromSmiles(smiles)
    names = []
    with open(SMARTS_FILE) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line:
                names.append(line.split(None, 1))
    hits = {}
    for smarts, name in names:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        n = len(mol.GetSubstructMatches(patt))
        if n:
            hits[name] = n
    return hits


def check(name, smiles, gamma):
    rec = {'name': name, 'smiles': smiles}
    try:
        device = get_device()
        rec['device'] = str(device)
        p = argparse.Namespace(smiles=smiles, n_target=1)
        model, data = build_model(p, device)
        rec['n_atoms'] = int(data.num_nodes)
        rec['n_edges'] = int(data.edge_index.shape[1]) if data.edge_index is not None else 0
        with torch.no_grad():
            _, cluster_results = model(data, eval_flag=True, cg_degree=gamma)
        groups = cluster_results[0]
        rec['forward_ok'] = True
        rec['n_groups'] = int(len(np.unique(groups)))
        rec['groups_readable'] = bool(len(groups) == data.num_nodes)
        rec['groups'] = [int(g) for g in groups]
        rec['smarts_coverage'] = smarts_coverage(smiles)
    except Exception as e:
        rec['forward_ok'] = False
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('molecules', help='text file: "<Name> <SMILES>" per line')
    p.add_argument('--gamma', type=float, default=1.0)
    args = p.parse_args()

    with open(args.molecules) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, smiles = line.split(None, 1)
            print(json.dumps(check(name, smiles.strip(), args.gamma)), flush=True)


if __name__ == '__main__':
    main()
