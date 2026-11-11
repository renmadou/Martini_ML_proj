import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MoleculeDataset, load_smiles_from_file
from torch_geometric.loader import DataLoader
from model.model import ModelCGMapping
from CGMapping_option import arg_parse
from utils import mkdir, flatten_and_offset_atom_group_indexs, compute_modularity_with_kernel

import math
import numpy as np
import torch
import torch.optim as optim
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import cairosvg



def clustering_results_to_communities(clustering_results):
    communities = []
    assert len(np.unique(clustering_results)) == max(clustering_results)+1
    for i in range(max(clustering_results)+1):
        communities.append([])
    for i in range(len(clustering_results)):
        communities[clustering_results[i]].append(i)
    return communities

def draw_graph(smiles, groups, save_path, caption=None):
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    molecule = AllChem.RemoveAllHs(molecule)
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(1200, 600)

    atom_index = list(range(molecule.GetNumAtoms()))

    # 为每个原子添加注释，显示序号
    for idx, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp('atomNote', f"{idx}")  # 注释为原子序号

    # 根据社区划分结果为原子设置颜色
    atom_colors = {}
    num_communities = len(groups)
    color_map = plt.get_cmap('tab10')  # 选择一个合适的颜色映射
    for i, community in enumerate(groups):
        for atom_idx in community:
            atom_colors[atom_idx] = tuple(color_map(i % num_communities)[:3])  # 提取 RGB 颜色

    # 获取仅在同一社区内的键
    highlight_bonds = []
    bond_colors = {}
    for i, community in enumerate(groups):
        community_set = set(community)
        for bond in molecule.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in community_set and end_idx in community_set:  # 仅保留同一社区内的键
                highlight_bonds.append(bond.GetIdx())
                bond_colors[bond.GetIdx()] = tuple(color_map(i % num_communities)[:3])  # 与社区颜色一致

    drawer.DrawMolecule(
        molecule,
        highlightAtoms=atom_index,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        # highlightBondColors={i: (0, 0, 0, 1) for i in non_cut_edges_indices},
        highlightAtomRadii=dict(zip(atom_index, [0.2] * len(atom_index)))
        # highlightAtomRadii=dict(zip(atom_index, [0.1] * len(atom_index)))
    )

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # 将图注添加到 SVG 文件下方
    if caption is not None:
        # 修复 RDKit 生成的 SVG 结构（去掉多余的 XML 声明部分）
        molecule_svg_content = svg.split("<svg")[1]
        molecule_svg_content = f"<svg{molecule_svg_content}"

        # 计算图注位置
        svg_width = 1200
        caption_font_size = 36
        caption_y = 660

        # 构建完整的 SVG 文件，添加居中图注
        svg = f"""
            <svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="700">
                <g transform="translate(0, 0)">
                    {molecule_svg_content}
                </g>
                <text x="{svg_width // 2}" y="{caption_y}" font-size="{caption_font_size}" fill="black" text-anchor="middle">{caption}</text>
            </svg>
            """
    svg_path = save_path + '.svg'
    png_path = save_path + '.png'
    with open(svg_path, 'w') as f:
        f.write(svg)

    # cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=500)

def main():
    # args = arg_parse()
    # args = torch.load(args.trained_ckpt)['args']



    # torch.cuda.set_device(0)
    # device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    # assert args.ckpt_dir is not None, '--ckpt is required'

    # smiles_vocab = load_smiles_vocab(args.vocabs_dir)
    # ckpt = '../../../test-dataset/test/ckpt/batch32_lr0.0001_numhiddenlayers5_group_pair_loss_ratio100_add_dist_ratio/best_model_epoch_1.pth'
    # ckpt = '../../../test-dataset/test/ckpt/leiden_triplet_pair_modularity/best_model_epoch_52.pth'
    ckpt = './1_epoch196.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = torch.load(ckpt)['args']
    edge_method = args.edge_method
    if args.edge_method in ['bonded+nonbonded', 'nonbonded']:
        args.edge_method = {'bonded+nonbonded': args.rcut}

    # smiles_datasets = '../../../test-dataset/vocab/mgssl_vocab.txt'
    resolution = 0.1
    max_iter = 200
    tolerance = 1
    save_folder = f'./resolution_{resolution}'
    mkdir(save_folder)


    smiles_list = ["COc1ccc(C2=CC(=O)C[C@@H](c3ccc(OC)cc3)C2)cc1", "COc1ccc2c(c1C)OC(=O)[C@@H](CC(=O)N[C@@H](CC1=c3ccccc3=[NH+]C1)C(=O)[O-])[C@H]2C", 'C=CCCOCC(=O)N(C)CCC#N']

    # smiles_list = load_smiles_from_file(file_path=args.dataset_dir, skip_lines=args.dataset_file_skip_lines)  # [:1000]

    datasets = MoleculeDataset(smiles_list, edge_method=args.edge_method, random_seed=args.seed)
    num_atom_features = datasets[0].atom_features.shape[1]
    dataloader = DataLoader(datasets, batch_size=1, num_workers=4, shuffle=True,
                            pin_memory=True)

    model = ModelCGMapping(num_atom_features=num_atom_features,
                           fg_num_hidden_layers=args.fg_num_hidden_layers,
                           fg_encoder_hidden_dim=args.fg_encoder_hidden_dim,
                           fg_encoder_output_dim=args.fg_encoder_output_dim,
                           cg_encoder_hidden_dim1=args.cg_encoder_hidden_dim1,
                           cg_encoder_hidden_dim2=args.cg_encoder_hidden_dim2,
                           cg_encoder_output_dim=args.cg_encoder_output_dim,
                           cg_encoder_model=args.cg_encoder_model,
                           clustering_method=args.clustering_method,
                           leiden_or_louvain_setting_dict=args.leiden_or_louvain_settings,
                           clustering_with_batch=args.clustering_with_batch,
                           edge_method=edge_method,
                           clustering_max_process=args.clustering_max_process,
                           seed=args.seed,
                           sigma=args.gau_sigma,
                           dropout=args.dropout,
                           init_method=args.init_method,
                           transformer_nhead=args.fg_transformer_nhead,
                           transformer_nlayer=args.fg_transformer_nlayer,
                           transformer_ffn_dim=args.fg_transformer_ffn_dim)

    model.to(device)

    model.load_state_dict(torch.load(ckpt)['model_state_dict'])

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for batch_id, data in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
            data = data.to(device)

            # if batch_id != 23:
            #     continue

            mol = Chem.MolFromSmiles(data.smiles[0])
            mol = Chem.AddHs(mol)
            mol = AllChem.RemoveAllHs(mol)

            Natom = mol.GetNumAtoms()
            Ntarget = math.ceil(Natom * resolution)
            left_cg_degree = -10.0
            right_cg_degree = 10.0

            for iter_id in range(max_iter):
                _, cluster_results = model(data, eval_flag=True, cg_degree=left_cg_degree)
                Ncluster = len(np.unique(cluster_results[0]))
                if Ncluster > Ntarget:
                    left_cg_degree += -10.0 * (iter_id + 1)
                else:
                    break
            if iter_id == max_iter - 1:
                print(f'can not find a reasonable left_cg_degree')
                continue

            for iter_id in range(max_iter):
                _, cluster_results = model(data, eval_flag=True, cg_degree=right_cg_degree)
                Ncluster = len(np.unique(cluster_results[0]))
                if Ncluster < Ntarget:
                    right_cg_degree += 10.0 * (iter_id + 1)
                else:
                    break
            if iter_id == max_iter - 1:
                print(f'can not find a reasonable left_cg_degree')
                continue

            for iter_id in range(max_iter):
                print(f'\n{data.smiles[0]}: {iter_id}/{max_iter}')
                mid_cg_degree = (left_cg_degree + right_cg_degree) / 2
                fg_features, cluster_results = model(data, eval_flag=True, cg_degree=mid_cg_degree)
                Ncluster_mid = len(np.unique(cluster_results[0]))

                if abs(Ncluster_mid - Ntarget) < tolerance:
                    break
                elif Ncluster_mid < Ntarget:
                    left_cg_degree = mid_cg_degree
                else:
                    right_cg_degree = mid_cg_degree

            if iter_id == max_iter - 1:
                print(f'Warning: {data.smiles[0]} can not reach the target cluster number {Ntarget}')
                continue

            cluster_results = flatten_and_offset_atom_group_indexs(cluster_results)
            modularity = compute_modularity_with_kernel(cluster_results=cluster_results, fg_features=fg_features,
                                                        edge_index=data.bonded_edge_index, sigma=1.0).item()
            # conductance = np.mean(model.conductance_list)

            # data = data.to(device)
            # print(data.smiles[0])
            #
            # # Forward pass
            # _, clustering_groups = model(data, eval_flag=True)
            # flatten_atom_group_labels = flatten_and_offset_atom_group_indexs(model.clusters)
            communities = clustering_results_to_communities(cluster_results)


            # Calculate metrics
            # accuracy, precision, recall, f1 = calculate_basic_metrics(true_labels=cg_group_gt.cpu().numpy(), predicted_labels=cg_group_pred.argmax(dim=1).cpu().numpy())
            # avg_modularity = float(model.modularity_list.mean())
            # avg_conductance = np.mean(model.condutance_list)
            caption = f'Modularity = {round(modularity, 2)}'
            save_path = os.path.join(save_folder, f'{batch_id}_{data.smiles[0]}')
            draw_graph(smiles=data.smiles[0], groups=communities, save_path=save_path, caption=caption)

            pass


if __name__ == '__main__':
    main()


