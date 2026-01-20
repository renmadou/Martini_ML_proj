import argparse
import random
import numpy as np
import torch


def arg_parse():
    parser = argparse.ArgumentParser('MolCluster')

    parser.add_argument('--gpu_idx', type=int, help='GPU device index')

    parser.add_argument('--title', type=str, default='CG')
    parser.add_argument('--checked_dataset', type=bool, help='if ture, will not check dataset again and please provide the checked dataset file by --dataset_dir')
    parser.add_argument('--dataset_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset_file_skip_lines', type=int, help='skipping lines of the dataset file')
    parser.add_argument('--tb_dir', help='tensorboard log directory')
    parser.add_argument('--ckpt_dir', type=str, help='checkpoint directory')
    parser.add_argument('--save_split_dataset_dir', type=str, help='the directory to save the split dataset(train, validation, test)')
    parser.add_argument('--pretrained_ckpt', type=str, help='trained ckpt .pth for finetune.py')

    parser.add_argument('--trained_ckpt', type=str, help='trained ckpt .pth (only for test.py)')
    parser.add_argument('--test_dataset_dir', type=str, help='test dataset directory (only for test.py)')

    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='number of epoch to train')
    parser.add_argument('--validation_split_ratio', type=float, help='validation split ratio')
    parser.add_argument('--test_split_ratio', type=float, help='test split ratio')
    parser.add_argument('--num_test', type=int, help='number of test')
    parser.add_argument('--num_workers', type=int, help='number of workers to load data')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--seed', type=int, help='random seed')

    parser.add_argument('--edge_method', help='edge method for graph construction')
    parser.add_argument('--rcut', type=float, help='cut off radius for graph construction')

    parser.add_argument('--batch_norm', type=bool, help='whether to use batch normalization')
    parser.add_argument('--init_method', type=str, help='initialization method for model parameters', choices=['kaiming', 'xavier', 'None'])

    parser.add_argument('--gau_sigma', type=float, help='gaussian kernel sigma', default=None)

    parser.add_argument('--fg_encoder_model', type=str, help='fine grained encoder graph network model',
                        choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--fg_encoder_hidden_dim', type=int, help='fine grained encoder hidden dimension')
    parser.add_argument('--fg_encoder_output_dim', type=int, help='fine grained encoder output dimension')
    parser.add_argument('--fg_num_hidden_layers', type=int, help='number of fg hidden layers')
    parser.add_argument('--dropout', type=float, help='dropout rate')

    parser.add_argument('--fg_transformer_nhead', type=int, help='number of fg transformer self-attention heads')
    parser.add_argument('--fg_transformer_nlayer', type=int, help='number of fg transformer layers')
    parser.add_argument('--fg_transformer_ffn_dim', type=int, help='number of fg transformer feed forward network dimension')

    parser.add_argument('--cg_encoder_hidden_dim1', type=int, help='coarse grained encoder hidden dimension in first layer')
    parser.add_argument('--cg_encoder_hidden_dim2', type=int, help='coarse grained encoder hidden dimension in second layer')
    parser.add_argument('--cg_encoder_output_dim', type=int, help='coarse grained encoder output dimension')

    parser.add_argument('--cg_encoder_model', type=str, help='coarse grained encoder graph network model',
                        choices=['gcn', 'gat', 'gin', 'sage', None])
    parser.add_argument('--group_pair_loss_ratio', type=float, help='group pair loss ratio')
    parser.add_argument('--modularity_loss_ratio', type=float, help='modularity loss ratio')
    parser.add_argument('--triplet_loss_ratio', type=float, help='triplet loss ratio')
    parser.add_argument('--type_loss_ratio', type=float, help='CG type loss ratio')
    parser.add_argument('--clustering_method', type=str, help='clustering method',
                        choices=['leiden', 'louvain', 'finch', 'optics', 'dbscan', 'birch'])
    parser.add_argument('--clustering_max_process', type=int, help='clustering max process for parallel computing')
    parser.add_argument('--leiden_or_louvain_settings', type=dict, help='settings of leiden/louvain clustering')
    parser.add_argument('--smarts_file', type=str, help='predefined SMARTS gruops file')

    parser.set_defaults(seed=7777,
                        lr=1e-3,
                        batch_size=32,
                        epochs=100,
                        validation_split_ratio=0.1,
                        test_split_ratio=0.1,
                        num_test=5,
                        num_workers=8,

                        gpu_idx=0,

                        edge_method='bonded+nonbonded',  # bonded, nonbonded, bonded+nonbonded
                        rcut=100.0,  # cutoff radius in Angstrom
                        batch_norm=True,
                        init_method='xavier',  # 'kaiming', 'xavier', None

                        gau_sigma=1.0,

                        fg_encoder_model='gat',  # 'gcn', 'gat', 'gin', 'sage'
                        fg_encoder_hidden_dim=32,
                        fg_encoder_output_dim=64,
                        fg_num_hidden_layers=2,
                        dropout=0.0,

                        fg_transformer_nhead=4,
                        fg_transformer_nlayer=3,
                        fg_transformer_ffn_dim=64,

                        cg_encoder_model=None,  # 'gcn', 'gat', 'gin', 'sage', None
                        cg_encoder_hidden_dim1=256,
                        cg_encoder_hidden_dim2=128,
                        cg_encoder_output_dim=64,

                        checked_dataset=True,
                        dataset_dir='../../checked_Dataset/bace.csv', # '../../MoleculeNet_Dataset/Classification/muv.csv',  #'../../test-dataset/zinc.txt',  # '../../test-dataset/small_shot.txt', # zinc.txt',  # hgraph2graph_smiles.txt
                        dataset_file_skip_lines=0,
                        ckpt_dir='../../test-dataset/test/ckpt',
                        tb_dir='../../test-dataset/test/tb_log',
                        save_split_dataset_dir='../../test-dataset/split_dataset/bace', # '../../test-dataset/split_dataset',  # '../../test-dataset/split_dataset',
                        pretrained_ckpt=None,  # '../../test-dataset/pretrained/best_model_epoch_0.pth',  #None

                        # trained_ckpt='../../test-dataset/test/ckpt/leiden_triplet_pair_modularity/best_model_epoch_52.pth',
                        trained_ckpt='../../test-dataset/test/ckpt/best_model_epoch_0.pth',
                        test_dataset_dir='/home/xiaoyedi/data/work/research/Properties-Prediction/CG/MoleculeNet_Dataset/All_split_dataset/bace_dataset/test.csv',

                        # clustering task
                        group_pair_loss_ratio=1.0,
                        modularity_loss_ratio=1.0,
                        triplet_loss_ratio=1.0,
                        type_loss_ratio=1.0,
                        clustering_method='leiden',  # 'leiden', 'louvain', 'finch', 'optics', 'dbscan', 'birch'
                        clustering_max_process=16,
                        leiden_or_louvain_settings={'edge_for_leiden_or_louvain': 'bond',  #{'rcut': 5.0},  # rcut (Angstrom)
                                                    'edge_weighted_function': 'gaussian',
                                                    'gaussian_sigma': 1.0,
                                                    'cg_degree(gamma)': 1.0,
                                                    'gamma2': 0.05,
                                                    'theta': 0.3,
                                                    'max_iter': 20},
                        smarts_file='../../test-dataset/martini2_vocab-2.txt' # '../../test-dataset/functional_group_smarts.txt',  # hgraph2graph_vocab.txt
                        )

    args = parser.parse_args()

    if args.seed is not None:
        # seed = int(time.time() * 256)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    return args