# mapping
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dataset import MoleculeDataset, load_smiles_from_file, RandomSplitter
from model.model import ModelCGMapping
from model.model_utils import Timer
from model.loss import LossRecorder, predefined_group_pair_loss, cut_triplet_loss

from utils import mkdir, read_smarts_file, MetricsRecorder, check_mol_embed_configuration, \
    flatten_and_offset_atom_group_indexs, compute_modularity_with_kernel, count_predefined_group_departure

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import random
from torch_geometric.loader import DataLoader


## properties
from dataset import MoleculeWithPropertyDataset, RandomSplitter, ScaffoldSplitter, RandomScaffoldSplitter

from model.loss import LossRecorder

from utils import mkdir, MetricsRecorder, check_mol_embed_configuration, get_dataset_and_settings
from model.model import ModelProperty

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torch_geometric.loader import DataLoader
import numpy as np

def eval_mapping():
    # from CGMapping_option import arg_parse

    # args = arg_parse()
    ckpt = '../../../Result/Mapping/preservation_rate_grouppairlossratio/martinivocab/0.01_epoch8.pth'
    # ckpt = '../../../Result/Mapping/preservation_rate_grouppairlossratio/functonalvocab/100_epoch195.pth' #1-15
    args = torch.load(ckpt)['args']
    edge_method = args.edge_method
    if args.edge_method in ['bonded+nonbonded', 'nonbonded']:
        args.edge_method = {'bonded+nonbonded': args.rcut}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(0)
    # args = arg_parse()

    # args.trained_ckpt = '/home/xiaoyedi/data/work/research/Properties-Prediction/CG/test/ckpt/bace/best_model_epoch_69.pth'
    # args = torch.load(args.trained_ckpt)['args']
    args.test_dataset_dir = '../../../MoleculeNet_Dataset/split_dataset/zinc10000/test.csv'
    args.smarts_file = '../../../test-dataset/martini2_vocab-2.txt'
    # args.smarts_file = '../../../test-dataset/functional_group_smarts.txt'
    args.num_test = 1


    smarts_groups = read_smarts_file(args.smarts_file)

    smiles_list = load_smiles_from_file(file_path=args.test_dataset_dir, skip_lines=1)# [:1000]
    if not args.checked_dataset:
        random.shuffle(smiles_list)  # [:1000]
        smiles_list = check_mol_embed_configuration(smiles_list, label_list=None, random_seed=args.seed)
    datasets = MoleculeDataset(smiles_list, edge_method=args.edge_method, random_seed=args.seed)
    num_atom_features = datasets[0].atom_features.shape[1]

    dataloader = DataLoader(datasets, batch_size=32, num_workers=args.num_workers, shuffle=False,
                            pin_memory=True)
    args.init_method = None

    model = ModelCGMapping(num_atom_features=num_atom_features,
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
                           seed=args.seed,
                           sigma=args.gau_sigma,
                           dropout=args.dropout,
                           init_method=args.init_method,
                           transformer_nhead=args.fg_transformer_nhead,
                           transformer_nlayer=args.fg_transformer_nlayer,
                           transformer_ffn_dim=args.fg_transformer_ffn_dim)
    model.to(device)
    # model.load_state_dict(torch.load(ckpt)['model_state_dict'])

    # args.init_method = None
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(ckpt)['model_state_dict']
    #
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # Set up optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metric_dict = {'modularity': [], 'condutance': [], 'preservation_rate': []}

    for test_id in range(args.num_test):

        print(f'\n--------------------- Test {test_id+1}/{args.num_test} ---------------------')
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            all_modularity = []
            all_condutance = []
            all_num_departure_group, all_num_match_group = [], []

            for batch_id, data in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
                data = data.to(device)

                # Forward pass
                fg_features, cluster_results = model(data, eval_flag=True)

                count_predefined_group = count_predefined_group_departure(predefined_smarts_groups=smarts_groups, smiles_list=data.smiles,
                                                                          cluster_labels=cluster_results)
                if count_predefined_group is not None:
                    num_departure_group, num_match_group = count_predefined_group
                    all_num_departure_group.append(num_departure_group)
                    all_num_match_group.append(num_match_group)

                cluster_results = flatten_and_offset_atom_group_indexs(cluster_results)
                modularity = compute_modularity_with_kernel(cluster_results=cluster_results, fg_features=fg_features,
                                                            edge_index=data.edge_index, sigma=1.0).item()
                conductance = np.mean(model.conductance_list)

                all_modularity.append(modularity)
                all_condutance.append(conductance)

        metric_dict['modularity'].append(np.mean(all_modularity))
        metric_dict['condutance'].append(np.mean(all_condutance))
        metric_dict['preservation_rate'].append(1 - np.mean(sum(all_num_departure_group) / sum(all_num_match_group)))

        print(f"\nmodularity: {np.mean(all_modularity):.4f} ± {np.std(all_modularity):.4f}")
        print(f"condutance: {np.mean(all_condutance):.4f} ± {np.std(all_condutance):.4f}")
        print(f"preservation_rate: {1 - np.mean(sum(all_num_departure_group) / sum(all_num_match_group)):.4f} ± {np.std(sum(all_num_departure_group) / sum(all_num_match_group)):.4f}")

    for m in metric_dict.keys():
        print(f'\nAverage above {args.num_test} test,  {m}: {sum(metric_dict[m]) / args.num_test:.4f} ± {np.std(metric_dict[m]):.4f}')


def main():
    timer = Timer()

    # eval_propeties()
    eval_mapping()

    total_time = round(timer.record_time() / 60, 2)
    print(f"\n Total training time：{total_time} min")


if __name__ == '__main__':
    main()


