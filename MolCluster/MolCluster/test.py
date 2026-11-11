# mapping
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

def eval_propeties():
    """
        Evaluate the model performance based on the given metric.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.
            device (torch.device): The device (CPU or GPU) on which evaluation should run.
            recoder (object): A recorder object to log evaluation results.
            epoch (int): The current epoch number (used for logging).
            metric (str): The evaluation metric, one of ['ROC-AUC', 'PRC-AUC', 'MAE', 'RMSE'].

        Returns:
            None
        """
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(0)

    from option import arg_parse
    args = arg_parse()

    trained_ckpt = '/home/xiaoyedi/data/work/research/Properties-Prediction/CG/test/ckpt/bace/best_model_epoch_69.pth'
    args = torch.load(trained_ckpt)['args']
    args.test_dataset_dir = arg_parse().test_dataset_dir

    args.num_test = 5

    seeds = [7777, 23, 56789, 6, 102456]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    smiles_list, labels, output_dim, loss_fn, task, metric, spliter = get_dataset_and_settings(args.test_dataset_dir)

    metric_dict = {}
    if metric in ['ROC-AUC', 'PRC-AUC']:
        metric_dict = {metric: [], 'F1_score': [], 'Accuracy': [], 'Recall': [], 'Precision': []}
    elif metric in ['MAE', 'RMSE']:
        metric_dict[metric] = []

    for test_id in range(args.num_test):

        print(f'\n--------------------- Test {test_id+1}/{args.num_test} ---------------------')

        args.seed = seeds[test_id]

        if not args.checked_dataset:
            smiles_list, labels = check_mol_embed_configuration(smiles_list, labels, args.seed)

        # smiles_list, labels = check_mol_embed_configuration(smiles_list[:50], labels, args.seed)

        datasets = MoleculeWithPropertyDataset(smiles_list=smiles_list, label_list=labels,
                                               edge_method=args.edge_method, random_seed=7777)
        num_atom_features = datasets[0][0].atom_features.shape[1]

        dataloader = DataLoader(datasets, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                pin_memory=True)

        model = ModelProperty(num_atom_features=num_atom_features,
                              fg_num_hidden_layers=args.fg_num_hidden_layers,
                              fg_encoder_hidden_dim=args.fg_encoder_hidden_dim,
                              fg_encoder_output_dim=output_dim,
                              task=task,
                              seed=args.seed,
                              sigma=args.gau_sigma,
                              dropout=args.dropout,
                              init_method=args.init_method,
                              transformer_nhead=args.fg_transformer_nhead,
                              transformer_nlayer=args.fg_transformer_nlayer,
                              transformer_ffn_dim=args.fg_transformer_ffn_dim)
        model.to(device)

        # Set up optimizer and loss function
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        model.load_state_dict(torch.load(trained_ckpt)['model_state_dict'])


        model.eval()  # Set model to evaluation mode

        total_preds = torch.Tensor().to(device)   # To store predicted values
        total_labels = torch.Tensor().to(device)  # To store true labels

        # Disable gradient computation
        with torch.no_grad():
            if metric in ['ROC-AUC', 'PRC-AUC']:
                for batch_id, batch_item in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
                    data = batch_item[0].to(device)
                    data_label = batch_item[1].to(device)

                    pred = model(data)

                    # Handle invalid labels (-1 indicates missing or invalid labels)
                    c_valid = (data_label != -1)
                    data_label = torch.where(data_label == -1, torch.zeros_like(data_label), data_label)
                    # Accumulate valid predictions and labels
                    total_preds = torch.cat((total_preds, pred[c_valid]), 0)
                    total_labels = torch.cat((total_labels, data_label[c_valid]), 0)

                total_labels = total_labels.cpu().numpy().flatten()
                total_preds = total_preds.cpu().numpy().flatten()
                binary_preds = (total_preds >= 0.5).astype(int)

                # Calculate classification metrics
                f1 = f1_score(total_labels, binary_preds)
                accuracy = accuracy_score(total_labels, binary_preds)
                precision = precision_score(total_labels, binary_preds)
                recall = recall_score(total_labels, binary_preds, zero_division=0)

                if metric == 'ROC-AUC':
                    AUC = roc_auc_score(total_labels, total_preds)
                elif metric == 'PRC-AUC':
                    precision_curve, recall_curve, _ = precision_recall_curve(total_labels, binary_preds)
                    AUC = auc(recall_curve, precision_curve)

                metric_dict['F1_score'].append(f1)
                metric_dict['Accuracy'].append(accuracy)
                metric_dict['Precision'].append(precision)
                metric_dict['Recall'].append(recall)
                metric_dict[metric].append(AUC)

                print(f"Test {test_id+1}/{args.num_test}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, {metric}: {AUC:.4f}")


            elif metric in ['MAE', 'RMSE']:
                for batch_id, batch_item in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
                    data = batch_item[0].to(device)
                    data_label = batch_item[1].to(device)

                    pred = model(data)
                    total_preds = torch.cat((total_preds, pred), 0)
                    total_labels = torch.cat((total_labels, data_label), 0)

                # Calculate the chosen regression metric (MAE or RMSE)
                total_labels = total_labels.cpu().numpy().flatten()
                total_preds = total_preds.cpu().numpy().flatten()
                if metric == 'MAE':
                    mae = mean_absolute_error(total_labels, total_preds)

                elif metric == 'RMSE':
                    rmse = mean_squared_error(total_labels, total_preds, squared=False)
                metric_dict[metric].append(mae) if metric == 'MAE' else metric_dict[metric].append(rmse)

                print(f"\nTest {test_id+1}/{args.num_test}, {metric}: {metric_dict[metric][test_id]:.4f}")

    for m in metric_dict.keys():
        print(f'\nAverage above {args.num_test} test,  {m}: {sum(metric_dict[m])/args.num_test:.4f} ± {np.std(metric_dict[m]):.4f}')


def eval_mapping():
    from CGMapping_option import arg_parse
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(0)
    args = arg_parse()

    # args.trained_ckpt = '/home/xiaoyedi/data/work/research/Properties-Prediction/CG/test/ckpt/bace/best_model_epoch_69.pth'
    args = torch.load(args.trained_ckpt)['args']
    args.test_dataset_dir = arg_parse().test_dataset_dir

    smarts_groups = read_smarts_file(args.smarts_file)

    smiles_list = load_smiles_from_file(file_path=args.test_dataset_dir, skip_lines=args.dataset_file_skip_lines)# [:1000]
    if not args.checked_dataset:
        random.shuffle(smiles_list)  # [:1000]
        smiles_list = check_mol_embed_configuration(smiles_list, label_list=None, random_seed=args.seed)
    datasets = MoleculeDataset(smiles_list, edge_method=args.edge_method, random_seed=args.seed)
    num_atom_features = datasets[0].atom_features.shape[1]

    dataloader = DataLoader(datasets, batch_size=1, num_workers=args.num_workers, shuffle=False,
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
                           clustering_max_process=args.clustering_max_process,
                           seed=args.seed,
                           sigma=args.gau_sigma,
                           dropout=args.dropout,
                           init_method=args.init_method,
                           transformer_nhead=args.fg_transformer_nhead,
                           transformer_nlayer=args.fg_transformer_nlayer,
                           transformer_ffn_dim=args.fg_transformer_ffn_dim)
    model.to(device)
    model.load_state_dict(torch.load(args.trained_ckpt)['model_state_dict'])

    # Set up optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metric_dict = {'modularity': [], 'condutance': [], 'departure_ratio': []}

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
        metric_dict['departure_ratio'].append(np.mean(sum(all_num_departure_group) / sum(all_num_match_group)))

        print(f"\nmodularity: {np.mean(all_modularity):.4f} ± {np.std(all_modularity):.4f}")
        print(f"condutance: {np.mean(all_condutance):.4f} ± {np.std(all_condutance):.4f}")
        print(f"departure_ratio: {np.mean(sum(all_num_departure_group) / sum(all_num_match_group)):.4f} ± {np.std(sum(all_num_departure_group) / sum(all_num_match_group)):.4f}")

    for m in metric_dict.keys():
        print(f'\nAverage above {args.num_test} test,  {m}: {sum(metric_dict[m]) / args.num_test:.4f} ± {np.std(metric_dict[m]):.4f}')


def main():
    timer = Timer()

    eval_propeties()
    # eval_mapping()

    total_time = round(timer.record_time() / 60, 2)
    print(f"\n Total training time：{total_time} min")


if __name__ == '__main__':
    main()


