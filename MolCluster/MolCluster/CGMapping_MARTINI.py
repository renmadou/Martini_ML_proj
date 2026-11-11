from dataset import MoleculeDataset, load_smiles_from_file, RandomSplitter, load_martini_json_file, MoleculeWithPropertyDataset
from model.model import ModelCGMapping, ModelCGMapping_MARTINI
from model.model_utils import Timer
from model.loss import LossRecorder, predefined_group_pair_loss, cut_triplet_loss, PairLoss
from CGMapping_option import arg_parse
from utils import mkdir, read_smarts_file, MetricsRecorder, check_mol_embed_configuration, flatten_and_offset_atom_group_indexs, flatten_and_offset_atom_group_indexs_to_list, compute_modularity_with_kernel,  edge_cut_prec_recall_fscore, cg_type_prec_recall_fscore

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import random
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.cluster import spectral_clustering


def graph_cuts(fg_embed, edge_index, num_cg, bandwidth=1.0, kernel='rbf', device=torch.device(0), random_state=7777):
    affinity = compute_affinity(fg_embed, edge_index, bandwidth, kernel, device)
    # print(affinity)
    pred_cg_idx = spectral_clustering(affinity.cpu().numpy(), n_clusters=num_cg, assign_labels='discretize',
                                          random_state=random_state)

    return pred_cg_idx, affinity


def compute_affinity(fg_embed, edge_index, bandwidth=1.0, kernel='rbf', device=torch.device(0)):
    if kernel == 'rbf':
        num_nodes = fg_embed.shape[0]
        fg_embed = fg_embed.to(device)
        pairwise_dist = torch.norm(fg_embed[edge_index[0]] - fg_embed[edge_index[1]], dim=1).to(torch.device(0))
        # pairwise_dist = torch.norm(fg_embed.reshape(n, 1, d) - fg_embed.reshape(1, n, d), dim=2).to(torch.device(0))

        pairwise_dist = pairwise_dist ** 2
        affinity = torch.exp(-pairwise_dist / (2 * bandwidth ** 2))
        # affinity = affinity * adj
        affinity = torch.sparse.LongTensor(edge_index, affinity, (num_nodes, num_nodes)).to_dense()
    elif kernel == 'linear':
        affinity = torch.nn.functional.relu(fg_embed @ fg_embed.t())
    else:
        assert False

    return affinity




def train(model, datasets, num_workers, val_ratio, test_ratio, optimizer, device, num_epochs, batch_size, tb_dir, argparse, test_dataset,
          ckpt_dir, smarts_groups, sigma, save_split_dataset, group_pair_loss_ratio, modularity_loss_ratio, triplet_loss_ratio, type_loss_ratio):
    """
        Train the neural network model with modularity loss and predefined group pair loss.

        Parameters:
        - model: The neural network model to be trained.
        - datasets: The input dataset, typically containing molecular graph data.
        - num_workers: Number of worker threads for data loading.
        - val_ratio: Fraction of data reserved for validation.
        - test_ratio: Fraction of data reserved for testing.
        - optimizer: Optimization algorithm for updating model weights.
        - device: Device (e.g., 'cpu' or 'cuda') to run the training.
        - num_epochs: Total number of training epochs.
        - batch_size: Number of samples per batch for training.
        - tb_dir: Directory for saving TensorBoard logs.
        - ckpt_dir: Directory for saving model checkpoints.
        - smarts_groups: Predefined SMARTS groups for enforcing chemical constraints.
        """

    # Split dataset into training and validation sets
    train_size = int((1 - val_ratio) * len(datasets))
    val_size = len(datasets) - train_size
    train_dataset, val_dataset = random_split(datasets, [train_size, val_size])

    # Create DataLoaders for training, validation and test
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False,
                                pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False,
                                pin_memory=True)

    num_batches = len(train_dataloader)

    mkdir(tb_dir)
    mkdir(ckpt_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    best_metric = float('-inf')

    # Initialize LossRecorder
    train_loss_recorder = LossRecorder(
        mode='Train',
        num_batch=len(train_dataloader),
        num_epoch=num_epochs,
        best_result_keys=['Loss'],  # Specify best result keys
        keys=['Loss', 'Predefined_group_pair_loss', 'Triplet_loss', 'CG_type_loss'],
        writer=writer
    )

    val_metric_recorder = MetricsRecorder(
        mode='Val',
        num_batch=1,
        num_epoch=num_epochs,
        best_result_keys=['AMI', 'Type_precision'],  # Specify best result keys
        keys=['Modularity', 'AMI', 'AMI+Type_precision', 'Cut_precision', 'Cut_recall', 'Cut_f1_score', 'Type_precision', 'Type_recall', 'Type_f1_score'],
        writer=writer
    )

    best_ckpt_path = None

    for epoch in tqdm.tqdm(range(0, num_epochs), total=num_epochs, dynamic_ncols=True):
        model.train()

        for batch_idx, data in enumerate(train_dataloader):
            print(f'\n---------------- Epoch: {epoch+1}/{num_epochs};  Batch: {batch_idx+1}/{num_batches} ----------------')

            data = data.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            fg_features, fg_cg_type_pred = model(data)

            cluster_results = [np.array(group) for group in data.group]

            triplet_loss = triplet_loss_ratio * cut_triplet_loss(fg_features=fg_features, cluster_results=cluster_results, margin=1.0, sigma=sigma)

            group_labels_flatten = flatten_and_offset_atom_group_indexs(cluster_results)
            pair_loss_fn = PairLoss()
            group_pair_loss = group_pair_loss_ratio * pair_loss_fn(fg_features=fg_features, group_labels=group_labels_flatten)

            criterion = torch.nn.CrossEntropyLoss()
            cg_type_loss = type_loss_ratio * criterion(fg_cg_type_pred, data.atom_CG_types.reshape(-1))


            # Total loss
            loss = group_pair_loss + triplet_loss + cg_type_loss

            train_loss_recorder.update(epoch=epoch, batch=batch_idx,
                                       key_val_dict={'Loss': loss.item(),
                                                     'Predefined_group_pair_loss': group_pair_loss.item(),
                                                     'Triplet_loss': triplet_loss.item(),
                                                     'CG_type_loss': cg_type_loss.item()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        train_loss_recorder.recoder_tb(epoch)
        train_loss_recorder.summary(epoch)

        #  Validation step after each epoch
        val(model, val_dataloader, device, epoch=epoch, recoder=val_metric_recorder)

        val_metric_recorder.recoder_tb(epoch)
        val_metric_recorder.summary(epoch)

        metric = val_metric_recorder.get_epoch_metric(epoch=epoch, metric_item='AMI') + val_metric_recorder.get_epoch_metric(epoch=epoch, metric_item='Type_precision')

        # Save model checkpoint if validation loss improves
        if metric > best_metric:
            best_metric = metric
            checkpoint_path = os.path.join(ckpt_dir, f'best_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': metric,
                'args': argparse,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

            best_ckpt_path = checkpoint_path

    test(model=model, dataloader=test_dataloader, device=device, ckpt=best_ckpt_path)

def val(model, dataloader, device, recoder, epoch):
    """
    Evaluates the model on a validation dataset.

    Args:
        model: The neural network model to be evaluated.
        dataloader: The DataLoader providing batches of validation data.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        recoder: An object to record and summarize evaluation metrics.
        epoch: The current epoch number, used for recording results.

    Returns:
        None
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        all_modularity, all_conductance = [], []
        all_ami, all_precision, all_recall, all_f_score = [], [], [], []
        all_type_precision, all_type_recall, all_type_f_score = [], [], []

        for batch_id, data in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
            data = data.to(device)
            edge_index_cpu = data.bonded_edge_index.cpu().numpy()

            # Forward pass
            fg_features, fg_cg_type_pred = model(data)

            # use spectral clustering
            cluster_results, _ = graph_cuts(fg_features, data.bonded_edge_index, len(np.unique(data.group[0])), 1.0)

            # cluster_results = flatten_and_offset_atom_group_indexs(cluster_results)
            modularity = compute_modularity_with_kernel(cluster_results=cluster_results, fg_features=fg_features,
                                                        edge_index=data.bonded_edge_index, sigma=1.0).item()
            # conductance = np.mean(model.conductance_list)

            gt_cluster = flatten_and_offset_atom_group_indexs_to_list(data.group)
            adjusted_mutual_info = metrics.adjusted_mutual_info_score(gt_cluster, cluster_results)
            precision, recall, f_score = edge_cut_prec_recall_fscore(cluster_results, gt_cluster, edge_index_cpu)

            softmax_output = torch.nn.functional.softmax(fg_cg_type_pred, dim=1)
            predicted_cg_types_id = torch.argmax(softmax_output.cpu(), dim=1)
            type_precision, type_recall, type_f_score = cg_type_prec_recall_fscore(
                predict_result=predicted_cg_types_id, real_result=data.atom_CG_types.reshape(-1).cpu())

            all_modularity.append(modularity)
            # all_conductance.append(conductance)
            all_ami.append(adjusted_mutual_info)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f_score.append(f_score)
            all_type_precision.append(type_precision)
            all_type_recall.append(type_recall)
            all_type_f_score.append(type_f_score)

        recoder.update(epoch=epoch, batch=0, key_val_dict={'Modularity': np.mean(all_modularity),
                                                           'AMI': np.mean(all_ami),
                                                           'Cut_precision': np.mean(all_precision),
                                                           'Cut_recall': np.mean(all_recall),
                                                           'Cut_f1_score': np.mean(all_f_score),
                                                           'Type_precision': np.mean(all_type_precision),
                                                           'Type_recall': np.mean(all_type_recall),
                                                           'Type_f1_score': np.mean(all_type_f_score),
                                                           'AMI+Type_precision':np.mean(all_ami)+np.mean(all_type_precision)})


def test(model, dataloader, device, ckpt):
    print('\n-----------------------Test Result------------------------------------')

    model.load_state_dict(torch.load(ckpt)['model_state_dict'])

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        all_modularity, all_conductance = [], []
        all_ami, all_precision, all_recall, all_f_score = [], [], [], []
        all_type_precision, all_type_recall, all_type_f_score = [], [], []

        for batch_id, data in tqdm.tqdm(enumerate(dataloader), desc="Testing", leave=False):
            data = data.to(device)
            edge_index_cpu = data.bonded_edge_index.cpu().numpy()

            # Forward pass
            fg_features, fg_cg_type_pred = model(data)

            # use spectral clustering
            cluster_results, _ = graph_cuts(fg_features, data.bonded_edge_index, len(np.unique(data.group[0])), 1.0)

            modularity = compute_modularity_with_kernel(cluster_results=cluster_results, fg_features=fg_features,
                                                        edge_index=data.bonded_edge_index, sigma=1.0).item()

            gt_cluster = flatten_and_offset_atom_group_indexs_to_list(data.group)
            adjusted_mutual_info = metrics.adjusted_mutual_info_score(gt_cluster, cluster_results)
            precision, recall, f_score = edge_cut_prec_recall_fscore(cluster_results, gt_cluster, edge_index_cpu)

            softmax_output = torch.nn.functional.softmax(fg_cg_type_pred, dim=1)
            predicted_cg_types_id = torch.argmax(softmax_output.cpu(), dim=1)
            type_precision, type_recall, type_f_score = cg_type_prec_recall_fscore(
                predict_result=predicted_cg_types_id, real_result=data.atom_CG_types.reshape(-1).cpu())

            all_modularity.append(modularity)
            # all_conductance.append(conductance)
            all_ami.append(adjusted_mutual_info)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f_score.append(f_score)
            all_type_precision.append(type_precision)
            all_type_recall.append(type_recall)
            all_type_f_score.append(type_f_score)

        print(f'Modularity: {np.mean(all_modularity):.6f}, '
              f'AMI: {np.mean(all_ami):.6f}, '
              f'Cut_precision: {np.mean(all_precision):.6f}, '
              f'Cut_recall: {np.mean(all_recall):.6f}, '
              f'Cut_f1_score: {np.mean(all_f_score):.6f}, '
              f'Type_precision: {np.mean(all_type_precision):.6f}, '
              f'Type_recall: {np.mean(all_type_recall):.6f}, '
              f'Type_f1_score: {np.mean(all_type_f_score):.6f}')





def main():
    timer = Timer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(0)
    args = arg_parse()
    edge_method = args.edge_method

    if args.edge_method in ['bonded+nonbonded', 'nonbonded']:
        args.edge_method = {'bonded+nonbonded': args.rcut}

    smarts_groups = read_smarts_file(args.smarts_file)

    train_dataset_path = '../../checked_conf_MARTINI2_Dataset/train'
    test_dataset_path = '../../checked_conf_MARTINI2_Dataset/test'

    smiles_list, labels_list = load_martini_json_file(train_dataset_path)

    # smiles_list, labels_list = check_mol_embed_configuration(smiles_list, label_list=labels_list, random_seed=args.seed)

    # args.edge_method = {'bonded+nonbonded': 0.1}
    datasets = MoleculeDataset(smiles_list, edge_method=args.edge_method, random_seed=7777)
    num_atom_features = datasets[0].atom_features.shape[1]

    datasets_with_labels = []
    for data, labels in zip(datasets, labels_list):
        data.group = labels[0]
        data.atom_CG_types = labels[1]
        datasets_with_labels.append(data)


    smiles_list_test, labels_list_test = load_martini_json_file(test_dataset_path)
    test_datasets = MoleculeDataset(smiles_list_test, edge_method=args.edge_method, random_seed=7777)
    test_datasets_with_labels = []
    for data, labels in zip(test_datasets, labels_list_test):
        data.group = labels[0]
        data.atom_CG_types = labels[1]
        test_datasets_with_labels.append(data)

    model = ModelCGMapping_MARTINI(num_atom_features=num_atom_features,
                                   fg_num_hidden_layers=args.fg_num_hidden_layers,
                                   fg_encoder_hidden_dim=args.fg_encoder_hidden_dim,
                                   fg_encoder_output_dim=args.fg_encoder_output_dim,
                                   cg_encoder_hidden_dim1=args.cg_encoder_hidden_dim1,
                                   cg_encoder_hidden_dim2=args.cg_encoder_hidden_dim2,
                                   cg_encoder_output_dim=args.cg_encoder_output_dim,
                                   cg_encoder_model=args.cg_encoder_model,
                                   clustering_method=args.clustering_method,
                                   edge_method=edge_method,
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

    if args.pretrained_ckpt is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrained_ckpt)['model_state_dict']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    train(model=model, datasets=datasets_with_labels, num_workers=args.num_workers, val_ratio=args.validation_split_ratio,
          test_ratio=args.test_split_ratio, optimizer=optimizer, device=device, num_epochs=args.epochs,
          batch_size=args.batch_size, tb_dir=args.tb_dir, ckpt_dir=args.ckpt_dir, smarts_groups=smarts_groups,
          sigma=args.gau_sigma, save_split_dataset=args.save_split_dataset_dir, group_pair_loss_ratio=args.group_pair_loss_ratio,
          triplet_loss_ratio=args.triplet_loss_ratio, modularity_loss_ratio=args.modularity_loss_ratio, type_loss_ratio=args.type_loss_ratio, argparse=args, test_dataset=test_datasets_with_labels)

    total_time = round(timer.record_time()/60, 2)
    print(f"\n Total training time：{total_time} min")


if __name__ == '__main__':
    main()


