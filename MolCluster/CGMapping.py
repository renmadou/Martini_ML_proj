from dataset import MoleculeDataset, load_smiles_from_file, RandomSplitter
from model.model import ModelCGMapping
from model.model_utils import Timer
from model.loss import LossRecorder, predefined_group_pair_loss, cut_triplet_loss
from CGMapping_option import arg_parse
from utils import mkdir, read_smarts_file, MetricsRecorder, check_mol_embed_configuration, flatten_and_offset_atom_group_indexs, compute_modularity_with_kernel, compute_modularity_with_kernel_each_molecule, save_dataloader_to_csv, get_dataset_and_settings

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import random
from torch.optim.lr_scheduler import StepLR


def train(model, datasets, num_workers, val_ratio, test_ratio, optimizer, device, num_epochs, batch_size, tb_dir, seed,
          ckpt_dir, smarts_groups, sigma, save_split_dataset, group_pair_loss_ratio, modularity_loss_ratio, triplet_loss_ratio, argparse):
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
    train_dataloader, val_dataloader, test_dataloader = RandomSplitter(datasets=datasets, val_ratio=val_ratio,
                                                                       test_ratio=test_ratio, batch_size=batch_size,
                                                                       num_workers=num_workers, seed=seed)
    if save_split_dataset is not None:
        mkdir(save_split_dataset)
        for dataloader_item, dataloader_name in zip([train_dataloader, val_dataloader, test_dataloader], ['train.csv', 'val.csv', 'test.csv']):
            save_path = os.path.join(save_split_dataset, dataloader_name)
            save_dataloader_to_csv(dataloader=dataloader_item, output_path=save_path)

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
        keys=['Loss', 'Modularity_loss', 'Triplet_loss', 'Predefined_group_pair_loss'],
        writer=writer
    )

    val_metric_recorder = MetricsRecorder(
        mode='Val',
        num_batch=len(val_dataloader),
        num_epoch=num_epochs,
        best_result_keys=['Modularity'],  # Specify best result keys
        keys=['Modularity', 'Conductance'],
        writer=writer
    )

    for epoch in tqdm.tqdm(range(0, num_epochs), total=num_epochs, dynamic_ncols=True):
        model.train()

        for batch_idx, data in enumerate(train_dataloader):
            print(f'\n---------------- Epoch: {epoch+1}/{num_epochs};  Batch: {batch_idx+1}/{num_batches} ----------------')

            data = data.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            fg_features, cluster_results = model(data, eval_flag=False)

            triplet_loss = triplet_loss_ratio * cut_triplet_loss(fg_features=fg_features, cluster_results=cluster_results, margin=1.0, sigma=sigma)
            # modularity = compute_modularity_with_kernel_each_molecule(cluster_results=cluster_results, fg_features=fg_features, edge_index=data.bonded_edge_index, sigma=sigma)

            cluster_results = flatten_and_offset_atom_group_indexs(cluster_results)
            modularity = compute_modularity_with_kernel(cluster_results=cluster_results, fg_features=fg_features, edge_index=data.bonded_edge_index, gamma=1.0, sigma=sigma)
            modularity_loss = modularity_loss_ratio * (1 - modularity)

            group_pair_loss = group_pair_loss_ratio * predefined_group_pair_loss(predefined_smarts_groups=smarts_groups, smiles_list=data.smiles, fg_features=fg_features, sigma=sigma)


            # Total loss
            loss = modularity_loss + triplet_loss + group_pair_loss

            train_loss_recorder.update(epoch=epoch, batch=batch_idx,
                                       key_val_dict={'Loss': loss.item(),
                                                     'Predefined_group_pair_loss': group_pair_loss.item(),
                                                     'Modularity_loss': modularity_loss.item(),
                                                     'Triplet_loss': triplet_loss.item()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        train_loss_recorder.recoder_tb(epoch)
        train_loss_recorder.summary(epoch)

        #  Validation step after each epoch
        eval(model, val_dataloader, device, epoch=epoch, recoder=val_metric_recorder)

        val_metric_recorder.recoder_tb(epoch)
        val_metric_recorder.summary(epoch)

        metric = val_metric_recorder.get_epoch_metric(epoch=epoch, metric_item='Modularity')

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


def eval(model, dataloader, device, recoder, epoch):
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
        for batch_id, data in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
            data = data.to(device)

            # Forward pass
            fg_features, cluster_results = model(data, eval_flag=True)
            cluster_results = flatten_and_offset_atom_group_indexs(cluster_results)
            modularity = compute_modularity_with_kernel(cluster_results=cluster_results, fg_features=fg_features,
                                                        edge_index=data.edge_index, sigma=1.0).item()
            conductance = np.mean(model.conductance_list)

            recoder.update(epoch=epoch, batch=batch_id, key_val_dict={'Modularity': modularity,
                                                                      'Conductance': conductance})


def main():
    timer = Timer()

    args = arg_parse()
    edge_method = args.edge_method
    if args.edge_method in ['bonded+nonbonded', 'nonbonded']:
        args.edge_method = {'bonded+nonbonded': args.rcut}

    # torch.cuda.set_device(0)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    smarts_groups = read_smarts_file(args.smarts_file)

    smiles_list = load_smiles_from_file(file_path=args.dataset_dir, skip_lines=args.dataset_file_skip_lines)  # [:1000]
    if not args.checked_dataset:
        random.shuffle(smiles_list)  # [:1000]
        smiles_list = check_mol_embed_configuration(smiles_list, label_list=None, random_seed=7777)
    datasets = MoleculeDataset(smiles_list, edge_method=args.edge_method, random_seed=7777)
    num_atom_features = datasets[0].atom_features.shape[1]

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

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    train(model=model, datasets=datasets, num_workers=args.num_workers, val_ratio=args.validation_split_ratio,
          test_ratio=args.test_split_ratio, optimizer=optimizer, device=device, num_epochs=args.epochs,
          batch_size=args.batch_size, tb_dir=args.tb_dir, ckpt_dir=args.ckpt_dir, smarts_groups=smarts_groups,
          sigma=args.gau_sigma, save_split_dataset=args.save_split_dataset_dir, group_pair_loss_ratio=args.group_pair_loss_ratio,
          triplet_loss_ratio=args.triplet_loss_ratio, modularity_loss_ratio=args.modularity_loss_ratio, argparse=args, seed=args.seed)

    total_time = round(timer.record_time()/60, 2)
    print(f"\n Total training time：{total_time} min")


if __name__ == '__main__':
    main()


