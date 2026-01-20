from dataset import MoleculeWithPropertyDataset, RandomSplitter, ScaffoldSplitter, RandomScaffoldSplitter

from model.loss import LossRecorder
from option import arg_parse
from utils import mkdir, MetricsRecorder, check_mol_embed_configuration, get_dataset_and_settings, save_dataloader_to_csv
from model.model import ModelProperty, ModelPropertyClusterPool
from model.model_utils import Timer

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

import numpy as np
import datetime


def train(model, datasets, num_workers, spliter, val_ratio, test_ratio, optimizer, device, num_epochs, batch_size,
          tb_dir, ckpt_dir, loss_fn, metric, seed, save_split_dataset, argparse):
    """
        Trains a model on the given dataset and evaluates its performance on validation and test sets.

        Args:
            model: The model to train.
            datasets: The dataset containing samples for training, validation, and testing.
            num_workers: Number of worker threads for data loading.
            spliter: The strategy to split datasets ('random' or 'scaffold').
            val_ratio: Ratio of samples used for validation.
            test_ratio: Ratio of samples used for testing.
            optimizer: Optimizer for training the model.
            device: Device to run the model on (e.g., 'cpu' or 'cuda').
            num_epochs: Number of epochs for training.
            batch_size: Batch size for data loading.
            tb_dir: Directory for TensorBoard logs.
            ckpt_dir: Directory to save model checkpoints.
            loss_fn: Loss function for training.
            metric: Evaluation metric to monitor during training (e.g., 'MAE', 'ROC-AUC').

        Returns:
            None
        """

    # Split dataset into training, validation and test sets
    train_dataloader, val_dataloader, test_dataloader = None, None, None

    if spliter == 'random':
        train_dataloader, val_dataloader, test_dataloader = RandomSplitter(datasets=datasets, val_ratio=val_ratio,
                                                                           test_ratio=test_ratio, batch_size=batch_size,
                                                                           num_workers=num_workers, seed=seed)
    elif spliter == 'scaffold':
        train_dataloader, val_dataloader, test_dataloader = RandomScaffoldSplitter(datasets=datasets, val_ratio=val_ratio,
                                                                                   test_ratio=test_ratio, batch_size=batch_size,
                                                                                   num_workers=num_workers, seed=seed)

    if save_split_dataset is not None:
        mkdir(save_split_dataset)
        for dataloader_item, dataloader_name in zip([train_dataloader, val_dataloader, test_dataloader],
                                                    ['train.csv', 'val.csv', 'test.csv']):
            save_path = os.path.join(save_split_dataset, dataloader_name)
            save_dataloader_to_csv(dataloader=dataloader_item, output_path=save_path)

    num_batches = len(train_dataloader)

    mkdir(tb_dir)
    mkdir(ckpt_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    # Initialize LossRecorder
    train_loss_recorder = LossRecorder(
        mode='Train',
        num_batch=len(train_dataloader),
        num_epoch=num_epochs,
        best_result_keys=['Loss'],  # Specify best result keys
        keys=['Loss'],
        writer=writer
    )

    best_eval_metric = None
    if metric in ['MAE', 'RMSE']:
        metrics = [metric]
        best_eval_metric = float('inf')
    elif metric in ['ROC-AUC', 'PRC-AUC']:
        metrics = [metric, 'F1_score', 'Accuracy', 'Recall', 'Precision']
        best_eval_metric = float('-inf')
    val_metric_recorder = MetricsRecorder(
        mode='Val',
        num_batch=1,
        num_epoch=num_epochs,
        best_result_keys=[metric],  # Specify best result keys
        keys=metrics,
        writer=writer
    )

    # a = train_dataloader.dataset[1]
    best_ckpt_path = None

    for epoch in tqdm.tqdm(range(0, num_epochs), total=num_epochs, dynamic_ncols=True):
        model.train()

        for batch_idx, batch_item in enumerate(train_dataloader):
            print(f'\n---------------- Epoch: {epoch+1}/{num_epochs};  Batch: {batch_idx+1}/{num_batches} ----------------')

            if batch_idx==1006 and epoch == 0:
                print(' ')

            data = batch_item[0].to(device)
            # print(data.smiles)
            data_label = batch_item[1].to(device)

            # assert len(data.batch.unique()) == data.batch_size  # qm7

            if len(data.batch.unique()) != data.batch_size:
                continue  # QM7

            if torch.isnan(data.edge_attr).any():
                continue  # tox21

            if len(data.edge_index) == 0:
                continue  # FreeSolv

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model(data)

            loss = loss_fn(pred, data_label)

            train_loss_recorder.update(epoch=epoch, batch=batch_idx, key_val_dict={'Loss': loss.item()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        train_loss_recorder.recoder_tb(epoch)
        train_loss_recorder.summary(epoch)

        #  Validation step after each epoch
        val(model, val_dataloader, device, epoch=epoch, recoder=val_metric_recorder, metric=metric)

        val_metric_recorder.recoder_tb(epoch)
        val_metric_recorder.summary(epoch)

        eval_metric = val_metric_recorder.get_epoch_metric(epoch=epoch, metric_item=metric)

        # Save model checkpoint if validation loss improves

        if ((eval_metric > best_eval_metric) and (metric in ['ROC-AUC', 'PRC-AUC'])) or ((eval_metric < best_eval_metric) and (metric in ['MAE', 'RMSE'])):
            best_eval_metric = eval_metric
            checkpoint_path = os.path.join(ckpt_dir, f'best_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': eval_metric,
                'args': argparse,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}.")

            best_ckpt_path = checkpoint_path

    result_file = os.path.join(tb_dir, 'results.pth')
    test(model=model, dataloader=test_dataloader, device=device, ckpt=best_ckpt_path, metric=metric, result_file=result_file)


def val(model, dataloader, device, recoder, epoch, metric):
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
    model.eval()  # Set model to evaluation mode

    # Disable gradient computation
    with torch.no_grad():
        if metric in ['ROC-AUC', 'PRC-AUC']:

            total_preds = torch.tensor([], device=device)
            total_labels = torch.tensor([], device=device)

            for batch_id, batch_item in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
                data = batch_item[0].to(device)
                data_label = batch_item[1].to(device)
                data_label = torch.where(data_label == -1,
                                         torch.zeros_like(data_label),
                                         data_label)

                pred = model(data)

                total_preds = torch.cat([total_preds, pred], dim=0)
                total_labels = torch.cat([total_labels, data_label], dim=0)

            total_labels = total_labels.cpu().numpy()
            total_preds = total_preds.cpu().numpy()

            macro_auc = 0.0
            macro_f1 = 0.0
            macro_precision = 0.0
            macro_recall = 0.0
            accuracy = 0.0

            num_classes = total_labels.shape[1]
            for class_idx in range(num_classes):

                y_true = total_labels[:, class_idx]
                y_pred = total_preds[:, class_idx]

                # 计算二值预测
                y_pred_binary = (y_pred >= 0.5).astype(int)

                macro_f1 += f1_score(y_true, y_pred_binary, zero_division=0)
                macro_precision += precision_score(y_true, y_pred_binary, zero_division=0)
                macro_recall += recall_score(y_true, y_pred_binary, zero_division=0)
                accuracy += accuracy_score(y_true, y_pred_binary)

                if len(np.unique(y_true)) >= 2:
                    if metric == 'ROC-AUC':
                        class_auc = roc_auc_score(y_true, y_pred)
                    elif metric == 'PRC-AUC':
                        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
                        class_auc = auc(recall_curve, precision_curve)
                    macro_auc += class_auc
                else:
                    raise ValueError(
                        f"Class {class_idx} has only one label")

            macro_auc /= num_classes
            macro_f1 /= num_classes
            macro_precision /= num_classes
            macro_recall /= num_classes
            accuracy /= num_classes

            recoder.update(
                epoch=epoch,
                batch=0,
                key_val_dict={
                    metric: macro_auc,
                    'F1_score': macro_f1,
                    'Accuracy': accuracy,
                    'Recall': macro_recall,
                    'Precision': macro_precision
                }
            )

        elif metric in ['MAE', 'RMSE']:
            total_preds = torch.tensor([], device=device)
            total_labels = torch.tensor([], device=device)

            for batch_id, batch_item in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
                data = batch_item[0].to(device)
                data_label = batch_item[1].to(device)

                pred = model(data)

                total_preds = torch.cat([total_preds, pred], dim=0)
                total_labels = torch.cat([total_labels, data_label], dim=0)

            total_labels = total_labels.cpu().numpy()
            total_preds = total_preds.cpu().numpy()

            if total_labels.ndim == 1:
                total_labels = total_labels.reshape(-1, 1)
                total_preds = total_preds.reshape(-1, 1)

            num_targets = total_labels.shape[1]

            target_mae = []
            target_rmse = []

            for target_idx in range(num_targets):
                y_true = total_labels[:, target_idx]
                y_pred = total_preds[:, target_idx]

                target_mae.append(mean_absolute_error(y_true, y_pred))
                target_rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))

            # global_mae = mean_absolute_error(total_labels, total_preds)
            # global_rmse = np.sqrt(mean_squared_error(total_labels, total_preds))

            if metric == 'MAE':
                mae_avg, mae_std = np.mean(target_mae), np.std(target_mae)
                recoder.update(epoch=epoch, batch=0, key_val_dict={metric: mae_avg})
            elif metric == 'RMSE':
                rmse_avg, rmse_std = np.mean(target_rmse), np.std(target_rmse)
                recoder.update(epoch=epoch, batch=0, key_val_dict={metric: rmse_avg})


def test(model, dataloader, device, ckpt, metric, result_file):

    model.load_state_dict(torch.load(ckpt)['model_state_dict'])

    model.eval()  # Set model to evaluation mode

    total_preds = torch.Tensor().to(device)   # To store predicted values
    total_labels = torch.Tensor().to(device)  # To store true labels

    results = {'timestamp': datetime.datetime.now().isoformat(), 'metric': metric, 'checkpoint': os.path.basename(ckpt),}

    # Disable gradient computation
    with torch.no_grad():
        print('\n-----------------------Test Result------------------------------------')
        if metric in ['ROC-AUC', 'PRC-AUC']:

            macro_metrics = {
                'AUC': [], 'F1': [],
                'Precision': [], 'Recall': [],
                'Accuracy': []
            }

            for batch_id, batch_item in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
                data = batch_item[0].to(device)
                data_label = batch_item[1].to(device)
                data_label = torch.where(data_label == -1,
                                         torch.zeros_like(data_label),
                                         data_label)

                pred = model(data)

                total_preds = torch.cat([total_preds, pred], dim=0)
                total_labels = torch.cat([total_labels, data_label], dim=0)

            total_labels = total_labels.cpu().numpy()
            total_preds = total_preds.cpu().numpy()

            num_classes = total_labels.shape[1]
            for class_idx in range(num_classes):

                y_true = total_labels[:, class_idx]
                y_pred = total_preds[:, class_idx]

                y_pred_binary = (y_pred >= 0.5).astype(int)

                macro_metrics['F1'].append(f1_score(y_true, y_pred_binary, zero_division=0))
                macro_metrics['Precision'].append(precision_score(y_true, y_pred_binary, zero_division=0))
                macro_metrics['Recall'].append(recall_score(y_true, y_pred_binary, zero_division=0))
                macro_metrics['Accuracy'].append(accuracy_score(y_true, y_pred_binary))

                if len(np.unique(y_true)) >= 2:
                    if metric == 'ROC-AUC':
                        class_auc = roc_auc_score(y_true, y_pred)
                    elif metric == 'PRC-AUC':
                        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
                        class_auc = auc(recall_curve, precision_curve)
                    macro_metrics['AUC'].append(class_auc)
                else:
                    raise ValueError(
                        f"Class {class_idx} has only one label")

                # Print formatted results
                print(f"Class-{class_idx} {metric} :  {macro_metrics['AUC'][-1]:.6f} | F1: {macro_metrics['F1'][-1]:.6f} | "
                      f"Precision: {macro_metrics['Precision'][-1]:.6f} | Recall: {macro_metrics['Recall'][-1]:.6f} | Accuracy: {macro_metrics['Accuracy'][-1]:.6f}")

            results.update(macro_metrics)
            metric_names = list(macro_metrics.keys())
            print('\n------------------Average Result----------------------')
            for name in metric_names:
                values = macro_metrics.get(name, [])

                avg = np.mean(values)
                std = np.std(values)

                results[f'{name}_avg'] = float(avg)
                results[f'{name}_std'] = float(std)

                if name == 'AUC' and metric == 'PRC-AUC':
                    display_name = 'PRC-AUC'
                else:
                    display_name = name

                print_str = f"{display_name} :  {avg:.6f} ± {std:.6f}"
                print(print_str)

        elif metric in ['MAE', 'RMSE']:

            macro_metrics = {
                'MAE': [],
                'RMSE': [],
                'R2': []
            }

            for batch_id, batch_item in tqdm.tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
                data = batch_item[0].to(device)
                data_label = batch_item[1].to(device)

                pred = model(data)
                total_preds = torch.cat((total_preds, pred), 0)
                total_labels = torch.cat((total_labels, data_label), 0)

            total_labels = total_labels.cpu().numpy()
            total_preds = total_preds.cpu().numpy()

            if total_labels.ndim == 1:
                total_labels = total_labels.reshape(-1, 1)
                total_preds = total_preds.reshape(-1, 1)

            num_targets = total_labels.shape[1]

            for target_idx in range(num_targets):
                y_true = total_labels[:, target_idx]
                y_pred = total_preds[:, target_idx]

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                try:
                    r2 = r2_score(y_true, y_pred)
                except:
                    r2 = np.nan

                macro_metrics['MAE'].append(mae)
                macro_metrics['RMSE'].append(rmse)
                macro_metrics['R2'].append(r2)

                print(f'Target-{target_idx}  MAE: {mae:.6f} | RMSE: {rmse:.6f} | R2: {r2:.6f}')

            results.update(macro_metrics)

            print('\n------------------Average Results----------------------')
            metric_names = list(macro_metrics.keys())
            for name in metric_names:
                values = macro_metrics.get(name, [])

                avg = np.mean(values)
                std = np.std(values)

                results[f'{name}_avg'] = float(avg)
                results[f'{name}_std'] = float(std)

                print_str = f"{name} :  {avg:.6f} ± {std:.6f}"
                print(print_str)

        torch.save(results, result_file)

def main():
    timer = Timer()

    args = arg_parse()
    edge_method = args.edge_method
    if args.edge_method in ['bonded+nonbonded', 'nonbonded']:
        args.edge_method = {'bonded+nonbonded': args.rcut}

    # torch.cuda.set_device(0)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    smiles_list, labels, output_dim, loss_fn, task, metric, spliter = get_dataset_and_settings(args.dataset_dir)

    if not args.checked_dataset:
        smiles_list, labels = check_mol_embed_configuration(smiles_list, labels, args.seed)

    first_elapsed_time = timer.record_time()
    # print(f"\nfilter dataset time：{first_elapsed_time} s")

    datasets = MoleculeWithPropertyDataset(smiles_list=smiles_list, label_list=labels,
                                           edge_method=args.edge_method, random_seed=7777)
    num_atom_features = datasets[0][0].atom_features.shape[1]

    model = ModelProperty(num_atom_features=num_atom_features,
                          fg_num_hidden_layers=args.fg_num_hidden_layers,
                          fg_encoder_hidden_dim=args.fg_encoder_hidden_dim,
                          fg_encoder_output_dim=output_dim,
                          task=task,
                          init_method=args.init_method,
                          seed=args.seed,
                          sigma=args.gau_sigma,
                          dropout=args.dropout,
                          edge_method=edge_method,
                          transformer_nhead=args.fg_transformer_nhead,
                          transformer_nlayer=args.fg_transformer_nlayer,
                          transformer_ffn_dim=args.fg_transformer_ffn_dim,
                          )

    model.to(device)

    # load pretrained model
    if args.pretrained_ckpt is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrained_ckpt, map_location=torch.device(device))['model_state_dict']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    train(model=model, datasets=datasets, num_workers=args.num_workers, spliter=spliter,
          val_ratio=args.validation_split_ratio, test_ratio=args.test_split_ratio,
          optimizer=optimizer, device=device, num_epochs=args.epochs, batch_size=args.batch_size,
          tb_dir=args.tb_dir, ckpt_dir=args.ckpt_dir, loss_fn=loss_fn, metric=metric, seed=args.seed,
          save_split_dataset=args.save_split_dataset_dir, argparse=args)
    total_time = round(timer.record_time() / 60, 2)
    print(f"\n Total training time：{total_time} min")


if __name__ == '__main__':
    main()


