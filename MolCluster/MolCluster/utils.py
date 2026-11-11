import torch
from datetime import datetime
import os
import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rdkit import Chem
from itertools import islice
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import torch_geometric
from dataset import generate_scaffold


def current_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_time


def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def compute_modularity_with_kernel(cluster_results, fg_features, edge_index, gamma=1.0, sigma=None):
    """
    Compute the modularity of a graph using edge weights calculated via a Gaussian kernel.
    This version returns a PyTorch tensor for gradient computation.

    Parameters:
        cluster_results (Tensor): Shape (n,), community assignments for each node, e.g., [0, 0, 1, 1, 2].
        fg_features (Tensor): Shape (n, d), feature matrix for n nodes, where each node has d features.
        edge_index (Tensor): Shape (2, num_edges), tensor containing the edge indices.
        sigma (float): Standard deviation for the Gaussian kernel, controls the weight distribution.

    Returns:
        Tensor: The modularity value of the graph as a scalar tensor.
    """
    # Number of nodes and edges
    num_nodes = fg_features.size(0)

    # Extract features of the source and target nodes for each edge
    src_nodes = edge_index[0]  # Source node indices
    tgt_nodes = edge_index[1]  # Target node indices

    # Compute edge weights using the Gaussian kernel
    diff = fg_features[src_nodes] - fg_features[tgt_nodes]  # Pairwise feature differences
    if sigma is None:
        sigma = torch.std(diff, dim=0).mean()
    edge_weights = torch.exp(-torch.sum(diff ** 2, dim=1) / (2 * sigma ** 2))  # Gaussian kernel weights

    # Construct a sparse weighted adjacency matrix
    adj_matrix = torch.sparse_coo_tensor(edge_index, edge_weights, (num_nodes, num_nodes))

    # Calculate the degree of each node (sum of edge weights connected to the node)
    degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()  # Node degrees

    # Total weight of the graph (sum of all edge weights)
    m = torch.sum(degrees) / 2.0

    # Convert adjacency matrix to dense representation
    adj_dense = adj_matrix.to_dense()

    # Create a cluster indicator matrix
    cluster_matrix = torch.zeros(num_nodes, num_nodes, device=fg_features.device)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if cluster_results[i] == cluster_results[j]:
                cluster_matrix[i, j] = 1.0  # Same cluster

    # Compute modularity as a tensor
    modularity_matrix = adj_dense - gamma * torch.outer(degrees, degrees) / (2 * m)
    modularity = torch.sum(modularity_matrix * cluster_matrix) / (2 * m)

    return modularity

def compute_modularity_with_kernel_each_molecule(cluster_results, fg_features, edge_index, gamma=1.0, sigma=None):
    total_modularity = []
    offset = 0
    for cluster in cluster_results:
        cluster_offset = cluster + offset
        modularity = compute_modularity_with_kernel(cluster_offset, fg_features, edge_index, gamma, sigma)
        total_modularity.append(modularity)

        offset = offset + max(cluster) + 1

    return torch.sum(total_modularity) / len(total_modularity)

def flatten_and_offset_atom_group_indexs(groups):
    flat_groups = []  # To store the flattened 2D list of tensors
    offset = 0  # Initialize the offset as 0

    for group in groups:
        # Flatten each sublist into a 2D list and adjust offsets
        adjusted_tensor = group + offset  # Apply the current offset to each tensor
        flat_groups.extend(adjusted_tensor)  # Add to the flattened list
            # Update offset by the number of elements in the current group
        offset = max(flat_groups) + 1

    return flat_groups

def flatten_and_offset_atom_group_indexs_to_list(groups):
    flat_groups = []  # To store the flattened 2D list of tensors
    offset = 0  # Initialize the offset as 0

    for group in groups:
        # Flatten each sublist into a 2D list and adjust offsets
        adjusted_tensor = [i + offset for i in group]   # Apply the current offset to each tensor
        flat_groups.extend(adjusted_tensor)  # Add to the flattened list
            # Update offset by the number of elements in the current group
        offset = max(flat_groups) + 1

    return flat_groups

def load_smiles_vocab(file_path):
    """
    Load the SMILES vocabulary from a file and create a mapping from SMILES to labels.

    Parameters:
    - file_path (str): Path to the file containing SMILES strings.

    Returns:
    - smiles_to_label (dict): Dictionary mapping SMILES strings to label indices.
    """
    smiles_to_label = {}

    # Read SMILES from the file and assign a label index
    i = 0
    with open(file_path, 'r') as f:
        for index, line in enumerate(f):
            smiles = line.strip()  # Remove whitespace
            if smiles not in smiles_to_label.keys():
                smiles_to_label[smiles] = i  # Map SMILES to its index
                i += 1

    return smiles_to_label


def calculate_basic_metrics(true_labels, predicted_labels):
    """
    Calculate accuracy, precision, recall, and F1-score for multi-class classification.

    Parameters:
    - true_labels (list or numpy array): Ground truth class labels.
    - predicted_labels (list or numpy array): Predicted class labels.

    Returns:
    - accuracy (float): The accuracy of the predictions.
    - precision (float): The precision score of the predictions.
    - recall (float): The recall score of the predictions.
    - f1 (float): The F1 score of the predictions.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')  # Macro for per-class precision
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    return accuracy, precision, recall, f1


def calculate_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device):
    mx = max(clusters)
    MM = np.zeros((num_nodes, mx + 1))
    for i in range(len(clusters)):
        MM[i][clusters[i]] = 1
    MM = torch.tensor(MM).double().to(device)

    x = torch.matmul(torch.t(MM), torch_sparse_adj.double())
    x = torch.matmul(x, MM)
    x = torch.trace(x)

    y = torch.matmul(torch.t(MM), degree.double())
    y = torch.matmul(torch.t(y.unsqueeze(dim=0)), y.unsqueeze(dim=0))
    y = torch.trace(y)
    y = y / (2 * num_edges)
    return ((x - y) / (2 * num_edges)).item()


def calculate_conductance(clusters, Graph):

    assert len(np.unique(clusters)) == max(clusters) + 1, f'number of clusters {len(np.unique(clusters))} is not equal to max cluster {max(clusters)+1}'
    comms = [[] for i in range(max(clusters) + 1)]
    for i in range(len(clusters)):
        comms[clusters[i]].append(i)
    conductance=[]
    for com in comms:
        try:
            conductance.append(nx.conductance(Graph, com, weight='weight'))
        except:
            continue

    return conductance


class MetricsRecorder:
    def __init__(self, mode, num_batch, num_epoch, best_result_keys, keys, writer=None):
        """
        Initialize the MetricsRecorder.

        Parameters:
        - mode: The mode of operation, either 'train' or 'validation'.
        - num_batch: Total number of batches in an epoch.
        - num_epoch: Total number of epochs.
        - best_result_keys: Keys to track the best results (e.g., accuracy, F1-score).
        - keys: The keys representing different types of metrics to record (e.g., accuracy, precision, recall).
        - writer: TensorBoard writer to log the metric values (optional).
        """
        self.mode = mode  # train or validation
        self.writer = writer
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.keys = keys
        self.best_result_keys = best_result_keys
        self.best_epoch = None  # To keep track of the best epoch
        self.best_metric_sum = float('-inf')  # To keep track of the best metric sum

        # Initialize metric matrices
        self.key_to_mat_dict = {k: np.zeros((num_epoch, num_batch)) for k in keys}

    def update(self, epoch, batch, key_val_dict: dict):
        """
        Update the recorded metric values.

        Parameters:
        - epoch: The current epoch number.
        - batch: The current batch index.
        - key_val_dict: A dictionary containing metric names and their corresponding values.
        """
        for key, val in key_val_dict.items():
            self.key_to_mat_dict[key][epoch, batch] = val

        # Print average metrics so far for the current epoch
        print(f"\nUpdated metrics for {self.mode} - Epoch {epoch + 1}/{self.num_epoch}, Batch {batch + 1}/{self.num_batch}:")
        for key in self.keys:
            avg_metric_so_far = self.key_to_mat_dict[key][epoch, :batch + 1].mean()
            print(f"  {key}: {avg_metric_so_far:.4f}")

    def recoder_tb(self, epoch):
        """
        Record the average metrics to TensorBoard for the specified epoch.

        Parameters:
        - epoch: The current epoch number.
        """
        for key in self.keys:
            avg = self.key_to_mat_dict[key][epoch, :].mean()
            if self.writer:
                self.writer.add_scalar(f'{self.mode}/{key}', avg, epoch)

    def update_best_epoch(self, epoch):
        """
        Update the best epoch if the current metric sum is higher than the best recorded metrics sum.

        Parameters:
        - epoch: The current epoch number.
        """
        current_metric_sum = sum(self.key_to_mat_dict[key][epoch, :].mean() for key in self.best_result_keys)
        if current_metric_sum > self.best_metric_sum:
            self.best_metric_sum = current_metric_sum
            self.best_epoch = epoch

    def get_epoch_metric(self, epoch, metric_item):
        """
        Get the average value of a specific metric for the given epoch.

        Parameters:
        - epoch: The current epoch number.
        - metric_item: The metric to retrieve.
        """
        return self.key_to_mat_dict[metric_item][epoch, :].mean()

    def summary(self, epoch):
        """
        Print a summary of metric values for the current epoch.

        Parameters:
        - epoch (int): The current epoch number.
        """
        print(f"\nSummary for Epoch {epoch + 1}/{self.num_epoch}:")
        for key in self.keys:
            avg_metric = self.key_to_mat_dict[key][epoch, :].mean()
            print(f"  {key}: {avg_metric:.4f}")
        if self.best_epoch is not None:
            print(f"  Best epoch so far: {self.best_epoch + 1} with sum of best result keys: {self.best_metric_sum:.4f}")

        return self.best_epoch, self.best_metric_sum

def read_smarts_file(file_path):
    all_smarts = []
    with open(file_path, 'r') as file:
        for line in islice(file, 1, None):
            all_smarts.append(line.split(' ')[0])

    functional_groups = []
    for smarts in all_smarts:
        functional_group = Chem.MolFromSmarts(smarts)
        if functional_group is None:
            raise ValueError("Invalid SMARTS format")
        else:
            if functional_group.GetNumAtoms() > 1:
                functional_groups.append(functional_group)
    return functional_groups


def count_predefined_group_departure(predefined_smarts_groups, smiles_list, cluster_labels):

    # Use RDKit's GetSubstructMatches method to find matches for the functional group

    num_match_group = 0
    num_departure_group = 0
    for smiles, cluster_label in zip(smiles_list, cluster_labels):
        # Convert the SMILES string to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        mol = AllChem.RemoveAllHs(mol)

        for smarts in predefined_smarts_groups:
            matches = mol.GetSubstructMatches(smarts)

            if matches:
                num_match_group += len(matches)
                for i in matches:
                    if np.unique(cluster_label[list(i)]).size != 1:
                        num_departure_group += 1

                    # for j in range(len(i)-1):
                    #     if not cluster_label[i[j]] == cluster_label[i[j+1]]:
                    #         num_departure_group += 1
                    #         break
    if num_match_group:
        return num_departure_group, num_match_group
    else:
        return None


def check_mol_embed_configuration(smiles_list, label_list=None, random_seed=None, checkscaffold=False):

    valid_smiles_list = []
    valid_label_list = []
    unsuccess_smiles_list = []

    count = 1
    if label_list is not None:

        for smiles, label in zip(smiles_list, label_list):
            print(f"\n {count}/{len(smiles_list)} Embedding {smiles}...")
            if checkscaffold:
                try:
                    scaffold = generate_scaffold(smiles, include_chirality=True)
                except:
                    print(f'Error generating scaffold for molecule with SMILES {smiles}')
                    continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            if mol.GetNumAtoms() > 200:
                print(f"Too many atoms: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            # if mol.GetNumAtoms() < 2 or mol.GetBonds() is None:
            #     print(f"Too few atoms: {smiles}")
            #     unsuccess_smiles_list.append(smiles)
            #     continue
            #
            mol = Chem.AddHs(mol)
            # mol_noHs = AllChem.RemoveAllHs(mol)
            # if mol_noHs.GetNumAtoms() < 2 or mol_noHs.GetBonds() is None:
            #     print(f"Too few atoms: {smiles}")
            #     unsuccess_smiles_list.append(smiles)
            #     continue
            embed_params = AllChem.ETKDGv3()
            if random_seed is not None:
                embed_params.randomSeed = random_seed

            success1 = AllChem.EmbedMolecule(mol, embed_params) == 0
            if not success1:
                print(f"Embedding failed for: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            try:
                success2 = AllChem.UFFOptimizeMolecule(mol, maxIters=1000) == 0
            except Exception as e:
                print(f"Embedding failed for: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            if success2:
                valid_smiles_list.append(smiles)
                valid_label_list.append(label)
            else:
                print(f"UFF Optimization failed for: {smiles}")
                unsuccess_smiles_list.append(smiles)

                # success3 = AllChem.MMFFOptimizeMolecule(mol) == 0
                # if success3:
                #     valid_smiles_list.append(smiles)
                #     valid_label_list.append(label)
                # else:
                #     print(f"MMFF Optimization failed for: {smiles}")
                #     unsuccess_smiles_list.append(smiles)
            count += 1

        print(f"Number of valid SMILES: {len(valid_smiles_list)}/{len(smiles_list)}")
        return valid_smiles_list, valid_label_list

    else:
        for smiles in smiles_list:
            print(f"\n {count}/{len(smiles_list)} Embedding {smiles}...")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            if mol.GetNumAtoms() > 200:
                print(f"Too many atoms: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            mol = Chem.AddHs(mol)
            embed_params = AllChem.ETKDGv3()
            if random_seed is not None:
                embed_params.randomSeed = random_seed

            success1 = AllChem.EmbedMolecule(mol, embed_params) == 0
            if not success1:
                print(f"Embedding failed for: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            try:
                success2 = AllChem.UFFOptimizeMolecule(mol, maxIters=1000) == 0
            except Exception as e:
                print(f"Embedding failed for: {smiles}")
                unsuccess_smiles_list.append(smiles)
                continue

            if success2:
                valid_smiles_list.append(smiles)
            else:
                print(f"UFF Optimization failed for: {smiles}")
                unsuccess_smiles_list.append(smiles)
            count += 1

        print(f"Number of valid SMILES: {len(valid_smiles_list)}/{len(smiles_list)}")
        return valid_smiles_list



def get_dataset_and_settings(dataset_dir, return_label_name=False):
    import pandas as pd
    num_classification_dict = {'tox21': 12, 'muv': 17, 'sider': 27, 'clintox': 2, 'bace': 1, 'BBBP': 1, 'HIV': 1,
                               'toxcast': 617, 'qm9': 12, 'qm8': 12, 'qm7': 1, 'ESOL': 1, 'FreeSolv': 1,
                               'Lipophilicity': 1, 'PDBbind': 1}

    smiles_name_dict = {'tox21': 'smiles', 'muv': 'smiles', 'sider': 'smiles', 'clintox': 'smiles',
                        'bace': 'mol', 'BBBP': 'smiles', 'HIV': 'smiles', 'toxcast': 'smiles',
                        'qm9': 'smiles', 'qm8': 'smiles', 'qm7': 'smiles', 'ESOL': 'smiles',
                        'FreeSolv': 'smiles', 'Lipophilicity': 'smiles', 'PDBbind': 'smiles'}

    label_name_dict = {'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                                 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
                       'muv': ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689', 'MUV-692', 'MUV-712',
                               'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858',
                               'MUV-859'],
                       'sider': ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
                                 'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
                                 'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders',
                                 'Reproductive system and breast disorders',
                                 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                                 'General disorders and administration site conditions', 'Endocrine disorders',
                                 'Surgical and medical procedures', 'Vascular disorders',
                                 'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders',
                                 'Congenital, familial and genetic disorders', 'Infections and infestations',
                                 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
                                 'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions',
                                 'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders',
                                 'Injury, poisoning and procedural complications'],
                       'clintox': ['FDA_APPROVED', 'CT_TOX'],
                       'bace': ['Class'],
                       'BBBP': ['p_np'],
                       'HIV': ['HIV_active'],
                       'toxcast': ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive',
                         'APR_HepG2_CellCycleArrest_24h_dn',
                         'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
                         'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up',
                         'APR_HepG2_MitoMass_24h_dn', 'APR_HepG2_MitoMass_72h_dn',
                         'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
                         'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up',
                         'APR_HepG2_OxidativeStress_24h_up',
                         'APR_HepG2_OxidativeStress_72h_up',
                         'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up',
                         'APR_HepG2_p53Act_72h_up', 'ATG_AP_1_CIS_up', 'ATG_Ahr_CIS_up',
                         'ATG_BRE_CIS_up', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_up',
                         'ATG_DR4_LXR_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_EGR_CIS_up',
                         'ATG_ERE_CIS_up', 'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn',
                         'ATG_HIF1a_CIS_up', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn',
                         'ATG_ISRE_CIS_dn', 'ATG_MRE_CIS_up', 'ATG_NRF2_ARE_CIS_up',
                         'ATG_Oct_MLP_CIS_up', 'ATG_PBREM_CIS_up', 'ATG_PPARg_TRANS_up',
                         'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up',
                         'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up', 'ATG_RORE_CIS_up',
                         'ATG_RXRb_TRANS_up', 'ATG_SREBP_CIS_up', 'ATG_Sp1_CIS_up',
                         'ATG_TCF_b_cat_CIS_dn', 'ATG_VDRE_CIS_up', 'ATG_Xbp1_CIS_up',
                         'ATG_p53_CIS_dn', 'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down',
                         'BSK_3C_ICAM1_down', 'BSK_3C_IL8_down', 'BSK_3C_MCP1_down',
                         'BSK_3C_MIG_down', 'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down',
                         'BSK_3C_Thrombomodulin_up', 'BSK_3C_TissueFactor_down',
                         'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down', 'BSK_3C_uPAR_down',
                         'BSK_4H_Eotaxin3_down', 'BSK_4H_MCP1_down',
                         'BSK_4H_Pselectin_down', 'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down',
                         'BSK_4H_VEGFRII_down', 'BSK_4H_uPAR_down', 'BSK_BE3C_HLADR_down',
                         'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
                         'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down',
                         'BSK_BE3C_SRB_down', 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down',
                         'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down',
                         'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL8_down',
                         'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_MCP1_down',
                         'BSK_CASM3C_MCSF_down', 'BSK_CASM3C_MIG_down',
                         'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_SAA_down',
                         'BSK_CASM3C_SRB_down', 'BSK_CASM3C_Thrombomodulin_up',
                         'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
                         'BSK_CASM3C_uPAR_down', 'BSK_KF3CT_ICAM1_down',
                         'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down',
                         'BSK_KF3CT_MCP1_down', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down',
                         'BSK_KF3CT_TGFb1_down', 'BSK_KF3CT_TIMP2_down',
                         'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down',
                         'BSK_LPS_Eselectin_down', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL8_down',
                         'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down',
                         'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down',
                         'BSK_LPS_TissueFactor_down', 'BSK_LPS_VCAM1_down',
                         'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down', 'BSK_SAg_CD69_down',
                         'BSK_SAg_Eselectin_down', 'BSK_SAg_IL8_down', 'BSK_SAg_MCP1_down',
                         'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down',
                         'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down',
                         'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_IL8_down',
                         'BSK_hDFCGF_IP10_down', 'BSK_hDFCGF_MCSF_down',
                         'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down',
                         'BSK_hDFCGF_PAI1_down', 'BSK_hDFCGF_Proliferation_down',
                         'BSK_hDFCGF_SRB_down', 'BSK_hDFCGF_TIMP1_down',
                         'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn',
                         'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn',
                         'CEETOX_H295R_ESTRONE_dn', 'CEETOX_H295R_ESTRONE_up',
                         'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_NR_bER', 'NVS_NR_hER',
                         'NVS_NR_hPPARg', 'NVS_NR_hPXR', 'NVS_NR_mERa', 'OT_AR_ARSRC1_0960',
                         'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440', 'OT_ER_ERbERb_0480',
                         'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_FXR_FXRSRC1_0480',
                         'OT_NURR1_NURR1RXRa_0480', 'TOX21_ARE_BLA_agonist_ratio',
                         'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_LUC_MDAKB2_Antagonist',
                         'TOX21_AR_LUC_MDAKB2_Antagonist2', 'TOX21_AhR_LUC_Agonist',
                         'TOX21_Aromatase_Inhibition', 'TOX21_ERa_BLA_Antagonist_ratio',
                         'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_FXR_BLA_antagonist_ratio',
                         'TOX21_MMP_ratio_down', 'TOX21_TR_LUC_GH3_Antagonist',
                         'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p2_ch2',
                         'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                         'TOX21_p53_BLA_p3_ratio', 'TOX21_p53_BLA_p4_ratio',
                         'TOX21_p53_BLA_p5_ratio', 'Tanguay_ZF_120hpf_AXIS_up',
                         'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_JAW_up',
                         'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_PE_up',
                         'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_YSE_up'],
                       'qm9': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv'],
                       'qm8': ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
                               'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],  # potential bug  'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
                       'qm7': ['label'],
                       'ESOL': ['measured log solubility in mols per litre'],
                       'FreeSolv': ['expt'],
                       'Lipophilicity': ['exp'],
                       'PDBbind': ['pKi']}

    dataset, loss_fn, labels, task, spliter, metric = None, None, None, None, None, None

    for i in list(num_classification_dict.keys()):
        if i in dataset_dir:
            dataset = i
            break

    output_dim = num_classification_dict[dataset]
    smiles_name = smiles_name_dict[dataset]
    label_name = label_name_dict[dataset]
    if return_label_name:
        return smiles_name, label_name

    if dataset in ['tox21', 'muv', 'sider', 'clintox', 'bace', 'BBBP', 'HIV', 'toxcast']:
        loss_fn = torch.nn.BCELoss(reduction='mean')
        task = 'classification'
        metric = 'ROC-AUC'
        if dataset in ['PCBA']:
            metric = 'PRC-AUC'
    elif dataset in ['qm9', 'qm8', 'qm7']:
        loss_fn = torch.nn.L1Loss(reduction='mean')  # MAE loss
        task = 'regression'
        metric = 'MAE'
    elif dataset in ['ESOL', 'FreeSolv', 'Lipophilicity', 'PDBbind']:
        from model.loss import RMSELoss
        loss_fn = RMSELoss(reduction='mean')  # RMSE loss
        task = 'regression'
        metric = 'RMSE'

    spliter = 'random'
    if dataset in ['HIV', 'bace', 'BBBP']:
        spliter = 'scaffold'
    elif dataset == 'PDBbind':
        spliter = 'time'
    elif dataset == 'qm7':
        spliter = 'random' #'stratified'

    df = pd.read_csv(dataset_dir)
    smiles_list = df[smiles_name].tolist()
    labels = torch.tensor(df[label_name].fillna(0).values, dtype=torch.float32)
    return smiles_list, labels, output_dim, loss_fn, task, metric, spliter

def save_dataloader_to_csv(dataloader, output_path):
    """
    Save the dataset from a dataloader to a CSV file.

    Args:
        dataloader: The dataloader containing the dataset.
        output_path: Path to save the output CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open the CSV file for writing
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if type(dataloader.dataset[0]) == torch_geometric.data.data.Data:
            writer.writerow(["smiles"])
            for data in dataloader.dataset:
                smiles = data.smiles
                writer.writerow([smiles])

        elif len(dataloader.dataset[0]) == 2:
            # Write the header
            # writer.writerow(["smiles", "labels"])
            import pandas as pd

            smiles_name, label_name = get_dataset_and_settings(output_path, return_label_name=True)

            all_labels = []
            all_smiles = []

            # Iterate through the dataset and write the smiles and labels
            for data, labels in dataloader.dataset:
                all_smiles.append(data.smiles)
                all_labels.append(labels.tolist())

                # smiles = data.smiles
                # labels_list = labels.tolist()  # Convert tensor to list
                # writer.writerow([smiles, labels_list])

            df = pd.DataFrame(data=all_labels, columns=label_name)
            df.insert(0, smiles_name, all_smiles)
            df.to_csv(output_path, index=False)

from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import csr_matrix, coo_matrix, triu
def edge_cut_prec_recall_fscore(node_cluster_idx_pred, node_cluster_idx_gt, edge_index_lst):
    num_vertices = len(node_cluster_idx_pred)
    data = np.ones(edge_index_lst.shape[1])
    row = edge_index_lst[0]
    column = edge_index_lst[1]
    adj = coo_matrix((data, (row, column)), shape=(num_vertices, num_vertices))

    upper_adj = triu(adj, format='coo')

    num_edges = int(edge_index_lst.shape[1] / 2)

    # 1 if the edge is a cut, otherwise 0  edge_gt(ground truth) edge_pred(prediction)
    edge_pred, edge_gt = [-1 * np.ones(num_edges, dtype=np.int32) for _ in range(2)]

    for edge_idx, (u, v) in enumerate(zip(upper_adj.row, upper_adj.col)):
        edge_pred[edge_idx] = int(node_cluster_idx_pred[u] != node_cluster_idx_pred[v])
        edge_gt[edge_idx] = int(node_cluster_idx_gt[u] != node_cluster_idx_gt[v])

    prec, rec, fscore, _ = precision_recall_fscore_support(y_true=edge_gt, y_pred=edge_pred, average='binary')
    return prec, rec, fscore

def cg_type_prec_recall_fscore(real_result, predict_result):
    prec, rec, fscore, _ = precision_recall_fscore_support(real_result, predict_result, average='weighted')
    return prec, rec, fscore


def main():
    smiles = ['CCC12C3C4C5C6C7C8C9C%10C%11C%12C%13CC%14C%15CC%16C%17C%18C%19C%20C%21C%22C%23C%24C%25C%26C%27C%28C%29C%30C%31C%32C%33C%34C%35C%36C%37C%38C%39CCC%40C%41C%39C%38%39C%41%38C%40C%40C%41C%42%43CC%41%44C%40%38C%37%39C%36%44C%35%36C%42C%35%37C%43C%38C%39C%40C%41C%42C%43C3C13C%27(C%262C%251C4C52C%241C%231C62C72C%221C%211C82C92C%104C%115C%126C%14%13C%15%16C%176C%185C%194C%2012)C%281C%433C%422C%291C%301C%313C%324C%33(C%34%36%35)C%38%37C%394C%403C%4121']
    check_mol_embed_configuration(smiles, None, 7777)


    smarts_file = '../../test-dataset/functional_group_smarts.txt'
    read_smarts_file(smarts_file)

    vocab_file_path = '../../test-dataset/vocab/original_vocab.txt'
    smiles_vocab = load_smiles_vocab(vocab_file_path)

    # Convert the vocabulary to a tensor
    labels = torch.tensor(list(smiles_vocab.values()), dtype=torch.long)

    pass

if __name__ == '__main__':
    main()