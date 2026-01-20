import torch
import torch.nn as nn
import random
import networkx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class LossRecorder:
    def __init__(self, mode, num_batch, num_epoch, best_result_keys, keys, writer=None):
        """
        Initialize the LossRecorder.

        Parameters:
        - mode: The mode of operation, either 'train' or 'validation'.
        - num_batch: Total number of batches in an epoch.
        - num_epoch: Total number of epochs.
        - best_result_keys: Keys to track the best results.
        - keys: The keys representing different types of losses to record.
        - writer: TensorBoard writer to log the loss values (optional).
        """
        self.mode = mode  # train or validation
        self.writer = writer
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.keys = keys
        self.best_result_keys = best_result_keys
        self.best_epoch = None  # To keep track of the best epoch
        self.best_loss = float('inf')  # To keep track of the best loss

        self.key_to_mat_dict = {}
        for k in keys:
            self.key_to_mat_dict[k] = np.zeros((num_epoch, num_batch))

    def update(self, epoch, batch, key_val_dict: dict):
        """
        Update the recorded loss values.

        Parameters:
        - epoch: The current epoch number.
        - batch: The current batch index.
        - key_val_dict: A dictionary containing loss names and their corresponding values.
        """
        for key, val in key_val_dict.items():
            self.key_to_mat_dict[key][epoch, batch] = val

        # Print average losses so far for the current epoch
        print(f"\nUpdated losses for {self.mode} - Epoch {epoch + 1}/{self.num_epoch}, Batch {batch + 1}/{self.num_batch}:")
        for key in self.keys:
            avg_loss_so_far = self.key_to_mat_dict[key][epoch, :batch + 1].mean()
            print(f"  {key}: {avg_loss_so_far:.4f}")

    def recoder_tb(self, epoch):
        """
        Record the average losses to TensorBoard for the specified epoch.

        Parameters:
        - epoch: The current epoch number.
        """
        for key in self.keys:
            avg = self.key_to_mat_dict[key][epoch, :].mean()
            self.writer.add_scalar(f'{self.mode}/{key}', avg, epoch)

    def update_best_epoch(self, epoch, current_losses):
        """
        Update the best epoch if the current losses are lower than the best recorded losses.

        Parameters:
        - epoch: The current epoch number.
        - current_losses: A dictionary containing the current losses to compare.
        """
        current_loss_sum = sum(self.key_to_mat_dict[key][epoch, :].mean() for key in self.best_result_keys)
        if current_loss_sum < self.best_loss:
            self.best_loss = current_loss_sum
            self.best_epoch = epoch

    def get_epoch_loss(self, epoch, loss_item):
        return self.key_to_mat_dict[loss_item][epoch, :].mean()

    def summary(self, epoch):
        """
        summary of loss values for the current epoch.

        Parameters:
        - epoch (int): The current epoch number.
        """
        print(f"\nSummary for Epoch {epoch + 1}/{self.num_epoch}:")
        for key in self.keys:
            avg_loss = self.key_to_mat_dict[key][epoch, :].mean()
            print(f"  {key}: {avg_loss:.4f}")
        if self.best_epoch is not None:
            print(f"  Best epoch so far: {self.best_epoch+1} with sum of best result keys: {self.best_loss:.4f}")

        return self.best_epoch, self.best_loss


class CoarseGrainedLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(CoarseGrainedLoss, self).__init__()
        # Initialize CrossEntropyLoss with ignore_index to skip invalid labels (-1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predicted_coarse_grained_groups, cg_group_labels):
        """
        Compute the loss for predicted coarse-grained groups.

        Parameters:
        - predicted_coarse_grained_groups: The predicted logits, shape [batch_size, vocab_size]
        - cg_group_labels: Ground truth labels, shape [batch_size], where -1 represents ignored labels

        Returns:
        - loss: The calculated cross-entropy loss
        """
        # Calculate the cross-entropy loss
        loss = self.criterion(predicted_coarse_grained_groups, cg_group_labels)
        return loss


def modularity_loss(output, num_nodes, sparse_adj, degree, num_edges,  device):
    """
    Calculate the modularity loss. https://arxiv.org/abs/2312.12697

    This function computes the modularity loss based on a sampled subset of nodes.
    The modularity loss is related to the difference between the actual connections
    of nodes in a community and the expected connections based on node degrees,
    similar to the concept described in the "Modularity via Embedding Similarity" section.

    In that section, to compute modularity Q, we focus on a binary matrix M (which is NP-Hard to optimize directly).
    Instead, we use a similarity matrix based on node embeddings here. This similarity matrix is related to the
    actual connections of nodes (similar to the role of M), and we calculate values related to it and the expected
    connections based on degrees in this function.

    Args:
        output: The output embeddings of the nodes, similar to the node embeddings X described in the paper.
        num_nodes: The total number of nodes in the graph, used for sampling and scaling calculations.
        sparse_adj: The sparse adjacency matrix representing the graph, related to the adjacency matrix concept in modularity calculation.
        degree: The degree vector of the nodes, used to calculate the expected connections based on degrees.
        num_edges: The total number of edges in the graph, used for normalization in modularity loss calculation.
        device: The device (e.g., 'cpu' or 'cuda') on which to perform the calculations.

    Returns:
        The calculated modularity loss.
    """

    sample_size = int(1 * num_nodes)
    s = random.sample(range(0, num_nodes), sample_size)

    s_output = output[s, :]

    s_adj = sparse_adj[s, :][:, s]
    s_adj = convert_scipy_torch_sp(s_adj)
    s_degree = degree[s]

    x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))
    x = torch.matmul(x, s_output.double())
    x = torch.trace(x)

    y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
    y = (y ** 2).sum()
    y = y / (2 * num_edges)

    scaling=1
    # scaling = num_nodes ** 2 / (sample_size ** 2)

    m_loss = -((x - y) / (2 * num_edges)) * scaling

    # aux_loss = lam * aux_objective(output, s)

    return m_loss


def regularization_loss(output, num_nodes):
    ss = output.sum(dim=0)
    avg_embedding = ss / num_nodes
    reg_loss = (avg_embedding ** 2).sum()
    return reg_loss

    # sample_size = random.randint(1, num_nodes)  # int(1 * num_nodes)
    # s = random.sample(range(0, num_nodes), sample_size)
    # out = output[s, :]
    # ss = out.sum(dim=0)
    # ss = ss ** 2
    # ss = ss.sum()
    # avg_sim = ss / (len(s) ** 2)
    #
    # return avg_sim ** 2


def predefined_group_departure_loss(predefined_smarts_groups, smiles_list, cluster_labels):

    # Use RDKit's GetSubstructMatches method to find matches for the functional group

    num_match_smiles = 0
    num_departure_group = 0
    for smiles, cluster_label in zip(smiles_list, cluster_labels):
        # Convert the SMILES string to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)

        for smarts in predefined_smarts_groups:
            matches = mol.GetSubstructMatches(smarts)

            if matches:
                num_match_smiles += 1
                for i in matches:
                    if np.unique(cluster_label[list(i)]).size != 1:
                        num_departure_group += 1

                    # for j in range(len(i)-1):
                    #     if not cluster_label[i[j]] == cluster_label[i[j+1]]:
                    #         num_departure_group += 1
                    #         break
    if num_match_smiles:
        return torch.tensor(num_departure_group / num_match_smiles)
    else:
        return torch.tensor(0.0)


def cut_triplet_loss(fg_features, cluster_results, margin=1.0, sigma=None):
    """
    Compute cut triplet loss based on node features and clustering results.

    Parameters:
        fg_features (Tensor): Shape (n, d), feature matrix for all nodes.
        cluster_results (List[Tensor]): A list of tensors, each containing community assignments for nodes in a graph.
        margin (float): Margin parameter for triplet loss.

    Returns:
        Tensor: The calculated triplet loss.
    """
    loss = 0.0
    total_triplets = 0
    offset = 0

    for i, cluster in enumerate(cluster_results):
        num_nodes = len(cluster)
        mol_features = fg_features[offset:num_nodes+offset]
        for node in range(num_nodes):
            node_feat = mol_features[node]
            pos_mask = (cluster == cluster[node])
            neg_mask = (cluster != cluster[node])

            # Positive samples: Nodes in the same cluster
            pos_samples = mol_features[pos_mask]

            # Negative samples: Nodes in different clusters
            neg_samples = mol_features[neg_mask]

            if len(pos_samples) == 0 or len(neg_samples) == 0:
                continue  # Skip if no positive or negative samples exist

            # Compute Gaussian kernel similarities
            pos_diff = pos_samples - node_feat
            neg_diff = neg_samples - node_feat

            if sigma is None:
                sigma = torch.std(torch.cat([pos_diff, neg_diff], dim=0)).item()

            pos_sim = torch.exp(-torch.sum(pos_diff ** 2, dim=1) / (2 * sigma ** 2))
            neg_sim = torch.exp(-torch.sum(neg_diff ** 2, dim=1) / (2 * sigma ** 2))

            pos_sim_matrix = pos_sim.unsqueeze(1)  # (P, 1)
            neg_sim_matrix = neg_sim.unsqueeze(0)  # (1, N)
            loss_matrix = torch.relu(neg_sim_matrix - pos_sim_matrix + margin)
            loss += loss_matrix.sum()
            total_triplets += pos_sim.shape[0] * neg_sim.shape[0]

            # Triplet loss computation
            # for ps in pos_sim:
            #     for ns in neg_sim:
            #         triplet_loss = torch.relu(ns - ps + margin)  # Negative similarity for positive samples
            #         loss += triplet_loss
            #         total_triplets += 1
            # pass

        offset += num_nodes
    # Average the loss over all triplets
    if total_triplets > 0:
        loss = loss / total_triplets

    return loss


def cut_triplet_loss_mse(fg_features, cluster_results, margin=1.0):

    loss = 0.0
    total_triplets = 0
    offset = 0

    for i, cluster in enumerate(cluster_results):
        num_nodes = len(cluster)
        mol_features = fg_features[offset:num_nodes+offset]
        for node in range(num_nodes):
            node_feat = mol_features[node]
            pos_mask = (cluster == cluster[node])
            neg_mask = (cluster != cluster[node])

            # Positive samples: Nodes in the same cluster
            pos_samples = mol_features[pos_mask]

            # Negative samples: Nodes in different clusters
            neg_samples = mol_features[neg_mask]

            if len(pos_samples) == 0 or len(neg_samples) == 0:
                continue  # Skip if no positive or negative samples exist

            # Compute Gaussian kernel similarities
            pos_diff = pos_samples - node_feat
            neg_diff = neg_samples - node_feat

            pos_sim = torch.sum(pos_diff ** 2, dim=1)
            neg_sim = torch.sum(neg_diff ** 2, dim=1)

            pos_sim_matrix = pos_sim.unsqueeze(1)  # (P, 1)
            neg_sim_matrix = neg_sim.unsqueeze(0)  # (1, N)
            loss_matrix = torch.relu(pos_sim_matrix - neg_sim_matrix + margin)
            loss += loss_matrix.sum()
            total_triplets += pos_sim.shape[0] * neg_sim.shape[0]

        offset += num_nodes
    if total_triplets > 0:
        loss = loss / total_triplets

    return loss



def predefined_group_pair_loss(predefined_smarts_groups, smiles_list, fg_features, sigma=None):
    """
    Compute the loss for pairs of atoms within predefined SMARTS functional groups using the Gaussian kernel.

    Args:
        predefined_smarts_groups (list): A list of RDKit SMARTS patterns representing functional groups.
        smiles_list (list): A list of SMILES strings for the molecules in the batch.
        fg_features (torch.Tensor): A tensor of shape (N, D) where N is the number of atoms
                                     in the batch and D is the feature dimension.
        sigma (float): The width of the Gaussian kernel, controls the similarity.

    Returns:
        torch.Tensor: A scalar tensor representing the pair loss for functional group pairs.
    """
    # Offset to track the atom indices across molecules
    num_atom_offset = 0
    predefined_group_pairs = []  # List to store atom index pairs

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        mol = AllChem.RemoveAllHs(mol)

        # Iterate over all predefined SMARTS patterns
        for smarts in predefined_smarts_groups:
            # Find all matches for the current SMARTS pattern
            matches = mol.GetSubstructMatches(smarts)
            if matches:
                # Flatten the matches and adjust for atom index offset
                # matches = [atom_id + num_atom_offset for match in matches for atom_id in match]
                for match in matches:
                    match_offset = [atom_id + num_atom_offset for atom_id in match]
                    for i in range(len(match_offset)):
                        for j in range(i + 1, len(match_offset)):
                            pair = [match_offset[i], match_offset[j]]
                            if (pair not in predefined_group_pairs) and (pair[::-1] not in predefined_group_pairs):
                                predefined_group_pairs.append(pair)

                # Generate all unique pairs of matched atoms
                # for i in range(len(matches)):
                #     for j in range(i + 1, len(matches)):
                #         predefined_group_pairs.append([matches[i], matches[j]])

        # Update the offset for atom indices
        num_atom_offset += mol.GetNumAtoms()

    # Compute the pair loss based on fg_features using Gaussian kernel
    if predefined_group_pairs:
        predefined_group_pairs = torch.tensor(predefined_group_pairs, dtype=torch.long, device=fg_features.device)
        atom_i = fg_features[predefined_group_pairs[:, 0]]
        atom_j = fg_features[predefined_group_pairs[:, 1]]

        # Calculate the squared Euclidean distance between atom pairs
        diff = atom_i - atom_j

        dist_sq = torch.sum((atom_i - atom_j) ** 2, dim=-1)
        if sigma is None:
            sigma = torch.std(diff, dim=0).mean()
        # Apply the Gaussian kernel formula
        pairwise_sim = torch.exp(-dist_sq / (2 * sigma ** 2))

        # Loss is the negative mean of Gaussian kernel similarities (maximize similarity)
        pair_loss = 1 - pairwise_sim.mean()
    else:
        pair_loss = torch.tensor(0.0, device=fg_features.device)

    return pair_loss


def get_pair_indices(labels):

    label_tensor = torch.tensor(labels, dtype=torch.long)
    pairs = []
    unique_labels = label_tensor.unique()

    for label in unique_labels:
        indices = (label_tensor == label).nonzero(as_tuple=True)[0]
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pairs.append((indices[i].item(), indices[j].item()))

    return torch.tensor(pairs, dtype=torch.long).T if pairs else torch.empty((2, 0), dtype=torch.long)


class PairLoss(nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, fg_features, group_labels):
        pair_indices = get_pair_indices(group_labels)
        if pair_indices.size(1) == 0:
            return torch.tensor(0.0, device=fg_features.device)

        pos1_embed = fg_features[pair_indices[0]]
        pos2_embed = fg_features[pair_indices[1]]

        loss = self.mse_loss(pos1_embed, pos2_embed)
        return loss

def cal_modularity_weighed_graph(partition, G:networkx.Graph):
    m = sum(weight for _, _, weight in G.edges(data='weight', default=1.0))
    a = []
    e = []

    for community in partition:
        t = 0.0
        for node in community:
            t += G.degree(node, weight='weight')
        a.append(t / (2 * m))

    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if G.has_edge(community[i], community[j]):
                    t += G[community[i]][community[j]].get('weight', 1.0)
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q


def specific_value_loss(output, penalty_label_value, penalty_weight=1.0):
    return penalty_weight * torch.sum(output == penalty_label_value)


def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


# Example usage
if __name__ == "__main__":
    # Example predicted logits and ground truth labels
    predicted_coarse_grained_groups = torch.randn(19, 20)  # [batch_size, vocab_size]
    cg_group_labels = torch.tensor([625, -1, -1, -1, -1, -1, -1, -1, 515, -1, -1, -1, -1, -1, -1, -1])  # Ground truth

    # Initialize the custom loss function
    loss_fn = CoarseGrainedLoss(ignore_index=-1)

    # Compute the loss
    loss = loss_fn(predicted_coarse_grained_groups, cg_group_labels)
    print(f"Loss: {loss.item()}")
