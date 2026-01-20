import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, NNConv
from finch import FINCH
from sklearn.cluster import Birch, DBSCAN, OPTICS
from rdkit import Chem
from rdkit.Chem import AllChem
import collections
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import threading


from model_utils import separate_molecules_from_batch, aggregate, build_coarse_grained_edges, flatten_and_offset_edge_indexs, flatten_cg_node_features
from clustering import Leiden, Louvain, calculate_pairwise_weight_numpy, flatten_communities_to_labels, cal_modularity_of_weighed_graph




class EdgeNNWithTransformer(nn.Module):
    def __init__(self, num_edge_features, hidden_dim, dropout=0.1, num_heads=4, num_transformer_layers=2, init_method='xavier'):
        super(EdgeNNWithTransformer, self).__init__()

        # Initial Linear Layer
        self.edge_linear = nn.Linear(num_edge_features, hidden_dim)

        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # Feature dimension for Transformer
            nhead=num_heads,  # Number of attention heads
            dim_feedforward=hidden_dim * 2,  # Feedforward dimension
            dropout=dropout  # Dropout probability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_transformer_layers
        )

        # Final Linear Layer
        self.output_linear = nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        if init_method == 'xavier':
            nn.init.xavier_uniform_(self.output_linear.weight.data)
        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.output_linear.weight.data, nonlinearity='relu')

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_attr):
        # Initial Linear Transformation
        edge_features = self.edge_linear(edge_attr)

        # Prepare for Transformer (batch-first format)
        edge_features = edge_features.unsqueeze(1)  # (num_edges, 1, hidden_dim)

        # Transformer Encoding
        edge_features = self.transformer_encoder(edge_features)  # (num_edges, 1, hidden_dim)

        # Remove batch dimension
        edge_features = edge_features.squeeze(1)  # (num_edges, hidden_dim)

        # Final Linear Transformation
        edge_features = self.output_linear(edge_features)  # (num_edges, hidden_dim * hidden_dim)

        return edge_features



class FineGrainedEncoderProperty(nn.Module):
    def __init__(self, num_atom_features, num_hidden_layers, hidden_dim, output_dim, sigma, dropout=0.0,
                 transformer_nhead=4, transformer_nlayer=3, transformer_ffn_dim=64, init_method='kaiming',
                 edge_method='bonded+nonbonded'):
        super(FineGrainedEncoderProperty, self).__init__()

        self.edge_method = edge_method
        self.gau_sigma = sigma
        self.dropout = dropout
        num_edge_features = None
        if self.edge_method == 'bonded+nonbonded':
            num_edge_features = 17
        elif self.edge_method == 'bonded':
            num_edge_features = 13
        elif self.edge_method == 'nonbonded':
            num_edge_features = 4

        self.input_fc = nn.Embedding(119, hidden_dim)  # 118 is max atomic number

        hidden_dim = hidden_dim + num_atom_features
        # self.edge_nn = EdgeNNWithTransformer(
        #     num_edge_features=num_edge_features,
        #     hidden_dim=hidden_dim,
        #     dropout=self.dropout,
        #     num_heads=transformer_nhead,
        #     num_transformer_layers=transformer_nlayer
        # )

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_hidden_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.edge_nn = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim*hidden_dim)
        )

        # self.edge_nns = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(num_edge_features, hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout),
        #         nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        #     ) for _ in range(num_hidden_layers)
        # ])

        # self.conv = NNConv(hidden_dim, hidden_dim, self.edge_nn, aggr='add')
        self.convs = nn.ModuleList(
            [NNConv(hidden_dim, hidden_dim, self.edge_nn, aggr='add') for _ in range(num_hidden_layers)])
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        hidden_dim = hidden_dim + num_atom_features
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.num_hidden_layers = num_hidden_layers

        # Transformer encoder for atom features
        transformer_layer = nn.TransformerEncoderLayer(d_model=num_atom_features, nhead=transformer_nhead,
                                                       dim_feedforward=transformer_ffn_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_nlayer)

        if init_method == 'xavier':
            nn.init.xavier_uniform_(self.input_fc.weight.data)
            nn.init.xavier_uniform_(self.output_fc.weight.data)

            for layer in self.edge_nn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)

        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.input_fc.weight.data, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.output_fc.weight.data)

            for layer in self.edge_nn:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')

    def forward(self, data):

        x, edge_index, edge_attr_init = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        # Apply Transformer to atom_features
        atom_features = data.atom_features.unsqueeze(0)  # Add sequence dimension
        atom_features = self.transformer_encoder(atom_features).squeeze(0)

        # if torch.isnan(atom_features).any():
        #     print("NaN detected in model output!")
        #     atom_features = torch.nan_to_num(atom_features, nan=0.0)

        # if torch.isnan(edge_attr_init).any():  # tox21
        #     edge_attr_init = torch.where(torch.isnan(edge_attr_init), torch.zeros_like(edge_attr_init), edge_attr_init)
        # if torch.isnan(atom_features).any():
        #     atom_features = torch.where(torch.isnan(atom_features), torch.zeros_like(atom_features), atom_features)

        x_out = F.relu(self.input_fc(x)).squeeze(1)
        x_out = torch.cat([x_out, atom_features], dim=1)
        h = x_out.unsqueeze(0)
        edge_weights = self.gaussian_kernel_matrix(x_out, edge_index)

        edge_attr = edge_attr_init * edge_weights

        for i in range(self.num_hidden_layers):
            m = F.relu(self.convs[i](x_out, edge_index, edge_attr))
            m = F.dropout(m, p=self.dropout, training=self.training)
            x_out, h = self.gru(m.unsqueeze(0), h)

            h = h.squeeze(0)
            h = self.batch_norms[i](h)
            h = h.unsqueeze(0)

            x_out = x_out.squeeze(0)

            edge_weights = self.gaussian_kernel_matrix(x_out, edge_index)
            edge_attr = edge_attr_init * edge_weights

        x_out = torch.cat([x_out, data.atom_features], dim=1)
        x_out = F.normalize(x_out)
        x_out = self.output_fc(x_out)
        # x_out = global_mean_pool(x_out, batch)
        #
        # if self.task == 'classification':
        #     x_out = self.sigmoid(x_out)

        return x_out

    def gaussian_kernel_matrix(self, x, edge_index):
        sigma = self.gau_sigma
        src_nodes, tgt_nodes = edge_index

        diff = x[src_nodes] - x[tgt_nodes]
        if sigma is None:
            sigma = torch.std(diff)
        edge_weights = torch.exp(-torch.sum(diff ** 2, dim=1) / (2 * sigma ** 2))
        return edge_weights.unsqueeze(1)


class FineGrainedEncoder(nn.Module):
    def __init__(self, num_atom_features, num_hidden_layers, hidden_dim, output_dim, sigma, dropout=0.0,
                 transformer_nhead=4, transformer_nlayer=3, transformer_ffn_dim=64, init_method='kaiming',
                 task='regression', edge_method='bonded+nonbonded'):
        super(FineGrainedEncoder, self).__init__()

        self.task = task
        self.edge_method = edge_method
        self.gau_sigma = sigma
        self.dropout = dropout
        num_edge_features = None
        if self.edge_method == 'bonded+nonbonded':
            num_edge_features = 17
        elif self.edge_method == 'bonded':
            num_edge_features = 13
        elif self.edge_method == 'nonbonded':
            num_edge_features = 4

        self.input_fc = nn.Embedding(119, hidden_dim)  # 118 is max atomic number

        hidden_dim = hidden_dim + num_atom_features
        self.edge_nn = EdgeNNWithTransformer(
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            dropout=self.dropout,
            num_heads=transformer_nhead,
            num_transformer_layers=transformer_nlayer,
            init_method=init_method,
        )

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_hidden_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # self.edge_nn = nn.Sequential(
        #     nn.Linear(num_edge_features, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(hidden_dim, hidden_dim*hidden_dim)
        # )

        self.edge_nns = nn.ModuleList([self.edge_nn for _ in range(num_hidden_layers)])

        # self.conv = NNConv(hidden_dim, hidden_dim, self.edge_nn, aggr='add')
        self.convs = nn.ModuleList(
            [NNConv(hidden_dim, hidden_dim, self.edge_nns[i], aggr='add') for i in range(num_hidden_layers)])
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        hidden_dim = hidden_dim + num_atom_features
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # self.output_fc2 = nn.Linear(output_dim, target_dim)
        # self.sigmoid = nn.Sigmoid()

        self.num_hidden_layers = num_hidden_layers

        # Transformer encoder for atom features
        transformer_layer = nn.TransformerEncoderLayer(d_model=num_atom_features, nhead=transformer_nhead,
                                                       dim_feedforward=transformer_ffn_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_nlayer)

        if init_method == 'xavier':
            nn.init.xavier_uniform_(self.input_fc.weight.data)
            for layer in self.output_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)

            # nn.init.xavier_uniform_(self.output_fc2.weight.data)

            # for edge_nn in self.edge_nns:
            #     for layer in edge_nn:
            #         if isinstance(layer, nn.Linear):
            #             nn.init.xavier_uniform_(layer.weight.data)

        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.input_fc.weight.data, nonlinearity='relu')
            for layer in self.output_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')

            # nn.init.kaiming_uniform_(self.output_fc2.weight.data)

            # for edge_nn in self.edge_nns:
            #     for layer in edge_nn:
            #         if isinstance(layer, nn.Linear):
            #             nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')


    def forward(self, data, return_features=False):

        x, edge_index, edge_attr_init = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        # Apply Transformer to atom_features
        atom_features = data.atom_features.unsqueeze(0)  # Add sequence dimension
        atom_features = self.transformer_encoder(atom_features).squeeze(0)  # Remove sequence dimension

        # if torch.isnan(edge_attr_init).any():  # tox21
        #     edge_attr_init = torch.where(torch.isnan(edge_attr_init), torch.zeros_like(edge_attr_init), edge_attr_init)
        # if torch.isnan(atom_features).any():
        #     atom_features = torch.where(torch.isnan(atom_features), torch.zeros_like(atom_features), atom_features)

        x_out = F.relu(self.input_fc(x)).squeeze(1)
        x_out = torch.cat([x_out, atom_features], dim=1)
        h = x_out.unsqueeze(0)
        edge_weights = self.gaussian_kernel_matrix(x_out, edge_index)

        edge_attr = edge_attr_init * edge_weights

        for i in range(self.num_hidden_layers):
            m = F.relu(self.convs[i](x_out, edge_index, edge_attr))
            m = F.dropout(m, p=self.dropout, training=self.training)
            x_out, h = self.gru(m.unsqueeze(0), h)

            h = h.squeeze(0)
            h = self.batch_norms[i](h)
            h = h.unsqueeze(0)

            x_out = x_out.squeeze(0)

            edge_weights = self.gaussian_kernel_matrix(x_out, edge_index)
            edge_attr = edge_attr_init * edge_weights

        x_out = torch.cat([x_out, data.atom_features], dim=1)
        x_out = F.normalize(x_out)
        x_out = self.output_fc(x_out)
        # if return_features:
        #     return x_out
        #
        # x_out = global_mean_pool(x_out, batch)
        #
        # x_out = self.output_fc2(x_out)
        #
        # if self.task == 'classification':
        #     x_out = self.sigmoid(x_out)
        # fg_embed = F.normalize(x_out)

        return x_out

    def gaussian_kernel_matrix(self, x, edge_index):
        sigma = self.gau_sigma
        src_nodes, tgt_nodes = edge_index

        diff = x[src_nodes] - x[tgt_nodes]
        if sigma is None:
            sigma = torch.std(diff)
        edge_weights = torch.exp(-torch.sum(diff ** 2, dim=1) / (2 * sigma ** 2))
        return edge_weights.unsqueeze(1)

class FineGrainedEncoderMapping(nn.Module):
    def __init__(self, num_atom_features, num_hidden_layers, hidden_dim, output_dim, sigma, dropout=0.0, init_method='kaiming',
                 transformer_nhead=4, transformer_nlayer=3, transformer_ffn_dim=64, edge_method='bonded+nonbonded'):
        super(FineGrainedEncoderMapping, self).__init__()

        self.edge_method = edge_method
        self.gau_sigma = sigma
        self.dropout = dropout
        num_edge_features = None
        if self.edge_method == 'bonded+nonbonded':
            num_edge_features = 17
        elif self.edge_method == 'bonded':
            num_edge_features = 13
        elif self.edge_method == 'nonbonded':
            num_edge_features = 4

        self.input_fc = nn.Embedding(119, hidden_dim)  # 118 is max atomic number

        hidden_dim = hidden_dim + num_atom_features
        # self.edge_nn = EdgeNNWithTransformer(
        #     num_edge_features=num_edge_features,
        #     hidden_dim=hidden_dim,
        #     dropout=self.dropout,
        #     num_heads=transformer_nhead,
        #     num_transformer_layers=transformer_nlayer
        # )
        self.edge_nn = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim*hidden_dim)
        )

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_hidden_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.conv = NNConv(hidden_dim, hidden_dim, self.edge_nn, aggr='add')
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        hidden_dim = hidden_dim + num_atom_features
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.sigmoid = nn.Sigmoid()
        self.num_hidden_layers = num_hidden_layers

        # Transformer encoder for atom features
        transformer_layer = nn.TransformerEncoderLayer(d_model=num_atom_features, nhead=transformer_nhead,
                                                       dim_feedforward=transformer_ffn_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_nlayer)

        if init_method == 'xavier':
            nn.init.xavier_uniform_(self.input_fc.weight.data)
            for layer in self.output_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)

            for layer in self.edge_nn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)

        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.input_fc.weight.data, nonlinearity='relu')
            for layer in self.output_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')

            for layer in self.edge_nn:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')

    def forward(self, data):

        x, edge_index, edge_attr_init = data.x, data.edge_index, data.edge_attr
        # batch = data.batch

        # Apply Transformer to atom_features
        atom_features = data.atom_features.unsqueeze(0)  # Add sequence dimension
        atom_features = self.transformer_encoder(atom_features).squeeze(0)  # Remove sequence dimension

        x_out = F.relu(self.input_fc(x)).squeeze(1)
        x_out = torch.cat([x_out, atom_features], dim=1)
        h = x_out.unsqueeze(0)

        edge_weights = self.gaussian_kernel_matrix(x_out, edge_index)
        edge_attr = edge_attr_init * edge_weights

        for i in range(self.num_hidden_layers):
            m = F.relu(self.conv(x_out, edge_index, edge_attr))
            m = F.dropout(m, p=self.dropout, training=self.training)
            x_out, h = self.gru(m.unsqueeze(0), h)

            h = h.squeeze(0)
            h = self.batch_norms[i](h)
            h = h.unsqueeze(0)

            x_out = x_out.squeeze(0)

            edge_weights = self.gaussian_kernel_matrix(x_out, edge_index)
            edge_attr = edge_attr_init * edge_weights

        x_out = torch.cat([x_out, data.atom_features], dim=1)
        x_out = F.normalize(x_out)
        # x_out = global_mean_pool(x_out, batch)
        x_out = self.output_fc(x_out)
        # fg_embed = F.normalize(x_out)

        return x_out

    def gaussian_kernel_matrix(self, x, edge_index):
        sigma = self.gau_sigma
        src_nodes, tgt_nodes = edge_index
        diff = x[src_nodes] - x[tgt_nodes]
        if sigma is None:
            sigma = torch.std(diff)
        edge_weights = torch.exp(-torch.sum(diff ** 2, dim=1) / (2 * sigma ** 2))
        return edge_weights.unsqueeze(1)

class CoarseGrainedEncoder(nn.Module):
    """
    Clustering layer for extracting coarse-grained features.
    """

    def __init__(self, clustering_method, base_model=None, input_dim=64, hidden_dim1=256, hidden_dim2=128, output_dim=64):
        super(CoarseGrainedEncoder, self).__init__()

        if base_model is not None:
            # Activation function
            self.relu = nn.ReLU()

            if base_model == 'gcn':
                self.conv1 = GCNConv(input_dim, hidden_dim1)
                self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
                self.conv3 = GCNConv(hidden_dim2, output_dim)
            elif base_model == 'gat':
                self.conv1 = GATConv(input_dim, hidden_dim1)
                self.conv2 = GATConv(hidden_dim1, hidden_dim2)
                self.conv3 = GATConv(hidden_dim2, output_dim)
            elif base_model == 'gin':
                self.conv1 = GINConv(nn.Linear(input_dim, hidden_dim1))
                self.conv2 = GINConv(nn.Linear(hidden_dim1, hidden_dim2))
                self.conv3 = GINConv(nn.Linear(hidden_dim2, output_dim))
            elif base_model == 'sage':
                self.conv1 = SAGEConv(input_dim, hidden_dim1)
                self.conv2 = SAGEConv(hidden_dim1, hidden_dim2)
                self.conv3 = SAGEConv(hidden_dim2, output_dim)

        self.clustering_method = clustering_method

    def forward(self, fg_features, data, leiden_or_louvain_setting_dict:dict, eval_flag=False):
        """
        Forward pass for pooling and extracting coarse-grained features.

        Parameters:
        - fg_features (torch.Tensor): Fine-grained node features.
        - batch (torch.Tensor): Tensor indicating which molecule each node belongs to.

        Returns:
        - cg_features (torch.Tensor): Coarse-grained group feature matrix.
        """
        fg_edge_indexs = data.edge_index

        # Cluster and aggregate fine-grained features to obtain coarse-grained features
        cluster_labels, cg_features, cg_groups = self.cluster_fg_features_and_get_cg_features(fg_features=fg_features,  data=data, clustering_method=self.clustering_method,
                                                                                              leiden_or_louvain_setting_dict=leiden_or_louvain_setting_dict, eval_flag=eval_flag)
        cg_groups_flatten_and_offset = flatten_and_offset_edge_indexs(cg_groups)
        cg_features_flatten = flatten_cg_node_features(cg_features).to(fg_features.device)

        cg_edge_index = build_coarse_grained_edges(edge_index=fg_edge_indexs, groups=cg_groups_flatten_and_offset).to(
            fg_features.device)

        x, edge_index = cg_features_flatten, cg_edge_index
        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = x / (x.sum())
        x = (F.tanh(x)) ** 2
        x = F.normalize(x)

        return x, cg_groups_flatten_and_offset, cluster_labels

    def cluster_fg_features(self, fg_features, data, leiden_or_louvain_setting_dict:dict, clustering_method='leiden', eval_flag=False, seed=None, r_eps=1e-1):
        """
        Cluster the fine-grained features for each molecule using the specified clustering method.

        Parameters:
        - molecule_features (List[torch.Tensor]): List of node features for each molecule.
        - clustering_method (str): Clustering method to use. Options are 'finch', 'birch', 'dbscan', and 'optics'.
        - hierarchy (int): The level of clustering hierarchy to use (only relevant for some methods like FINCH).

        Returns:
        - coarse_grained_features (List[torch.Tensor]): Coarse-grained features for each molecule.
        - coarse_grained_groups (List[torch.Tensor]): Grouping information for each molecule.
        """



        batch = data.batch
        molecule_features = separate_molecules_from_batch(fine_grained_features=fg_features, batch=batch)

        cluster_labels = None

        group_labels = []

        self.modularity_list = []
        self.conductance_list = []

        for mol_idx, features in enumerate(molecule_features):
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)

            features_np = features.detach().cpu().numpy()
            if clustering_method == 'finch':
                # Perform FINCH clustering
                clusters, num_clusters, req_c = FINCH(data=features_np, distance='cosine', verbose=False)
                cluster_labels = clusters[:, 0]  # finch_hierarchy=0

            elif clustering_method == 'birch':
                # Perform BIRCH clustering
                birch_model = Birch(n_clusters=None)  # You can set n_clusters if desired
                cluster_labels = birch_model.fit_predict(features_np)

            elif clustering_method == 'dbscan':
                # Perform DBSCAN clustering
                dbscan_model = DBSCAN(eps=0.5, min_samples=2, metric='cosine')  # Adjust eps and min_samples as needed
                cluster_labels = dbscan_model.fit_predict(features_np)

            elif clustering_method == 'optics':
                # Perform OPTICS clustering
                optics_model = OPTICS(min_samples=2, metric='cosine')  # Adjust min_samples as needed
                cluster_labels = optics_model.fit_predict(features_np)

            elif clustering_method in ['louvain', 'leiden']:

                if leiden_or_louvain_setting_dict['gaussian_sigma'] is None:
                    diff = features_np[:, None, :] - features_np[None, :, :]
                    all_distances_squared = np.sum(diff ** 2, axis=-1)
                    mean_distance_squared = np.mean(all_distances_squared)
                    if mean_distance_squared == 0.0:
                        print('Setting sigma to 1.0 because mean_distance_squared is 0.0')
                        sigma = 1.0
                    else:
                        sigma = np.sqrt(mean_distance_squared)
                else:
                    sigma = leiden_or_louvain_setting_dict['gaussian_sigma']
                    assert sigma != 0.0, 'gaussian sigma should not be zero.'

                smiles = data.smiles[mol_idx]

                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                embed_params = AllChem.ETKDGv3()
                if seed is not None:
                    embed_params.randomSeed = seed
                AllChem.EmbedMolecule(mol, embed_params)
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                # AllChem.ComputeGasteigerCharges(mol)
                mol = AllChem.RemoveAllHs(mol)
                conf = mol.GetConformer()
                smiles = Chem.MolToSmiles(mol)

                num_node = mol.GetNumAtoms()
                if num_node == 1:
                    cluster_labels = [0]
                    modularity = 0
                    # conductance_list = [0]

                    group_labels.append(cluster_labels)
                    self.conductance_list.append(0)
                    continue
                    # return cluster_labels, modularity, conductance_list

                assert len(features_np) == num_node, f'{len(features_np)} != {num_node}'

                mol_graph = collections.defaultdict(dict)
                mol_nx_graph = nx.Graph()
                for atom in mol.GetAtoms():
                    mol_nx_graph.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
                if leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain'] == 'pairwise':
                    for i in range(num_node):
                        for j in range(i+1, num_node):
                            if AllChem.GetBondLength(conf, i, j) < r_eps:
                                continue
                            weight = calculate_pairwise_weight_numpy(feature1=features_np[i], feature2=features_np[j],
                                                                     method=leiden_or_louvain_setting_dict['edge_weighted_function'],
                                                                     sigma=sigma)
                            if weight <= 0.0 or np.isnan(weight):
                                # print(f'weight is {weight}')
                                weight = 1e-6
                            # if weight > 0.0:
                            mol_graph[i][j] = weight
                            mol_graph[j][i] = weight
                            mol_nx_graph.add_edge(i, j, weight=weight)

                elif type(leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain']) == dict:
                    rcut = leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain']['rcut']

                    mol = Chem.AddHs(mol)
                    embed_params = AllChem.ETKDGv3()
                    if seed is not None:
                        embed_params.randomSeed = seed
                    AllChem.EmbedMolecule(mol, embed_params)
                    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                    mol = AllChem.RemoveAllHs(mol)
                    conf = mol.GetConformer()

                    for i in range(mol.GetNumAtoms()):
                        for j in range(i + 1, mol.GetNumAtoms()):
                            dist = AllChem.GetBondLength(conf, i, j)  # Angstrom
                            if r_eps < dist < rcut:
                                weight = (1 - dist / rcut) * calculate_pairwise_weight_numpy(feature1=features_np[i],
                                                                                             feature2=features_np[j],
                                                                                             method=leiden_or_louvain_setting_dict['edge_weighted_function'],
                                                                                             sigma=sigma)  # (1 - dist / rcut) *
                                # weight = (1 - dist/rcut) * calculate_pairwise_weight_numpy(feature1=features_np[i],
                                #                                          feature2=features_np[j],
                                #                                          method=leiden_or_louvain_setting_dict[
                                #                                              'edge_weighted_function'],
                                #                                          sigma=sigma)
                                if weight <= 0.0 or np.isnan(weight):
                                    # print(f'weight is {weight}')
                                    weight = 1e-6
                                # if weight > 0.0:
                                mol_graph[i][j] = weight
                                mol_graph[j][i] = weight
                                mol_nx_graph.add_edge(i, j, weight=weight)



                elif leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain'] == 'bond':
                    for bond in mol.GetBonds():
                        bid = bond.GetBeginAtomIdx()
                        eid = bond.GetEndAtomIdx()
                        # if AllChem.GetBondLength(conf, bid, eid) < r_eps:
                        #     continue
                        weight = calculate_pairwise_weight_numpy(feature1=features_np[bid], feature2=features_np[eid],
                                                                 method=leiden_or_louvain_setting_dict['edge_weighted_function'],
                                                                 sigma=sigma)
                        if weight <= 0.0 or np.isnan(weight):
                            # print(f'weight is {weight}')
                            weight = 1e-6

                        mol_graph[bid][eid] = weight
                        mol_graph[eid][bid] = weight
                        mol_nx_graph.add_edge(bid, eid, weight=weight)

                else:
                    raise Exception('Edge type for Leiden or Louvain clustering is not supported')

                if clustering_method == 'louvain':
                    model = Louvain(graph=mol_graph, gamma=leiden_or_louvain_setting_dict['cg_degree(gamma)'])
                    communities = model.execute(max_iter=leiden_or_louvain_setting_dict['max_iter'], recoder_hierarchy=False)
                    # modularity = cal_modularity_of_weighed_graph(partition=communities, G=mol_nx_graph)
                    # self.modularity_list.append(modularity)

                    if len(data.edge_index[0].unique()) != len(data.batch):
                        a = set(data.edge_index[0].unique().numpy())
                        b = set(range(len(data.batch)))
                        c = b - a
                        for i in list(c):
                            communities.append([i])

                    cluster_labels = flatten_communities_to_labels(communities=communities)

                elif clustering_method == 'leiden':
                    model = Leiden(graph=mol_graph, nx_graph=mol_nx_graph,
                                   gamma=leiden_or_louvain_setting_dict['cg_degree(gamma)'],
                                   gamma2=leiden_or_louvain_setting_dict['gamma2'],
                                   theta=leiden_or_louvain_setting_dict['theta'])
                    communities = model.execute(max_iter=leiden_or_louvain_setting_dict['max_iter'], recoder_hierarchy=False)
                    # modularity = cal_modularity_of_weighed_graph(partition=communities, G=mol_nx_graph)
                    # self.modularity_list.append(modularity)

                    if len(data.edge_index[0].unique()) != len(data.batch):
                        a = set(data.edge_index[0].unique().numpy())
                        b = set(range(len(data.batch)))
                        c = b - a
                        for i in list(c):
                            communities.append([i])

                    cluster_labels = flatten_communities_to_labels(communities=communities)

                if eval_flag:
                    if len(communities) > 1:
                        for community in communities:
                            self.conductance_list.append(nx.conductance(mol_nx_graph, community, weight='weight'))

            else:
                raise ValueError(f"Unsupported clustering method: {clustering_method}")

            num_atom_of_molecule = features_np.shape[0]
            # if len(cluster_labels) != num_atom_of_molecule:
            assert len(cluster_labels) == num_atom_of_molecule, (
                f"Mismatch between number of cluster labels ({len(cluster_labels)}) and feature entries ({features_np.shape[0]})."
            )

            group_labels.append(cluster_labels)

        return group_labels

    @staticmethod
    def cluster_single_molecule_fg_features(features_np, data_smiles, leiden_or_louvain_setting_dict, clustering_method,
                         eval_flag, seed=None, r_eps=1e-1):

        # features_np = features.detach().cpu().numpy()
        cluster_labels = None
        modularity = None
        conductance = None
        conductance_list = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if clustering_method == 'finch':
            # Perform FINCH clustering
            clusters, num_clusters, req_c = FINCH(data=features_np, distance='cosine', verbose=False)
            cluster_labels = clusters[:, 0]  # finch_hierarchy=0

        elif clustering_method == 'birch':
            # Perform BIRCH clustering
            birch_model = Birch(n_clusters=None)
            cluster_labels = birch_model.fit_predict(features_np)

        elif clustering_method == 'dbscan':
            # Perform DBSCAN clustering
            dbscan_model = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
            cluster_labels = dbscan_model.fit_predict(features_np)

        elif clustering_method == 'optics':
            # Perform OPTICS clustering
            optics_model = OPTICS(min_samples=2, metric='cosine')
            cluster_labels = optics_model.fit_predict(features_np)

        elif clustering_method in ['louvain', 'leiden']:

            if leiden_or_louvain_setting_dict['gaussian_sigma'] is None:
                diff = features_np[:, None, :] - features_np[None, :, :]
                all_distances_squared = np.sum(diff ** 2, axis=-1)
                mean_distance_squared = np.mean(all_distances_squared)
                if mean_distance_squared == 0.0:
                    # print('Setting sigma to 1.0 because mean_distance_squared is 0.0')
                    sigma = 1.0
                else:
                    sigma = np.sqrt(mean_distance_squared)
            else:
                sigma = leiden_or_louvain_setting_dict['gaussian_sigma']

            mol = Chem.MolFromSmiles(data_smiles)
            mol = Chem.AddHs(mol)
            embed_params = AllChem.ETKDGv3()
            if seed is not None:
                embed_params.randomSeed = seed
            AllChem.EmbedMolecule(mol, embed_params)
            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
            # AllChem.ComputeGasteigerCharges(mol)
            mol = AllChem.RemoveAllHs(mol)
            conf = mol.GetConformer()
            smiles = Chem.MolToSmiles(mol)

            num_node = mol.GetNumAtoms()
            if num_node == 1:
                cluster_labels = [0]
                modularity = 0
                conductance_list = [0]
                return cluster_labels, modularity, conductance_list

            assert len(features_np) == num_node, f'{len(features_np)} != {num_node}'

            mol_graph = collections.defaultdict(dict)
            mol_nx_graph = nx.Graph()
            for atom in mol.GetAtoms():
                mol_nx_graph.add_node(atom.GetIdx(), symbol=atom.GetSymbol())

            if leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain'] == 'pairwise':
                for i in range(num_node):
                    for j in range(i + 1, num_node):
                        if AllChem.GetBondLength(conf, i, j) < r_eps:
                            continue
                        weight = calculate_pairwise_weight_numpy(feature1=features_np[i], feature2=features_np[j],
                                                                 method=leiden_or_louvain_setting_dict[
                                                                     'edge_weighted_function'],
                                                                 sigma=sigma)
                        if weight <= 0.0 or np.isnan(weight):
                            # print(f'weight is {weight}')
                            weight = 1e-6
                        # if weight > 0.0:
                        mol_graph[i][j] = weight
                        mol_graph[j][i] = weight
                        mol_nx_graph.add_edge(i, j, weight=weight)

            elif type(leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain']) == dict:
                rcut = leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain']['rcut']

                for i in range(mol.GetNumAtoms()):
                    for j in range(i + 1, mol.GetNumAtoms()):
                        dist = AllChem.GetBondLength(conf, i, j)  # (Angstrom)
                        if r_eps < dist < rcut:
                            weight = (1 - dist / rcut) * calculate_pairwise_weight_numpy(feature1=features_np[i],
                                                                                         feature2=features_np[j],
                                                                                         method=leiden_or_louvain_setting_dict['edge_weighted_function'],
                                                                                         sigma=sigma)  # (1 - dist / rcut) *
                            if weight <= 0.0 or np.isnan(weight):
                                # print(f'weight is {weight}')
                                weight = 1e-6
                            # if weight > 0.0:
                            mol_graph[i][j] = weight
                            mol_graph[j][i] = weight
                            mol_nx_graph.add_edge(i, j, weight=weight)



            elif leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain'] == 'bond':
                for bond in mol.GetBonds():
                    bid = bond.GetBeginAtomIdx()
                    eid = bond.GetEndAtomIdx()
                    # if AllChem.GetBondLength(conf, bid, eid) < r_eps:
                    #     continue
                    weight = calculate_pairwise_weight_numpy(feature1=features_np[bid], feature2=features_np[eid],
                                                             method=leiden_or_louvain_setting_dict[
                                                                 'edge_weighted_function'],
                                                             sigma=sigma)
                    if weight <= 0.0 or np.isnan(weight):
                        print(f'weight is {weight}')
                        weight = 1e-6

                    mol_graph[bid][eid] = weight
                    mol_graph[eid][bid] = weight
                    mol_nx_graph.add_edge(bid, eid, weight=weight)


            else:
                raise Exception('Edge type for Leiden or Louvain clustering is not supported')

            if clustering_method == 'louvain':
                model = Louvain(graph=mol_graph, gamma=leiden_or_louvain_setting_dict['cg_degree(gamma)'])
                communities = model.execute(max_iter=leiden_or_louvain_setting_dict['max_iter'],
                                            recoder_hierarchy=False)
                modularity = cal_modularity_of_weighed_graph(partition=communities, G=mol_nx_graph)

                # if len(data.edge_index[0].unique()) != len(data.batch):
                #     a = set(data.edge_index[0].unique().numpy())
                #     b = set(range(len(data.batch)))
                #     c = b - a
                #     for i in list(c):
                #         communities.append([i])

                cluster_labels = flatten_communities_to_labels(communities=communities)

            elif clustering_method == 'leiden':
                model = Leiden(graph=mol_graph, nx_graph=mol_nx_graph,
                               gamma=leiden_or_louvain_setting_dict['cg_degree(gamma)'],
                               gamma2=leiden_or_louvain_setting_dict['gamma2'],
                               theta=leiden_or_louvain_setting_dict['theta'])
                communities = model.execute(max_iter=leiden_or_louvain_setting_dict['max_iter'],
                                            recoder_hierarchy=False)
                modularity = cal_modularity_of_weighed_graph(partition=communities, G=mol_nx_graph)
                # if len(data.edge_index[0].unique()) != len(data.batch):
                #     a = set(data.edge_index[0].unique().numpy())
                #     b = set(range(len(data.batch)))
                #     c = b - a
                #     for i in list(c):
                #         communities.append([i])

                cluster_labels = flatten_communities_to_labels(communities=communities)

            if eval_flag:
                if len(communities) > 1:
                    for community in communities:
                        conductance = nx.conductance(mol_nx_graph, community, weight='weight')
                        conductance_list.append(conductance)

        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")

        num_atom_of_molecule = features_np.shape[0]
        assert len(cluster_labels) == num_atom_of_molecule, (
            f"{data_smiles}: Mismatch between number of cluster labels ({len(cluster_labels)}) and feature entries ({features_np.shape[0]})."
        )

        return cluster_labels, modularity, conductance_list

    def cluster_fg_features_mpi(self, max_processes, fg_features, data, leiden_or_louvain_setting_dict: dict, clustering_method='leiden', eval_flag=False, seed=None):

        batch = data.batch
        molecule_features = separate_molecules_from_batch(fine_grained_features=fg_features, batch=batch)

        group_labels = []

        self.modularity_list = []
        self.conductance_list = []

        lock = threading.Lock()

        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = {}

            for mol_idx, features in enumerate(molecule_features):
                data_smiles = data.smiles[mol_idx]
                features_np = features.detach().cpu().numpy()
                with lock:
                    future = executor.submit(
                        self.cluster_single_molecule_fg_features,
                        features_np, data_smiles, leiden_or_louvain_setting_dict,
                        clustering_method, eval_flag, seed
                    )
                    futures[future] = mol_idx

            wait(futures)
            # for future in as_completed(futures):
            #     pass
            # sorted_futures = dict(sorted(futures.items(), key=lambda x: x[1]))
            with lock:
                for future, mol_id in futures.items():
                    group_labels.append(future.result()[0])
                    # self.modularity_list.append(future.result()[1])
                    if eval_flag and future.result()[2]:
                        self.conductance_list.extend(future.result()[2])

        return group_labels


    def cluster_fg_features_batch(self, fg_features, data, leiden_or_louvain_setting_dict:dict, clustering_method='leiden', eval_flag=False, seed=None, r_eps=1e-1):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        features_np = fg_features.detach().cpu().numpy()

        cluster_labels = None

        self.modularity_list = []
        self.conductance_list = []

        if clustering_method == 'finch':
            # Perform FINCH clustering
            clusters, num_clusters, req_c = FINCH(data=features_np, distance='cosine', verbose=False)
            cluster_labels = clusters[:, 0]  # finch_hierarchy=0

        elif clustering_method == 'birch':
            # Perform BIRCH clustering
            birch_model = Birch(n_clusters=None)  # You can set n_clusters if desired
            cluster_labels = birch_model.fit_predict(features_np)

        elif clustering_method == 'dbscan':
            # Perform DBSCAN clustering
            dbscan_model = DBSCAN(eps=0.5, min_samples=2, metric='cosine')  # Adjust eps and min_samples as needed
            cluster_labels = dbscan_model.fit_predict(features_np)

        elif clustering_method == 'optics':
            # Perform OPTICS clustering
            optics_model = OPTICS(min_samples=2, metric='cosine')  # Adjust min_samples as needed
            cluster_labels = optics_model.fit_predict(features_np)

        elif clustering_method in ['louvain', 'leiden']:

            if leiden_or_louvain_setting_dict['gaussian_sigma'] is None:
                diff = features_np[:, None, :] - features_np[None, :, :]
                all_distances_squared = np.sum(diff ** 2, axis=-1)
                mean_distance_squared = np.mean(all_distances_squared)
                if mean_distance_squared == 0.0:
                    print('Setting sigma to 1.0 because mean_distance_squared is 0.0')
                    sigma = 1.0
                else:
                    sigma = np.sqrt(mean_distance_squared)
            else:
                sigma = leiden_or_louvain_setting_dict['gaussian_sigma']

            mol_graph = collections.defaultdict(dict)
            mol_nx_graph = nx.Graph()

            # if leiden_or_louvain_setting_dict['edge_for_leiden_or_louvain'] == 'pairwise':
            for edge_item in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()):
                i = edge_item[0]
                j = edge_item[1]
                weight = calculate_pairwise_weight_numpy(feature1=features_np[i], feature2=features_np[j],
                                                         method=leiden_or_louvain_setting_dict[
                                                             'edge_weighted_function'],
                                                         sigma=sigma)
                if weight > 0.0:
                    mol_graph[i][j] = weight
                    # mol_graph[j][i] = weight
                    mol_nx_graph.add_edge(i, j, weight=weight)

                elif weight <= 0.0 or np.isnan(weight):
                    print(f'weight is {weight}')
                    weight = 1e-6
                    mol_graph[i][j] = weight
                    mol_nx_graph.add_edge(i, j, weight=weight)

            if clustering_method == 'louvain':
                model = Louvain(graph=mol_graph, gamma=leiden_or_louvain_setting_dict['cg_degree(gamma)'])
                communities = model.execute(max_iter=leiden_or_louvain_setting_dict['max_iter'], recoder_hierarchy=False)
                modularity = cal_modularity_of_weighed_graph(partition=communities, G=mol_nx_graph)
                self.modularity_list.append(modularity)

                if len(data.edge_index[0].unique()) != len(data.batch):
                    a = set(data.edge_index[0].unique().numpy())
                    b = set(range(len(data.batch)))
                    c = b - a
                    for i in list(c):
                        communities.append([i])

                cluster_labels = flatten_communities_to_labels(communities=communities)

            elif clustering_method == 'leiden':
                model = Leiden(graph=mol_graph, nx_graph=mol_nx_graph,
                               gamma=leiden_or_louvain_setting_dict['cg_degree(gamma)'],
                               gamma2=leiden_or_louvain_setting_dict['gamma2'],
                               theta=leiden_or_louvain_setting_dict['theta'])
                communities = model.execute(max_iter=leiden_or_louvain_setting_dict['max_iter'], recoder_hierarchy=False)
                modularity = cal_modularity_of_weighed_graph(partition=communities, G=mol_nx_graph)
                self.modularity_list.append(modularity)

                if len(data.edge_index[0].unique()) != len(data.batch):
                    a = set(data.edge_index[0].unique().numpy())
                    b = set(range(len(data.batch)))
                    c = b - a
                    for i in list(c):
                        communities.append([i])

                cluster_labels = flatten_communities_to_labels(communities=communities)

            if eval_flag:
                if len(communities) > 1:
                    for community in communities:
                        self.conductance_list.append(nx.conductance(mol_nx_graph, community, weight='weight'))

        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")

        num_atom_of_molecule = features_np.shape[0]
        assert len(cluster_labels) == num_atom_of_molecule, (
            f"Mismatch between number of cluster labels ({len(cluster_labels)}) and feature entries ({features_np.shape[0]})."
        )

        return cluster_labels



    def get_modularity_list(self):
        return torch.tensor(self.modularity_list)