import os
import sys

sys.path.append(os.path.dirname(__file__))

from encoder import FineGrainedEncoderProperty, CoarseGrainedEncoder
import torch.nn as nn
from model_utils import Timer



class ModelCGMapping(nn.Module):
    """
    Full model combining FineGrainedEncoder, CoarseGrainedEncoder, and GraphDecoder.
    """

    def __init__(self, num_atom_features, fg_encoder_hidden_dim, fg_encoder_output_dim, fg_num_hidden_layers,
                 cg_encoder_hidden_dim1, cg_encoder_hidden_dim2, cg_encoder_output_dim, cg_encoder_model, init_method, edge_method,
                 clustering_method, leiden_or_louvain_setting_dict, clustering_max_process, clustering_with_batch, seed, sigma, dropout,
                 transformer_nhead, transformer_nlayer, transformer_ffn_dim):
        super(ModelCGMapping, self).__init__()

        self.fine_grained_encoder = FineGrainedEncoderProperty(num_atom_features=num_atom_features,
                                                               hidden_dim=fg_encoder_hidden_dim,
                                                               output_dim=fg_encoder_hidden_dim,
                                                               num_hidden_layers=fg_num_hidden_layers,
                                                               init_method=init_method,
                                                               sigma=sigma,
                                                               dropout=dropout,
                                                               edge_method=edge_method,
                                                               transformer_nhead=transformer_nhead,
                                                               transformer_nlayer=transformer_nlayer,
                                                               transformer_ffn_dim=transformer_ffn_dim)


        self.coarse_grained_encoder = CoarseGrainedEncoder(input_dim=fg_encoder_output_dim,
                                                           hidden_dim1=cg_encoder_hidden_dim1,
                                                           hidden_dim2=cg_encoder_hidden_dim2,
                                                           output_dim=cg_encoder_output_dim,
                                                           base_model=cg_encoder_model,
                                                           clustering_method=clustering_method)
        self.leiden_or_louvain_setting_dict = leiden_or_louvain_setting_dict
        self.clustering_max_process = clustering_max_process
        self.seed = seed
        self.clustering_method = clustering_method
        self.clustering_with_batch = clustering_with_batch

    def forward(self, data, eval_flag=False, cg_degree=None):
        """
        Forward pass through the full model.

        Parameters:
        - data (torch_geometric.data.Data): Input molecular graph data.

        """
        if cg_degree is not None:
            self.leiden_or_louvain_setting_dict['cg_degree(gamma)'] = cg_degree

        timer = Timer()
        fine_grained_features = self.fine_grained_encoder(data)
        first_elapsed_time = timer.record_time()
        # print(f"\nfg features encoder time：{first_elapsed_time} s")

        if self.clustering_with_batch:
            clusters_results = self.coarse_grained_encoder.cluster_fg_features_batch(fg_features=fine_grained_features,
                                                                                     data=data,
                                                                                     clustering_method=self.clustering_method,
                                                                                     leiden_or_louvain_setting_dict=self.leiden_or_louvain_setting_dict,
                                                                                     eval_flag=eval_flag,
                                                                                     seed=self.seed)


        else:
            if self.clustering_max_process == 1:
                clusters_results = self.coarse_grained_encoder.cluster_fg_features(fg_features=fine_grained_features,
                                                                                   data=data,
                                                                                   clustering_method=self.clustering_method,
                                                                                   leiden_or_louvain_setting_dict=self.leiden_or_louvain_setting_dict,
                                                                                   eval_flag=eval_flag, seed=self.seed)
            else:
                clusters_results = self.coarse_grained_encoder.cluster_fg_features_mpi(
                    max_processes=self.clustering_max_process,
                    fg_features=fine_grained_features, data=data,
                    clustering_method=self.clustering_method,
                    leiden_or_louvain_setting_dict=self.leiden_or_louvain_setting_dict,
                    eval_flag=eval_flag, seed=self.seed)

        # Step 2: Extract coarse-grained features using clustering

        second_elapsed_time = timer.record_time()
        print(f"\nfg features encoder time：{first_elapsed_time} s \n clustering time：{second_elapsed_time} s")

        self.modularity_list = self.coarse_grained_encoder.get_modularity_list().to(fine_grained_features.device)
        if eval_flag:
            self.conductance_list = self.coarse_grained_encoder.conductance_list

        return fine_grained_features, clusters_results

class ModelCGMapping_MARTINI(nn.Module):
    """
    Full model combining FineGrainedEncoder, CoarseGrainedEncoder, and GraphDecoder.
    """

    def __init__(self, num_atom_features, fg_encoder_hidden_dim, fg_encoder_output_dim, fg_num_hidden_layers,
                 cg_encoder_hidden_dim1, cg_encoder_hidden_dim2, cg_encoder_output_dim, cg_encoder_model, init_method, edge_method,
                 clustering_method, leiden_or_louvain_setting_dict, clustering_max_process, seed, sigma, dropout,
                 transformer_nhead, transformer_nlayer, transformer_ffn_dim):
        super(ModelCGMapping_MARTINI, self).__init__()

        self.fine_grained_encoder = FineGrainedEncoderProperty(num_atom_features=num_atom_features,
                                                               hidden_dim=fg_encoder_hidden_dim,
                                                               output_dim=fg_encoder_hidden_dim,
                                                               num_hidden_layers=fg_num_hidden_layers,
                                                               init_method=init_method,
                                                               sigma=sigma,
                                                               dropout=dropout,
                                                               edge_method=edge_method,
                                                               transformer_nhead=transformer_nhead,
                                                               transformer_nlayer=transformer_nlayer,
                                                               transformer_ffn_dim=transformer_ffn_dim)

        self.leiden_or_louvain_setting_dict = leiden_or_louvain_setting_dict
        self.clustering_max_process = clustering_max_process
        self.seed = seed
        self.clustering_method = clustering_method

        self.NUM_CG_TYPES = 18
        # output_dim = output_dim + num_atom_features
        self.cg_type_fc = nn.Sequential(nn.Linear(fg_encoder_hidden_dim+64, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.NUM_CG_TYPES))
        if init_method == 'xavier':
            for layer in self.cg_type_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
        elif init_method == 'kaiming':
            for layer in self.cg_type_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')

    def forward(self, data, cg_degree=None):
        """
        Forward pass through the full model.

        Parameters:
        - data (torch_geometric.data.Data): Input molecular graph data.

        """
        if cg_degree is not None:
            self.leiden_or_louvain_setting_dict['cg_degree(gamma)'] = cg_degree

        # timer = Timer()
        # Step 1: Extract fine-grained features using GNN
        fine_grained_features = self.fine_grained_encoder(data)

        node_cg_type_pred = self.cg_type_fc(fine_grained_features)

        return fine_grained_features, node_cg_type_pred
