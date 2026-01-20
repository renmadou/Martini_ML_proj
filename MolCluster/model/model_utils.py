import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
import collections
import time

logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)

class Timer:
    def __init__(self):
        """
        Initialize the Timer class and record the start time.

        This constructor initializes the Timer object by setting the start_time and prev_time attributes.
        The start_time represents the moment when the timer is first initialized or reset.
        The prev_time is initially set to the same value as the start_time and will be updated
        each time a time interval is recorded.
        """
        self.start_time = None
        self.prev_time = None
        self.reset()

    def reset(self):
        """
        Reset the timer by setting the start time and the previous time to the current time.

        This method is used to restart the timer. It sets both the start_time and prev_time
        attributes to the current time obtained from the time.time() function. This allows for
        a fresh start in measuring time intervals.
        """
        self.start_time = time.time()
        self.prev_time = self.start_time

    def record_time(self):
        """
        Record the time interval between the current time and the previous record time,
        and update the previous record time to the current time.

        Returns:
            float: The time interval (in seconds) between this record and the previous one.

        This method calculates the elapsed time since the last record by subtracting the
        prev_time from the current_time obtained from the time.time() function. Then it
        updates the prev_time to the current_time for the next record. The returned value
        is the time interval that has passed between the two consecutive records.
        """
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        self.prev_time = current_time
        return elapsed_time

    def total_elapsed_time(self):
        """
        Calculate the total elapsed time from the start time to the current time.

        Returns:
            float: The total elapsed time (in seconds) from the start time to the current time.

        This method determines the total time that has passed since the timer was initially
        started or last reset. It subtracts the start_time from the current_time obtained
        from the time.time() function to get the overall elapsed time.
        """
        current_time = time.time()
        return current_time - self.start_time


def separate_molecules_from_batch(fine_grained_features, batch):
    """
    Separate node features by molecule based on batch information.

    Parameters:
    - fine_grained_features (torch.Tensor): Fine-grained node feature matrix, shape (total_nodes, feature_dim)
    - batch (torch.Tensor): Tensor indicating which molecule each node belongs to, shape (total_nodes,)

    Returns:
    - List[torch.Tensor]: A list of node feature tensors for each molecule
    """
    # Get unique batch indices (molecule identifiers)
    unique_batches = batch.unique()

    # Create a list of node features for each molecule based on batch indices
    molecule_features = [fine_grained_features[batch == i] for i in unique_batches]

    return molecule_features


def aggregate(features, method='mean', axis=0):
    """
    Aggregate node features based on the specified method.

    Parameters:
    - features (List[torch.Tensor]): List of tensors representing node features.
    - method (str): Aggregation method, either 'mean' (default) or other custom methods.

    Returns:
    - torch.Tensor: Aggregated feature tensor.
    """
    features = torch.stack(features)  # Stack the list of tensors into a single tensor

    if method == 'mean':
        return torch.mean(features, dim=axis)  # Compute mean along the 0th dimension

def aggregate_np(features, method='mean'):
    features = np.array(features)
    if method == 'mean':
        return np.mean(features, axis=1)


def build_coarse_grained_edges(edge_index, groups):
    """
    Convert original atomic edge_index to coarse-grained edge_index.

    Parameters:
    - edge_index (torch.Tensor): Original full-atom edge index, shape [2, num_edges].
    - groups (List[List[int]]): List of lists, where each inner list represents the atomic ids belonging to the same coarse-grained group.

    Returns:
    - cg_edge_index (torch.Tensor): Coarse-grained edge index, shape [2, num_coarse_edges].
    """
    # Create a mapping from atomic ids to coarse-grained group ids
    atom_to_cg = {}
    for cg_id, group in enumerate(groups):
        for atom_id in group:
            atom_to_cg[int(atom_id)] = cg_id

    # Initialize set to store unique coarse-grained edges
    cg_edges_set = set()

    # Iterate over original edge_index and map to coarse-grained edges
    for i in range(edge_index.shape[1]):
        atom_u, atom_v = edge_index[0, i], edge_index[1, i]

        # Get the coarse-grained ids for the atoms
        cg_u = atom_to_cg[atom_u.item()]
        cg_v = atom_to_cg[atom_v.item()]

        # Add the edge between the coarse-grained groups (ignoring self-loops)
        if cg_u != cg_v:
            cg_edges_set.add((cg_u, cg_v))
            cg_edges_set.add((cg_v, cg_u))  # Assuming undirected graph

    # Convert the set of coarse-grained edges to a tensor
    if cg_edges_set:
        cg_edges = torch.tensor(list(cg_edges_set), dtype=torch.long).t().contiguous()
    else:
        # If there are no edges, return an empty tensor
        cg_edges = torch.empty((2, 0), dtype=torch.long)

    return cg_edges


def flatten_and_offset_edge_indexs(groups):
    """
    Flatten a 3D list (or list of lists of tensors) into a 2D list (list of tensors).
    Adjust the offsets of each tensor based on the cumulative number of elements
    encountered in the previous groups.

    Parameters:
    - groups (List[List[torch.Tensor]]): 3D list where each sublist contains tensors
      representing groupings of atomic indices.

    Returns:
    - List[torch.Tensor]: 2D list where each tensor has been adjusted with the offset.
    """
    flat_groups = []  # To store the flattened 2D list of tensors
    offset = 0  # Initialize the offset as 0

    for group in groups:
        # Flatten each sublist into a 2D list and adjust offsets
        for sub_group in group:
            adjusted_tensor = sub_group + offset  # Apply the current offset to each tensor
            flat_groups.append(adjusted_tensor)  # Add to the flattened list
            # Update offset by the number of elements in the current group
        offset = sum(len(i) for i in flat_groups)

    return flat_groups


def flatten_cg_node_features(cg_features):
        flat_groups = []  # To store the flattened 2D list of tensors
        for group in cg_features:
            # Flatten each sublist into a 2D list and adjust offsets
            for sub_group in group:
                adjusted_tensor = sub_group  # Apply the current offset to each tensor
                flat_groups.append(adjusted_tensor)  # Add to the flattened list

        return torch.stack(flat_groups)


def check_functional_group(smiles, functional_group_smarts):
    """
    Check if a molecule contains a specific functional group and return the atom IDs involved.

    Parameters:
    - smiles (str): The SMILES string representing the molecule.
    - functional_group_smarts (str): The SMARTS pattern representing the functional group.

    Returns:
    - List[List[int]]: A list of lists, where each inner list contains atom IDs for a match.
      Returns an empty list if no matches are found.
    """
    try:
        # Convert the SMILES string to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES format")

        # Convert the SMARTS pattern to an RDKit molecule object
        functional_group = Chem.MolFromSmarts(functional_group_smarts)
        if functional_group is None:
            raise ValueError("Invalid SMARTS format")

        # Use RDKit's GetSubstructMatches method to find matches for the functional group
        matches = mol.GetSubstructMatches(functional_group)

        # Return a list of atom IDs for each match
        return [list(match) for match in matches]

    except Exception as e:
        # Handle errors and print the exception message
        print(f"Error: {e}")
        return []


def group_to_smiles(group, data, verbose=True):
    """
    Converts a group of atom indices into a SMILES string using the molecular structure from 'data',
    preserving the original bond types from the molecular graph.

    Parameters:
    - group (Tensor): Tensor of atom indices in the group.
    - data (torch_geometric.data.Data): PyTorch Geometric data object containing atom, bond information,
      and bond types in `edge_attr`.

    Returns:
    - smiles (str): The SMILES string corresponding to the group of atoms.
    """
    # Step 1: Get atom types for the atoms in the group
    atom_types = data.atom_types  # Assumes `data.atom_types` contains atom types for each index
    edge_index = data.edge_index  # Get the full molecular graph's edges
    edge_attr = data.edge_attr  # Get the bond types (edge attributes)

    # Step 2: Create a mask to extract subgraph
    device = edge_index.device
    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    mask[group] = True  # Set mask to True for the atoms in the group

    # Step 3: Extract subgraph edges corresponding to the group
    subgraph_mask = mask[edge_index[0]] & mask[edge_index[1]]
    subgraph_edges = edge_index[:, subgraph_mask]
    subgraph_edge_attr = edge_attr[subgraph_mask]  # Extract bond types for the subgraph (non-direction edge)

    # Step 4: Create RDKit molecule
    mol = Chem.RWMol()

    # Step 5: Add atoms to the RDKit molecule
    atom_map = {}  # Maps PyTorch Geometric atom indices to RDKit atom indices
    for i, atom_idx in enumerate(group):
        atom_type = atom_types[atom_idx].item()  # Get the atomic number of the atom
        atom = Chem.Atom(atom_type)  # Create an RDKit atom object
        mol_idx = mol.AddAtom(atom)  # Add atom to RDKit molecule
        atom_map[atom_idx.item()] = mol_idx  # Map atom index to RDKit atom index

    # Track whether any atoms are unbonded
    unbonded_atoms = set(group.tolist())

    # sanitizemol label
    sanitizemol_label = True

    # Step 6: Add bonds from the subgraph
    for i, edge in enumerate(subgraph_edges.t()):
        atom_u, atom_v = edge
        if mol.GetBondBetweenAtoms(atom_map[atom_u.item()], atom_map[atom_v.item()]):
            continue

        # If a bond is added, remove these atoms from the unbonded_atoms set
        unbonded_atoms.discard(atom_u.item())
        unbonded_atoms.discard(atom_v.item())

        bond_type_value = subgraph_edge_attr[i].item()  # Get the bond type from edge_attr

        # Convert bond type value to RDKit bond type
        if bond_type_value == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_type_value == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_type_value == 3:
            bond_type = Chem.BondType.TRIPLE
        elif bond_type_value == 4:
            bond_type = Chem.BondType.AROMATIC
        else:
            raise ValueError(f"Unsupported bond type: {bond_type_value}")

        # Add the bond with the correct bond type
        mol.AddBond(atom_map[atom_u.item()], atom_map[atom_v.item()], bond_type)

    # Step 7: Sanitize the molecule to ensure valid bonding
        # Try to sanitize the molecule and generate the SMILES
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomKekulizeException, ValueError) as e:
        # If an exception occurs, skip this molecule and return empty values
        if verbose:
            print(f'{Chem.MolToSmiles(mol)} non-ring atom 1 marked aromatic')
        sanitizemol_label = False

    smiles = Chem.MolToSmiles(mol)
    # has_unbonded_atom = len(group) != len(atom_map)
    has_unbonded_or_isolated_atom = (len(unbonded_atoms) != 0) or ('.' in smiles)
    return smiles, has_unbonded_or_isolated_atom, sanitizemol_label


def get_group_labels(groups, smiles_vocab, data, verbose=True):
    """
    Get the coarse-grained group labels based on the SMILES vocabulary.
    If a group's SMILES is not in the vocabulary and has no unbonded atoms,
    the SMILES is added to the vocabulary (dynamic vocabulary method).

    Parameters:
    - groups (list of Tensors): A list where each element is a tensor containing atom indices for each group.
    - smiles_vocab (dict): A dictionary mapping SMILES strings to label indices.
    - data (torch_geometric.data.Data): PyTorch Geometric data object containing atom and bond information.

    Returns:
    - group_labels (list of int): A list of label indices for each group.
    """

    print('Get the coarse-grained group labels based on the SMILES vocabulary...\n')

    group_labels = []
    # smiles_vocab_extended = smiles_vocab.copy()
    smiles_vocab_extended = []

    for group in groups:

        if verbose:
            print(f'\nSMILES: {data.smiles[data.batch[group.max()]]}')
        # Convert the group of atom indices to a SMILES string
        smiles, has_unbonded_atom_or_isolated_atom, sanitizemol_label = group_to_smiles(group, data, verbose)

        # Look up the SMILES string in the vocabulary and get the label
        label = len(smiles_vocab)  # add an additional label (last vocab index + 1 )  to represent the unkonwn cg group.
        if sanitizemol_label:

            if smiles in smiles_vocab:
                label = smiles_vocab[smiles]
            else:
                # If not found and there are no unbonded atoms, add to the vocabulary
                if (not has_unbonded_atom_or_isolated_atom) and (smiles not in smiles_vocab_extended) and sanitizemol_label:

                    if verbose:
                        print(f"Adding new SMILES to vocab: {smiles}")
                    # smiles_vocab_extended[smiles] = new_label_index  # Assign a new label
                    # label = new_label_index
                    # new_label_index += 1
                    smiles_vocab_extended.append(smiles)
                else:
                    # If there are unbonded atoms and the SMILES is not found, raise an error
                    # raise ValueError(f"Group has unbonded atoms and SMILES '{smiles}' not found in vocabulary!")
                    if verbose:
                        print(f"Group has unbonded atoms ({has_unbonded_atom_or_isolated_atom}) or SMILES '{smiles}' not found in vocabulary!")

        else:
            if verbose:
                print(f'unsanitize mol form: {smiles}')

        group_labels.append(label)

    group_labels = torch.tensor(group_labels)

    return group_labels, smiles_vocab_extended


BOND_TYPE_DICT = {Chem.rdchem.BondType.SINGLE: 1,
                  Chem.rdchem.BondType.DOUBLE: 2,
                  Chem.rdchem.BondType.TRIPLE: 3,
                  Chem.rdchem.BondType.AROMATIC: 1.5,
                  Chem.rdchem.BondType.UNSPECIFIED: 0}

def get_smiles_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = collections.defaultdict(dict)

    for bond in mol.GetBonds():
        bid = bond.GetBeginAtomIdx()
        eid = bond.GetEndAtomIdx()
        weight = BOND_TYPE_DICT[bond.GetBondType()]
        G[bid][eid] = weight
        G[eid][bid] = weight

    return G




def main():
    smiles = "CCCc1ccccc1CCCCc2ccccc2N"  # Benzene
    functional_group_smarts = "*1:*:*:*:*:*1"  # SMARTS for benzene ring

    result = check_functional_group(smiles, functional_group_smarts)
    print("Matched functional group atom IDs:", result)

    # groups = [[torch.tensor([0, 1, 2])], [torch.tensor([0, 2]), torch.tensor([1, 11, 16])]]
    # flat_result = flatten_and_offset_edge_indexs(groups)
    #
    # print(flat_result)
    #
    # # Example usage
    # edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 6, 7, 7, 8, 8],
    #                            [1, 8, 0, 1, 1, 3, 2, 4, 3, 7, 6, 8, 7, 0]], dtype=torch.long)
    #
    # groups = [[0, 1, 2], [3, 4], [5], [6, 7], [8]]  # Coarse-grained groups: atoms [0, 1, 2] -> group 0, atoms [3, 4] -> group 1
    #
    # cg_edge_index = build_coarse_grained_edges(edge_index, groups)
    # print(cg_edge_index)


if __name__ == '__main__':
    main()