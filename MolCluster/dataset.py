import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, Subset
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.loader import DataLoader
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from torch.utils.data import random_split
import os
import json
import random


def normalize_dict(input_dict):
    df = pd.DataFrame(list(input_dict.items()), columns=['Key', 'Value'])
    min_max_scaler = MinMaxScaler()
    df['Value_Normalized'] = min_max_scaler.fit_transform(df[['Value']])
    return dict(zip(df['Key'], df['Value_Normalized']))


# Element = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Fe', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Ag', 'Cd', 'Sn', 'Te', 'I']

AtomicNumber = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10, 'Na':11, 'Mg':12, 'Al':13,
                'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18, 'K':19, 'Ca':20, 'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25,
                'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
                'Rb':37, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47,
                'Cd':48, 'In':49, 'Sn':50, 'Sb':51, 'Te':52, 'I':53, 'Xe':54, 'Cs':55, 'Ba':56, 'La':57, 'Ce':58,
                'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69,
                'Yb':70, 'Lu':71, 'Hf':72, 'Ta':73, 'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
                'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86, 'Fr':87, 'Ra':88, 'Ac':89, 'Th':90, 'Pa':91,
                'U':92, 'Np':93, 'Pu':94, 'Am':95, 'Cm':96, 'Bk':97, 'Cf':98, 'Es':99, 'Fm':100, 'Md':101, 'No':102,
                'Lr':103, 'Rf':104, 'Db':105, 'Sg':106, 'Bh':107, 'Hs':108, 'Mt':109, 'Ds':110, 'Rg':111, 'Cn':112,
                'Nh':113, 'Fl':114, 'Mc':115, 'Lv':116, 'Ts':117, 'Og':118}
Element = list(AtomicNumber.keys())

# Thermochemical electronegativities of the elements from https://www.nature.com/articles/s41467-021-22429-0
Electronegativity = {'H':3.04, 'He':4.42, 'Li':2.17, 'Be':2.42, 'B':3.04, 'C':3.15, 'N':3.56, 'O':3.78, 'F':4.00,
                     'Ne':4.44, 'Na':2.15, 'Mg':2.39, 'Al':2.52, 'Si':2.82, 'P':3.16, 'S':3.44, 'Cl':3.50, 'Ar':3.57,
                     'K':2.07, 'Ca':2.20, 'Sc':2.35, 'Ti':2.23, 'V':2.08, 'Cr':2.12, 'Mn':2.20, 'Fe':2.32, 'Co':2.34,
                     'Ni':2.32, 'Cu':2.86, 'Zn':2.26, 'Ga':2.43, 'Ge':2.79, 'As':3.15, 'Se':3.37, 'Br':3.45, 'Kr':3.37,
                     'Rb':2.07, 'Sr':2.13, 'Y':2.52, 'Zr':2.05, 'Nb':2.59, 'Mo':2.47, 'Tc':2.82, 'Ru':2.68, 'Rh':2.65,
                     'Pd':2.70, 'Ag':2.88, 'Cd':2.36, 'In':2.29, 'Sn':2.68, 'Sb':3.05, 'Te':3.14, 'I':3.20, 'Xe':3.12,
                     'Cs':1.97, 'Ba':2.02, 'Hf':2.01, 'Ta':2.32, 'W':2.42, 'Re':2.59, 'Os':2.72, 'Ir':2.79, 'Pt':2.98,
                     'Au':2.81, 'Hg':2.92, 'Tl':2.26, 'Pb':2.62, 'Bi':2.69, 'Po':2.85, 'At':3.04, 'Rn':3.04, 'Fr':2.01,
                     'Ra':2.15, 'Rf':2.27, 'Db':2.38, 'Sg':2.51, 'Bh':2.48, 'Hs':2.52, 'Mt':2.66, 'Ds':2.73, 'Rg':2.83,
                     'Cn':3.03, 'Nh':2.49, 'Fl':2.57, 'Mc':2.21, 'Lv':2.42, 'Ts':2.61, 'Og':2.59, 'La':2.49, 'Ce':2.61,
                     'Pr':2.24, 'Nd':2.11, 'Pm':2.24, 'Sm':1.90, 'Eu':1.81, 'Gd':2.40, 'Tb':2.29, 'Dy':2.07, 'Ho':2.12,
                     'Er':2.02, 'Tm':2.03, 'Yb':1.78, 'Lu':2.68, 'Ac':2.22, 'Th':2.62, 'Pa':2.33, 'U':2.45, 'Np':2.35,
                     'Pu':2.22, 'Am':2.28, 'Cm':2.31, 'Bk':2.08, 'Cf':2.18, 'Es':2.29, 'Fm':2.38, 'Md':2.47, 'No':2.06,
                     'Lr':2.10}
Electronegativity = normalize_dict(Electronegativity)

# first ionization energy in kJ/mol  from https://chem.libretexts.org/Courses/Oregon_Institute_of_Technology/OIT%3A_CHE_202_-_General_Chemistry_II/Unit_3%3A_Periodic_Patterns/3.3%3A_Trends_in_Ionization_Energy
FirstIonizationEnergy = {'H':1310, 'He':2370, 'Li':520, 'Be':900, 'B':800, 'C':1090, 'N':1400, 'O':1310, 'F':1680,
                         'Ne':2080, 'Na':490, 'Mg':730, 'Al':580, 'Si':780, 'P':1060, 'S':1000, 'Cl':1250, 'Ar':1520,
                         'K':420, 'Ca':590, 'Sc':630, 'Ti':660, 'V':650, 'Cr':660, 'Mn':710, 'Fe':760, 'Co':760,
                         'Ni':730, 'Cu':740, 'Zn':910, 'Ga':580, 'Ge':780, 'As':960, 'Se':950, 'Br':1140, 'Kr':1350,
                         'Rb':400, 'Sr':550, 'Y':620, 'Zr':660, 'Nb':670, 'Mo':680, 'Tc':700, 'Ru':710, 'Rh':720,
                         'Pd':800, 'Ag':730, 'Cd':870, 'In':560, 'Sn':700, 'Sb':830, 'Te':870, 'I':1010, 'Xe':1170,
                         'Cs':380, 'Ba':500, 'La':540, 'Hf':700, 'Ta':760, 'W':770, 'Re':760, 'Os':840, 'Ir':890,
                         'Pt':870, 'Au':890, 'Hg':1000, 'Tl':590, 'Pb':710, 'Bi':800, 'Po':810, 'At':920, 'Rn':1030,
                         'Fr':380, 'Ra':510}
FirstIonizationEnergy = normalize_dict(FirstIonizationEnergy)

# electron affinity in eV from https://pubs.acs.org/doi/abs/10.1021/ed067p307
ElectronAffinity = {'H':0.754, 'He':0.0, 'Li':0.618, 'Be':0.0, 'B':0.277, 'C':1.263, 'N':0.0, 'O':1.461, 'F':3.399,
                    'Na':0.548, 'Mg':0.0, 'Al':0.441, 'Si':1.385, 'P':0.746, 'S':2.077, 'Cl':3.617, 'Ar':0, 'K':0.501,
                    'Ca':0.0, 'Sc':0.188, 'Ti':0.079, 'V':0.525, 'Cr':0.666, 'Mn':0.0, 'Fe':0.163, 'Co':0.661,
                    'Ni':1.156, 'Cu':1.228, 'Zn':0.0, 'Ga':0.3, 'Ge':1.35, 'As':0.81, 'Se':2.021, 'Br':3.365, 'Kr':0.0,
                    'Rb':0.486, 'Sr':0.0, 'Y':0.307, 'Zr':0.426, 'Nb':0.893, 'Mo':0.746, 'Tc':0.55, 'Ru':1.05, 'Rh':1.137,
                    'Pd':0.557, 'Ag':1.302, 'Cd':0.0, 'In':0.3, 'Sn':1.2, 'Sb':1.07, 'Te':1.971, 'I':3.059,'Xe':0.0,
                    'Cs':0.472, 'Ba':0.0, 'La':0.5, 'Hf':0.0, 'Ta':0.322, 'W':0.815, 'Re':0.15, 'Os':1.1, 'Ir':1.565,
                    'Pt':2.128, 'Au':2.309, 'Hg':0.0, 'Tl':0.2, 'Pb':0.36, 'Bi':0.946, 'Po':1.9, 'At':2.8, 'Rn':0.0,
                    'Fr':0.470, 'Ra':0.0, 'Ce':0.5}
ElectronAffinity = normalize_dict(ElectronAffinity)

# atomic radii in e*bohr−3 from https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/chem.201602949
AtomicRadius = {'H':1.54, 'He':1.34, 'Li':2.20, 'Be':2.19, 'B':2.05, 'C':1.90, 'N':1.79, 'O':1.71, 'F':1.63, 'Ne':1.56,
                'Na':2.25, 'Mg':2.40, 'Al':2.39, 'Si':2.32, 'P':2.23, 'S':2.14, 'Cl':2.06, 'Ar':1.97, 'K':2.34,
                'Ca':2.70, 'Sc':2.63, 'Ti':2.37, 'V':2.52, 'Cr':2.33, 'Mn':2.42, 'Fe':2.26, 'Co':2.22, 'Ni':2.19,
                'Cu':2.17, 'Zn':2.22, 'Ga':2.33, 'Ge':2.34, 'As':2.31, 'Se':2.24, 'Br':2.19, 'Kr':2.12, 'Rb':2.40,
                'Sr':2.79, 'Y':2.74, 'Zr':2.68, 'Nb':2.51, 'Mo':2.44, 'Tc':2.41, 'Ru':2.37, 'Rh':2.33, 'Pd':2.15,
                'Ag':2.25, 'Cd':2.38, 'In':2.46, 'Sn':2.48, 'Sb':2.46, 'Te':2.42, 'I':2.38, 'Xe':2.32, 'Cs':2.49,
                'Ba':2.93, 'La':2.84, 'Ce':2.82, 'Pr':2.86, 'Nd':2.84, 'Pm':2.83, 'Sm':2.80, 'Eu':2.80, 'Gd':2.77,
                'Tb':2.76, 'Dy':2.75, 'Ho':2.73, 'Er':2.72, 'Tm':2.71, 'Yb':2.77, 'Lu':2.70, 'Hf':2.64, 'Ta':2.58,
                'W':2.53, 'Re':2.49, 'Os':2.44, 'Ir':2.33, 'Pt':2.30, 'Au':2.26, 'Hg':2.29, 'Tl':2.42, 'Pb':2.49,
                'Bi':2.50, 'Po':2.50, 'At':2.47, 'Rn':2.43, 'Fr':2.58, 'Ra':2.92, 'Ac':2.93, 'Th':2.89, 'Pa':2.85,
                'U':2.83, 'Np':2.80, 'Pu':2.78, 'Am':2.76, 'Cm':2.64}

AtomicRadius = normalize_dict(AtomicRadius)

# Valence Electrons from https://en.wikipedia.org/wiki/Valence_electron
ValenceElectrons = {'H':1, 'He':2, 'Li':1, 'Be':2, 'B':3, 'C':4, 'N':5, 'O':6, 'F':7, 'Ne':8, 'Na':1, 'Mg':2, 'Al':3,
                    'Si':4, 'P':5, 'S':6, 'Cl':7, 'Ar':8, 'K':1, 'Ca':2, 'Sc':3, 'Ti':4, 'V':5, 'Cr':6, 'Mn':7, 'Fe':8,
                    'Co':9, 'Ni':10, 'Cu':11, 'Zn':12, 'Ga':3, 'Ge':4, 'As':5, 'Se':6, 'Br':7, 'Kr':8, 'Rb':1, 'Sr':2,
                    'Y':3, 'Zr':4, 'Nb':5, 'Mo':6, 'Tc':7, 'Ru':8, 'Rh':9, 'Pd':10, 'Ag':11, 'Cd':12, 'In':3, 'Sn':4,
                    'Sb':5, 'Te':6, 'I':7, 'Xe':8, 'Cs':1, 'Ba':2, 'La':3, 'Ce':4, 'Pr':5, 'Nd':6, 'Pm':7, 'Sm':8,
                    'Eu':9, 'Gd':10, 'Tb':11, 'Dy':12, 'Ho':13, 'Er':14, 'Tm':15, 'Yb':16, 'Lu':3, 'Hf':4, 'Ta':5,
                    'W':6, 'Re':7, 'Os':8, 'Ir':9, 'Pt':10, 'Au':11, 'Hg':12, 'Tl':3, 'Pb':4, 'Bi':5, 'Po':6, 'At':7,
                    'Rn':8, 'Fr':1, 'Ra':2, 'Ac':3, 'Th':4, 'Pa':5, 'U':6, 'Np':7, 'Pu':8, 'Am':9, 'Cm':10, 'Bk':11,
                    'Cf':12, 'Es':13, 'Fm':14, 'Md':15, 'No':16, 'Lr':3, 'Rf':4, 'Db':5, 'Sg':6, 'Bh':7, 'Hs':8,
                    'Mt':9, 'Ds':10, 'Rg':11, 'Cn':12, 'Nh':3, 'Fl':4, 'Mc':5, 'Lv':6, 'Ts':7, 'Og':8}
ValenceElectrons = normalize_dict(ValenceElectrons)

# from rdkit
# element = rdchem.GetPeriodicTable().GetElementSymbol(atomic_num)
# mass = rdchem.GetPeriodicTable().GetAtomicWeight(atomic_num)
AtomicMass = {'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.812, 'C': 12.011, 'N': 14.007, 'O': 15.999,
              'F': 18.998, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.067,
              'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.944,
              'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.39,
              'Ga': 69.723, 'Ge': 72.61, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.8, 'Rb': 85.468,
              'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.94, 'Tc': 98.0, 'Ru': 101.07,
              'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.412, 'In': 114.818, 'Sn': 118.711, 'Sb': 121.76,
              'Te': 127.6, 'I': 126.904, 'Xe': 131.29, 'Cs': 132.905, 'Ba': 137.328, 'La': 138.906, 'Ce': 140.116,
              'Pr': 140.908, 'Nd': 144.24, 'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.925,
              'Dy': 162.5, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.934, 'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49,
              'Ta': 180.948, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.078, 'Au': 196.967,
              'Hg': 200.59, 'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0,
              'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.038, 'Pa': 231.036, 'U': 238.029, 'Np': 237.0,
              'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0,
              'No': 259.0, 'Lr': 262.0, 'Rf': 267.0, 'Db': 268.0, 'Sg': 269.0, 'Bh': 270.0, 'Hs': 269.0, 'Mt': 278.0,
              'Ds': 281.0, 'Rg': 281.0, 'Cn': 285.0, 'Nh': 284.0, 'Fl': 289.0, 'Mc': 288.0, 'Lv': 293.0, 'Ts': 292.0,
              'Og': 294.0}
AtomicMass = normalize_dict(AtomicMass)

# Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.OTHER
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED]
len_HYBRIDIZATION = len(HYBRIDIZATION)

BOND_TYPE = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def compute_bonded_potential(bonded_para, r):
    # https://pubs.acs.org/doi/10.1021/ja00051a040
    # kb, r0 = AllChem.GetUFFBondStretchParams(mol, i, j)
    kb, r0 = bonded_para[0], bonded_para[1]
    return 0.5 * kb * (r - r0) ** 2  # kcal/mol

def compute_lj_potential(LJ_para, r):
    # https://pubs.acs.org/doi/10.1021/ja00051a040
    # LJ_para = AllChem.GetUFFVdWParams(mol, i, j)
    # if LJ_para is None:
    #     raise ValueError(
    #         f"Unsupported atom pair: {mol.GetAtomWithIdx(i).GetSymbol()} and {mol.GetAtomWithIdx(j).GetSymbol()} for LJ parameters")
    xij, Dij = LJ_para[0], LJ_para[1]
    U_r_12 = Dij * (xij / r)**12
    U_r_6 = - 2 * Dij * (xij / r)**6
    U_r = U_r_12 - U_r_6  # 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

    # if U_r_12 > 1e10:
    #     print("U_r_12: ", U_r_12)
    #     U_r_12 = 1e10
    # if U_r_6 < -1e10:
    #     print("U_r_6: ", U_r_6)
    #     U_r_6 = -1e10

    return U_r_12, U_r_6  # kcal/mol


def compute_electrostatic_potential(mol, i, j, r):
    # https://pubs.acs.org/doi/10.1021/ja00051a040
    charge_i = float(mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
    if np.isnan(charge_i):
        charge_i = mol.GetAtomWithIdx(i).GetFormalCharge()

    charge_j = float(mol.GetAtomWithIdx(j).GetDoubleProp("_GasteigerCharge"))
    if np.isnan(charge_j):
        charge_j = mol.GetAtomWithIdx(j).GetFormalCharge()  # Formal charge

    if np.isinf(charge_i) or np.isnan(charge_i):
        charge_i = 0.0
    if np.isinf(charge_j) or np.isnan(charge_j):
        charge_j = 0.0

    electrostatic_potential = 332.0637 * (charge_i * charge_j) / r  # kcal/mol






    return electrostatic_potential

def smiles_to_molecular_graph(smiles, edge_method, random_seed, r_eps=1e-1):
    """
    Converts a SMILES string into a molecular graph and returns the graph in PyTorch Geometric format.

    Parameters:
    - smiles (str): SMILES representation of the molecule.

    Returns:
    - data (torch_geometric.data.Data): Molecular graph data containing node features and edge indices.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    embed_params = AllChem.ETKDGv3()
    if random_seed is not None:
        embed_params.randomSeed = random_seed
    AllChem.EmbedMolecule(mol, embed_params)
    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    # AllChem.ComputeGasteigerCharges(mol)
    mol = AllChem.RemoveAllHs(mol)
    conf = mol.GetConformer()
    smiles_updated = Chem.MolToSmiles(mol)

    AllChem.ComputeGasteigerCharges(mol)

    # Generate node features: could include atom type, degree, charge, etc.
    atom_features = []
    atom_types = []
    atom_numbers = []

    # atom features aromaticity[1] + degree[9] + num_hydrogens[9] + hybridization[6] + charge[1] + mass[1] + radius[1]
    # + electron_affinity[1] + first_ionization_energy[1] + electronegativity[1] + valence_electrons[1] = 32
    for atom in mol.GetAtoms():

        aromaticity = [int(atom.GetIsAromatic())]

        degree = [0] * 9  # Assuming degrees up to 8
        if atom.GetDegree() > 8:
            degree[9] = 1
        else:
            degree[atom.GetDegree()] = 1

        num_hydrogens = [0] * 9  # Assuming up to 8 hydrogens
        num_hydrogens[atom.GetTotalNumHs()] = 1

        hybridization = [0] * len_HYBRIDIZATION  # Assuming 6 possible hybridization types
        hybridization_type = atom.GetHybridization()
        if hybridization_type not in HYBRIDIZATION:
            # raise ValueError(f"Unknown hybridization type: {hybridization_type}")
            print(f"Unknown hybridization type: {hybridization_type}")
            hybridization[5] = 1  # Chem.rdchem.HybridizationType.UNSPECIFIED
        else:
            hybridization[HYBRIDIZATION.index(hybridization_type)] = 1

        charge = float(atom.GetProp('_GasteigerCharge'))
        if np.isnan(charge):
            charge = [atom.GetFormalCharge() / 8]  # Formal charge
        else:
            charge = [round(charge / 8, 5)]  # renormalizaiton to 1

        if np.isinf(charge):  # `'C[Se]CCC1OCC2OC(n3cc(C)c(=O)[nH]c3=O)CC2N1O'` of MUV dataset will get a inf value of charge via `atom.GetProp('_GasteigerCharge')` at `atom.GetIdx()=15;atom.GetSymbol()='O'`
            charge = [0.0]

        atom_symbol = atom.GetSymbol()
        mass = [AtomicMass.get(atom_symbol, 0.0)]
        radius = [AtomicRadius.get(atom_symbol, 0.0)]
        electron_affinity = [ElectronAffinity.get(atom_symbol, 0.0)]
        first_ionization_energy = [FirstIonizationEnergy.get(atom_symbol, 0.0)]
        electronegativity = [Electronegativity.get(atom_symbol, 0.0)]
        valence_electrons = [ValenceElectrons.get(atom_symbol, 0.0)]

        atom_feature = aromaticity + degree + num_hydrogens + hybridization + charge + mass + radius + electron_affinity + first_ionization_energy + electronegativity + valence_electrons

        atom_features.append(atom_feature)
        atom_types.append(atom.GetSymbol())
        atom_numbers.append(AtomicNumber.get(atom_symbol))

    # Convert node features to a tensor
    x = torch.LongTensor(atom_numbers).reshape(-1, 1)
    atom_features = torch.tensor(atom_features, dtype=torch.float)
    atom_types = torch.tensor([Chem.GetPeriodicTable().GetAtomicNumber(symbol) for symbol in atom_types],
                              dtype=torch.long)

    # Generate edge indices and edge attributes (bond types)
    edge_index = []
    edge_attr = []

    bonded_edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bonded_edge_index.append([i, j])
        bonded_edge_index.append([j, i])

    if edge_method == 'bonded':  # edge_attr 7+1+4+1 = 13
        for bond in mol.GetBonds():
            bond_length = [AllChem.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]  # (Angstrom)

            if bond_length[0] < r_eps:
                continue

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bonded_para = AllChem.GetUFFBondStretchParams(mol, i, j)
            if bonded_para is None:
                print(f'Unsupported atom bond: {mol.GetAtomWithIdx(i).GetSymbol()} and {mol.GetAtomWithIdx(j).GetSymbol()} for bonded parameters')
                continue

            # Add both directions (i->j and j->i) for undirected graph representation
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_dir = [0] * 7
            bond_dir[bond.GetBondDir()] = 1

            # in_ring = [int(bond.IsInRing())]

            edge_type = [0]*4
            bond_type = bond.GetBondType()
            if bond_type not in BOND_TYPE:
                print(f"Unsupported bond type: {bond_type}")
            else:
                edge_type[BOND_TYPE.index(bond_type)] = 1

            # bond_length = [AllChem.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]  # (Angstrom)
            bonded_potential = [compute_bonded_potential(bonded_para, bond_length[0])]

            edge_attr_item = bond_dir + edge_type + bond_length + bonded_potential
            edge_attr.append(edge_attr_item)
            edge_attr.append(edge_attr_item)

    elif list(edge_method.keys())[0] == 'bonded+nonbonded':  # edge_attr 7+1+5+1+3 = 17

        for bond in mol.GetBonds():
            bond_length = [AllChem.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]  # (Angstrom)
            if bond_length[0] < r_eps:
                continue

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bonded_para = AllChem.GetUFFBondStretchParams(mol, i, j)
            if bonded_para is None:
                print(
                    f'Unsupported atom bond: {mol.GetAtomWithIdx(i).GetSymbol()} and {mol.GetAtomWithIdx(j).GetSymbol()} for bonded parameters')
                continue

            # Add both directions (i->j and j->i) for undirected graph representation
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_dir = [0] * 7
            bond_dir[bond.GetBondDir()] = 1

            # in_ring = [int(bond.IsInRing())]

            edge_type = [0]*5
            bond_type = bond.GetBondType()
            if bond_type not in BOND_TYPE:
                print(f"Unsupported bond type: {bond_type}")
            else:
                edge_type[BOND_TYPE.index(bond_type)] = 1

            bonded_potential = [compute_bonded_potential(bonded_para, bond_length[0])]

            edge_attr_item = bond_dir + edge_type + bond_length + bonded_potential + [0, 0, 0]
            edge_attr.append(edge_attr_item)
            edge_attr.append(edge_attr_item)

        rcut = edge_method['bonded+nonbonded']
        for i in range(mol.GetNumAtoms()):
            for j in range(i+1, mol.GetNumAtoms()):
                dist = AllChem.GetBondLength(conf, i, j)
                if r_eps < dist < rcut:
                    if [i, j] not in edge_index:
                        lj_para = AllChem.GetUFFVdWParams(mol, i, j)
                        if lj_para is None:
                            print(f"Unsupported atom pair: {mol.GetAtomWithIdx(i).GetSymbol()} and {mol.GetAtomWithIdx(j).GetSymbol()} for LJ parameters")
                            continue
                        LJ = list(compute_lj_potential(lj_para, dist))  # [LJ_12, LJ_6]
                        Colulomb = [compute_electrostatic_potential(mol, i, j, dist)]
                        bond_dir = [0] * 7
                        edge_type = [0] * 5
                        edge_type[4] = 1
                        # in_ring = [0]
                        bonded_potential = [0]
                        bond_length = [dist]  # (Angstrom)
                        edge_attr_item = bond_dir + edge_type + bond_length + bonded_potential + LJ + Colulomb
                        edge_attr.append(edge_attr_item)
                        edge_attr.append(edge_attr_item)

                        edge_index.append([i, j])
                        edge_index.append([j, i])

    elif list(edge_method.keys())[0] == 'nonbonded':  # edge_attr 1+3 = 4

        rcut = edge_method['nonbonded']
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                dist = AllChem.GetBondLength(conf, i, j)
                if r_eps < dist < rcut:
                    lj_para = AllChem.GetUFFVdWParams(mol, i, j)
                    if lj_para is None:
                        print(
                            f"Unsupported atom pair: {mol.GetAtomWithIdx(i).GetSymbol()} and {mol.GetAtomWithIdx(j).GetSymbol()} for LJ parameters")
                        continue

                    LJ = list(compute_lj_potential(lj_para, dist))  # [LJ_12, LJ_6]
                    Colulomb = [compute_electrostatic_potential(mol, i, j, dist)]
                    bond_length = [dist]  # (Angstrom)
                    edge_attr_item = bond_length + LJ + Colulomb
                    edge_attr.append(edge_attr_item)
                    edge_attr.append(edge_attr_item)

                    edge_index.append([i, j])
                    edge_index.append([j, i])

    else:
        raise ValueError(f"Unsupported edge_method: {edge_method}")


    # Convert edge indices and edge attributes to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose to match PyG format
    bonded_edge_index = torch.tensor(bonded_edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Package the data into a PyTorch Geometric Data object
    data = Data(x=x, atom_features=atom_features, edge_index=edge_index, edge_attr=edge_attr, bonded_edge_index=bonded_edge_index)
    data.atom_types = atom_types
    data.smiles = smiles

    return data


def load_smiles_from_file(file_path, skip_lines=0):
    """
    Loads a list of SMILES strings from a file, optionally skipping the first few lines.

    Parameters:
    - file_path (str): Path to the file containing SMILES strings.
    - skip_lines (int): Number of lines to skip before starting to read SMILES strings.

    Returns:
    - smiles_list (list): List of SMILES strings.
    """
    with open(file_path, 'r') as f:
        # Skip the specified number of lines
        for _ in range(skip_lines):
            next(f)
        # Read the remaining lines as SMILES strings
        smiles_list = f.read().splitlines()
    return smiles_list



def preprocess_smiles(smiles_list):
    """
    Preprocesses a list of SMILES strings, converting them into molecular graph data.

    Parameters:
    - smiles_list (list): List of SMILES strings.

    Returns:
    - graph_data_list (list): List of molecular graph data.
    """
    graph_data_list = []
    for smiles in smiles_list:
        try:
            graph_data = smiles_to_molecular_graph(smiles)
            graph_data_list.append(graph_data)
        except ValueError as e:
            print(f"Skipping invalid SMILES: {smiles}. Error: {e}")
    return graph_data_list


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, edge_method='bonded', random_seed=7777):
        self.smiles_list = smiles_list
        self.edge_method = edge_method
        self.random_seed = random_seed

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        data = smiles_to_molecular_graph(smiles=smiles, edge_method=self.edge_method, random_seed=self.random_seed)
        return data


class MoleculeWithPropertyDataset(Dataset):
    def __init__(self, smiles_list, label_list, edge_method='bonded', random_seed=7777):
        self.smiles_list = smiles_list
        self.label_list = label_list
        self.edge_method = edge_method
        self.random_seed = random_seed

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.label_list[idx]
        data = smiles_to_molecular_graph(smiles=smiles, edge_method=self.edge_method, random_seed=self.random_seed)
        return data, label

def RandomSplitter(datasets, val_ratio, test_ratio, batch_size, num_workers, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    train_size = int((1 - val_ratio - test_ratio) * len(datasets))
    val_size = int(len(datasets) * val_ratio)
    test_size = len(datasets) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size], generator=generator)

    # Create DataLoaders for training, validation and test
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, generator=generator) # , drop_last=True
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

# Helper function to generate Bemis-Murcko scaffold
def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MakeScaffoldGeneric(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


# ScaffoldSplitter function
def ScaffoldSplitter(datasets, val_ratio, test_ratio, batch_size, num_workers, seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    N = len(datasets)
    train_ratio = 1 - val_ratio - test_ratio

    # Generate scaffold groups
    scaffolds = defaultdict(list)
    for idx in range(N):
        smiles = datasets.smiles_list[idx]
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(idx)

    # Sort scaffolds by size (descending order)
    sorted_scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    # Allocate indices to train, validation, and test
    train_cutoff = int(train_ratio * N)   # cutoff index
    valid_cutoff = int((train_ratio + val_ratio) * N) - train_cutoff

    train_idx, val_idx, test_idx = [], [], []
    for group in sorted_scaffold_groups:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= valid_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    # Create Subsets for DataLoader
    train_dataset = Subset(datasets, train_idx)
    val_dataset = Subset(datasets, val_idx)
    test_dataset = Subset(datasets, test_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, generator=generator,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


# RandomScaffoldSplitter function
def RandomScaffoldSplitter(datasets, val_ratio, test_ratio, batch_size, num_workers, seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # rng = np.random.RandomState(seed)

        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    N = len(datasets)
    train_ratio = 1 - val_ratio - test_ratio

    rng = np.random.RandomState(seed)

    # Generate scaffold groups
    scaffolds = defaultdict(list)
    for idx in range(N):
        smiles = datasets.smiles_list[idx]
        try:
            scaffold = generate_scaffold(smiles, include_chirality=True)
        except:
            print('Error generating scaffold for molecule {} with SMILES {}'.format(idx, smiles))

        scaffolds[scaffold].append(idx)

    # Shuffle scaffold groups
    scaffold_groups = list(scaffolds.values())
    rng.shuffle(scaffold_groups)

    # Allocate indices to train, validation, and test
    train_cutoff = int(train_ratio * N)
    valid_cutoff = int((train_ratio + val_ratio) * N) - train_cutoff

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= valid_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    # Create Subsets for DataLoader
    train_dataset = Subset(datasets, train_idx)
    val_dataset = Subset(datasets, val_idx)
    test_dataset = Subset(datasets, test_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, generator=generator,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_all_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_list.append(os.path.join(root, file))
    return file_list

CG_TYPE_DICT = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'NDA': 5, 'ND': 6, 'NA': 7, 'N0': 8, 'P1': 9, 'P2': 10,
                'P3': 11, 'P4': 12, 'P5': 13, 'QDA': 14, 'QD': 15, 'QA': 16, 'Q0': 17}
def load_martini_json_file(file_path):

    smiles_list = []
    labels_list = []

    json_list = get_all_files_in_directory(file_path)
    for json_file in json_list:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            smiles = json_data['smiles']
            group = json_data['cgnodes']

            fg_beads = json_data['nodes']
            fg_beads.sort(key=lambda x: x['id'])

            atom_CG_types = torch.LongTensor([list(CG_TYPE_DICT.keys()).index(bead['cg_type']) for bead in fg_beads]).reshape(-1, 1)

            smiles_list.append(smiles)
            labels_list.append([group, atom_CG_types])

    return smiles_list, labels_list



def main():

    json_path = '../../MARTINI2_Dataset'

    load_martini_json_file(json_path)

    smiles_file = '../../test-dataset/grovr-tryout.csv'
    skip_lines = 1
    smiles_list = load_smiles_from_file(file_path=smiles_file, skip_lines=skip_lines)
    graph_data_list = preprocess_smiles(smiles_list)

    pass


if __name__ == "__main__":
    main()