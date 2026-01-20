# MolCluster

MolCluster is an unsupervised machine learning framework for automatically determining coarse-grained (CG) representations of molecules by combining graph neural networks (GNNs) with community detection algorithms.

This code accompanies our research on data-driven molecular coarse-graining, where atomistic molecular graphs are mapped to chemically meaningful CG representations without predefined grouping rules.

---

## Overview

Coarse-grained (CG) modeling is a key technique for extending the accessible time and length scales of molecular simulations. However, traditional CG mapping schemes often rely on handcrafted rules or predefined force-field templates, which limits their generality and transferability.

MolCluster provides a fully unsupervised alternative by:

- Representing molecules as atomistic graphs  
- Learning chemically informed edge weights using a GNN  
- Applying community detection to identify CG groups  
- Generating CG mappings at multiple resolutions via a tunable parameter  

The framework is general, interpretable, and applicable to a wide range of molecular systems.

---

## Methodology

The MolCluster workflow consists of four main steps:

1. **Molecular Graph Construction**  
   Atoms are treated as nodes, and chemical bonds or interactions are encoded as edges.

2. **GNN-based Edge Weight Learning**  
   A graph neural network learns a weighted adjacency matrix that reflects chemical connectivity and similarity.

3. **Community Detection**  
   A community detection algorithm (e.g., Leiden) is applied to the weighted graph, where each detected community corresponds to one CG bead.

4. **Resolution Control**  
   A resolution parameter controls the granularity of the CG mapping, enabling multiple CG representations for the same molecule.

---

## Installation (Conda / YAML)

We recommend using conda to ensure a reproducible environment.

### Create the environment

```bash
conda env create -f environment.yaml

### Activate the environment

```bash
conda activate molcluster

