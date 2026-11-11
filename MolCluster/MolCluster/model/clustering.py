import collections
import random
import time
import networkx as nx
import copy
import matplotlib.pyplot as plt
from math import exp
import numpy as np
from typing import Optional


# https://blog.csdn.net/qq_16543881/article/details/122825957


def load_graph(path):
    # Load graph data from a file into a dictionary
    G = collections.defaultdict(dict)
    with open(path) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = int(vertices[0])
            v_j = int(vertices[1])
            w = 1.0  # Default weight is 1.0 if not provided
            G[v_i][v_j] = w
            G[v_j][v_i] = w
    return G


# Vertex class stores community and node ID information
class Vertex:
    def __init__(self, vid, cid, nodes, k_in=0):
        self._vid = vid  # Node ID
        self._cid = cid  # Community ID
        self._nodes = nodes  # Nodes within the community
        self._kin = k_in  # Weight of edges within the community


class Louvain:
    def __init__(self, graph, gamma=1):
        # self._G = graph
        self._G = copy.deepcopy(graph)
        self._gamma = gamma  # Resolution parameter for modularity
        self._m = 0  # Total number of edges (will be updated dynamically)
        self._cid_vertices = {}  # Stores community information (community ID and its nodes)
        self._vid_vertex = {}  # Stores node information (node ID and corresponding Vertex instance)
        for vid in self._G.keys():
            self._cid_vertices[vid] = {vid}  # Initially, each node is its own community
            self._vid_vertex[vid] = Vertex(vid, vid, {vid})  # Each node is its own community ID
            self._m += sum([1 for neighbor in self._G[vid].keys() if neighbor > vid])  # Calculate edge count

    # Modularity optimization phase
    def move_nodes(self, move_node_max_iter=500):
        mod_inc = False  # Indicates whether the algorithm should continue
        visit_sequence = self._G.keys()
        random.shuffle(list(visit_sequence))  # Shuffle node visit order
        iteration = 1
        while True:
            if iteration > move_node_max_iter:
                print(f'leiden clustering iteration in move_nodes process > {move_node_max_iter}, break')
                break
            iteration += 1
            can_stop = True  # Indicates if the first phase can terminate
            for v_vid in visit_sequence:
                v_cid = self._vid_vertex[v_vid]._cid  # Get community ID of the node
                k_v = sum(self._G[v_vid].values()) + self._vid_vertex[v_vid]._kin  # Degree of the node
                cid_Q = {}  # Stores communities with positive modularity gain
                for w_vid in self._G[v_vid].keys():  # Iterate over neighbors
                    w_cid = self._vid_vertex[w_vid]._cid  # Get community ID of the neighbor
                    if w_cid in cid_Q:
                        continue
                    else:
                        tot = sum([sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        k_v_in = sum([v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        delta_Q = k_v_in - self._gamma * k_v * tot / self._m
                        cid_Q[w_cid] = delta_Q
                cid, max_delta_Q = sorted(cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                if max_delta_Q > 0.0 and cid != v_cid:  # Update community if modularity gain is positive
                    self._vid_vertex[v_vid]._cid = cid
                    self._cid_vertices[cid].add(v_vid)
                    self._cid_vertices[v_cid].remove(v_vid)
                    can_stop = False
                    mod_inc = True
            if can_stop:  # Terminate if no updates
                break
        return mod_inc

    # Network aggregation phase
    def aggregate(self):
        cid_vertices = {}
        vid_vertex = {}
        for cid, vertices in self._cid_vertices.items():  # Iterate over communities
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            for vid in vertices:  # Treat all nodes in a community as a single node
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                for k, v in self._G[vid].items():  # Calculate total edge weight within the community
                    if k in vertices:
                        new_vertex._kin += v / 2.0
            cid_vertices[cid] = {cid}  # Update community ID
            vid_vertex[cid] = new_vertex
        G = collections.defaultdict(dict)
        for cid1, vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2, vertices2 in self._cid_vertices.items():
                if cid2 <= cid1 or len(vertices2) == 0:
                    continue
                edge_weight = 0.0
                for vid in vertices1:
                    for k, v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
        self._cid_vertices = cid_vertices  # Update communities and nodes
        self._vid_vertex = vid_vertex
        # self._G = G
        self._G = copy.deepcopy(G)

    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():  # Gather all non-empty communities
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(list(c))
        return communities

    def get_hierarchical_communities(self, hierarchy=None):
        if hierarchy is None:
            return self.hierarchical_communities
        else:
            return self.hierarchical_communities[hierarchy]

    def execute(self, max_iter=100, recoder_hierarchy=False, move_node_max_iter=500):
        if recoder_hierarchy:
            self.hierarchical_communities = []
        iter_time = 0
        while True:
            mod_inc = self.move_nodes(move_node_max_iter=500)  # Optimize modularity
            if recoder_hierarchy:
                self.hierarchical_communities.append(self.get_communities())
            if iter_time >= max_iter:
                return self.get_communities()
            iter_time += 1
            if mod_inc:
                self.aggregate()  # Aggregate graph
            else:
                return self.get_communities()



# Implementation of the Leiden algorithm
class Leiden:
    def __init__(self, graph, nx_graph: nx.Graph, gamma=1, gamma2=0.01, theta=0.3):
        """
        Initialize the Leiden algorithm with parameters.
        """
        self._G = copy.deepcopy(graph)
        self.graph = copy.deepcopy(nx_graph)
        self._gamma = gamma  # Resolution parameter for modularity
        self._gamma2 = gamma2  # Parameter for refinement phase
        self._theta = theta  # Temperature for weighted random selection
        self._m = 0  # Total number of edges
        self._cid_vertices = {}  # Communities
        self._vid_vertex = {}  # Nodes and their properties
        for vid in self._G.keys():
            self._cid_vertices[vid] = {vid}
            self._vid_vertex[vid] = Vertex(vid, vid, {vid})
            self._m += sum(1 for neighbor in self._G[vid].keys() if neighbor > vid)

    # Move nodes to optimize modularity
    def move_nodes(self, move_node_max_iter=500):
        """
        First phase of the Leiden algorithm: Local movement of nodes to improve modularity.
        """
        self._cid_vertices_singleton = copy.deepcopy(self._cid_vertices)
        self._vid_vertex_singleton = copy.deepcopy(self._vid_vertex)
        visit_sequence = list(self._G.keys())
        random.shuffle(list(visit_sequence))  # Randomize node visit order
        iteration = 1

        while visit_sequence:
            if iteration > move_node_max_iter:
                print(f'leiden clustering iteration in move_nodes process > {move_node_max_iter}, break')
                break
            iteration += 1
            v_vid = visit_sequence.pop(0)  # Process nodes in randomized order
            v_cid = self._vid_vertex[v_vid]._cid
            k_v = sum(self._G[v_vid].values()) + self._vid_vertex[v_vid]._kin  # Degree of the node
            cid_Q = {}  # Store communities with modularity gains
            for w_vid in self._G[v_vid].keys():
                w_cid = self._vid_vertex[w_vid]._cid
                if w_cid in cid_Q:
                    continue
                tot = sum(
                    [sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]]
                )
                if w_cid == v_cid:
                    tot -= k_v
                k_v_in = sum([v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                delta_Q = k_v_in - self._gamma * k_v * tot / self._m
                cid_Q[w_cid] = delta_Q
            if cid_Q:
                cid, max_delta_Q = sorted(cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                if max_delta_Q > 0.0 and cid != v_cid:
                    self._vid_vertex[v_vid]._cid = cid
                    self._cid_vertices[cid].add(v_vid)
                    self._cid_vertices[v_cid].remove(v_vid)
                    for neighbor in self._G[v_vid].keys():
                        if (self._vid_vertex[neighbor]._cid != cid) and (neighbor not in visit_sequence):
                            visit_sequence.append(neighbor)

    # Refine communities after local movement
    def refine_community(self):
        """
        Refine the partitioning of nodes within communities.
        """
        for community_id, community in self._cid_vertices.items():
            if not community:
                continue
            size_community = sum(len(self._vid_vertex[i]._nodes) for i in community)
            R = set()
            for v in community:
                size_v = len(self._vid_vertex[v]._nodes)
                if nx.cut_size(self.graph, [v], community - {v}) >= self._gamma2 * size_v * (size_community - size_v):
                    R.add(v)

            for v in R:
                v_cid_singleton = self._vid_vertex_singleton[v]._cid
                if len(self._cid_vertices_singleton[v]) == 1:
                    T = set()
                    for c_id, community_singleton in self._cid_vertices_singleton.items():
                        size_community_singleton = len(self._vid_vertex_singleton[c_id]._nodes)
                        if (
                            community_singleton <= community
                            and nx.cut_size(self.graph, community_singleton, community - community_singleton)
                            >= self._gamma2 * size_community_singleton * (size_community - size_community_singleton)
                            and c_id != v_cid_singleton
                        ):
                            T.add(c_id)

                    candidates = [
                        (t, self.calculate_delta_Q(v, t)) for t in T if self.calculate_delta_Q(v, t) > 0
                    ]
                    if candidates:
                        try:
                            weights = [exp(delta_Q / self._theta) for _, delta_Q in candidates]
                        except:
                            weights = [delta_Q for _, delta_Q in candidates]
                        target_cid = random.choices(candidates, weights=weights, k=1)[0][0]
                        self._vid_vertex_singleton[v]._cid = target_cid
                        self._cid_vertices_singleton[target_cid].add(v)
                        self._cid_vertices_singleton[v_cid_singleton].remove(v)

        mod_inc = self._cid_vertices != self._cid_vertices_singleton
        if mod_inc:
            self._vid_vertex = copy.deepcopy(self._vid_vertex_singleton)
            self._cid_vertices = copy.deepcopy(self._cid_vertices_singleton)
        return mod_inc

    # Aggregate communities into supernodes
    def aggregate(self):
        """
        Aggregates communities into supernodes for the next phase of the Leiden algorithm.
        """
        new_graph = nx.Graph()
        cid_vertices = {}
        vid_vertex = {}
        for cid, vertices in self._cid_vertices.items():
            if not vertices:
                continue
            new_vertex = Vertex(cid, cid, set())
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                for k, v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v / 2.0
            cid_vertices[cid] = {cid}
            vid_vertex[cid] = new_vertex
        G = collections.defaultdict(dict)
        for cid1, vertices1 in self._cid_vertices.items():
            if not vertices1:
                continue
            for cid2, vertices2 in self._cid_vertices.items():
                if cid2 <= cid1 or not vertices2:
                    continue
                edge_weight = 0.0
                for vid in vertices1:
                    for k, v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
                    new_graph.add_edge(cid1, cid2, weight=edge_weight)
        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = copy.deepcopy(G)
        self.graph = copy.deepcopy(new_graph)

    def execute(self, max_iter=100, recoder_hierarchy=False, move_node_max_iter=500):
        """
        Execute the Leiden algorithm, repeating until no further improvement.
        """
        if recoder_hierarchy:
            self.hierarchical_communities = []
        iter_time = 0
        while True:
            self.move_nodes(move_node_max_iter=500)
            mod_inc = self.refine_community()
            if recoder_hierarchy:
                self.hierarchical_communities.append(self.get_communities())
            if iter_time >= max_iter:
                return self.get_communities()
            iter_time += 1
            if mod_inc:
                self.aggregate()
            else:
                return self.get_communities()

    def calculate_delta_Q(self, v, target_community):
        """
        Calculate the modularity gain ΔQ for moving a node to a target community.
        """
        v_cid = self._vid_vertex_singleton[v]._cid
        k_v = sum(self._G[v].values()) + self._vid_vertex_singleton[v]._kin
        tot = sum(
            sum(self._G[k].values()) + self._vid_vertex_singleton[k]._kin
            for k in self._cid_vertices_singleton[target_community]
        )
        if v_cid == target_community:
            tot -= k_v
        k_v_in = sum(
            weight for neighbor, weight in self._G[v].items() if neighbor in self._cid_vertices_singleton[target_community]
        )
        delta_Q = k_v_in - self._gamma * k_v * tot / self._m
        return delta_Q

    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(list(c))
        return communities

    def get_hierarchical_communities(self, hierarchy=None):
        if hierarchy is None:
            return self.hierarchical_communities
        else:
            return self.hierarchical_communities[hierarchy]


# Visualize community results
def showCommunity(G, partition, pos):
    cluster = {}
    labels = {}
    for index, item in enumerate(partition):
        for nodeID in item:
            labels[nodeID] = r'$' + str(nodeID) + '$'
            cluster[nodeID] = index
    colors = ['r', 'g', 'b', 'y', 'm']
    shapes = ['v', 'D', 'o', '^', '<']
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=colors[index],
                               node_shape=shapes[index],
                               node_size=350,
                               alpha=1)
    edges = {len(partition): []}
    for link in G.edges():
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)
    for index, edgelist in enumerate(edges.values()):
        if index < len(partition):
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color=colors[index])
        else:
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=3, alpha=0.8, edge_color=colors[index])
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    plt.axis('off')
    plt.show()


def cal_Q(partition, G):
    m = len(G.edges(None, False))
    a = []
    e = []
    for community in partition:
        t = 0.0
        for node in community:
            t += len([x for x in G.neighbors(node)])
        a.append(t / (2 * m))
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if G.has_edge(community[i], community[j]):
                    t += 1.0
        e.append(t / (2 * m))
    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q


class Graph_from_txt:
    def __init__(self):
        self.graph = nx.Graph()

    def createGraph(self, filename):
        file = open(filename, 'r')
        for line in file.readlines():
            nodes = line.split()
            edge = (int(nodes[0]), int(nodes[1]))
            self.graph.add_edge(*edge)
        return self.graph

def calculate_pairwise_weight_numpy(
    feature1: np.ndarray,
    feature2: np.ndarray,
    method: str = "gaussian",
    sigma: Optional[float] = None
) -> float:
    """
    Calculate the weight between two points based on their feature vectors using NumPy.

    Parameters:
    ----------
    feature1 : np.ndarray
        The feature vector of the first point. Shape: (num_features,).
    feature2 : np.ndarray
        The feature vector of the second point. Shape: (num_features,).
    method : str
        The method for computing the weight. Options are:
        - "cosine" : Cosine similarity.
        - "euclidean" : Inverse of Euclidean distance.
        - "dot" : Dot product of feature vectors.
        - "gaussian" : Gaussian kernel function.
    sigma : float, optional
        The parameter for the Gaussian kernel. Required if `method="gaussian"`.

    Returns:
    -------
    weight : float
        The computed weight between the two points.
    """
    # Ensure feature vectors are 1D
    if feature1.ndim != 1 or feature2.ndim != 1:
        raise ValueError("Both feature1 and feature2 must be 1D arrays.")

    # Cosine similarity
    if method == "cosine":
        norm1 = np.linalg.norm(feature1) + 1e-8
        norm2 = np.linalg.norm(feature2) + 1e-8
        weight = np.dot(feature1, feature2) / (norm1 * norm2)

    # Inverse Euclidean distance
    elif method == "euclidean":
        distance = np.linalg.norm(feature1 - feature2) + 1e-8
        weight = 1 / distance

    # Dot product
    elif method == "dot":
        weight = np.dot(feature1, feature2)

    # Gaussian kernel
    elif method == "gaussian":
        if sigma is None:
            raise ValueError("Parameter 'sigma' must be provided for Gaussian kernel.")
        distance_squared = np.sum((feature1 - feature2) ** 2)
        weight = np.exp(-distance_squared / (2 * sigma ** 2))

    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods are 'cosine', 'euclidean', 'dot', 'gaussian'.")

    return weight


def flatten_communities_to_labels(communities):
    num_nodes = sum([len(i) for i in communities])
    labels = np.zeros(num_nodes, dtype=int)
    for i, community in enumerate(communities):
        for node in community:
            labels[node] = i
    return labels


def cal_modularity_of_weighed_graph(partition, G:nx.Graph):
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

if __name__ == '__main__':
    G = load_graph('../data/club.txt')
    obj = Graph_from_txt()
    G1 = obj.createGraph("../data/club.txt")
    pos = nx.spring_layout(G1)
    start_time = time.time()
    algorithm = Louvain(G)
    communities = algorithm.execute()
    end_time = time.time()
    communities = sorted(communities, key=lambda b: -len(b))
    count = 0
    for communitie in communities:
        count += 1
        print("Community", count, " ", communitie)

    print("Modularity Q:", cal_Q(communities, G1))
    print(f'Execution Time: {end_time - start_time}')
    showCommunity(G1, communities, pos)
