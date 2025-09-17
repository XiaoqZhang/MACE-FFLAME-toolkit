"""
fflame.sample.selection
-----------------------

This module provides utilities for selecting representative MOF structures
from a dataset using K-Means clustering and sort the structures for 
efficient DFT calculations.

Main features
-------------
- Compute SOAP descriptors for MOFs.
- Cluster structures with KMeans to select representative subsets.
- (Optional) Order the selected structures using a graph-based
  minimum spanning tree (MST) and depth-first traversal.

Example
-------
>>> from ase.io import read
>>> from fflame.sample.selection import selector
>>>
>>> frames = [read("MOF1.cif"), read("MOF2.cif"), read("MOF3.cif")]
>>> selected_frames, indices = selector(frames, number=2, sort=True)
>>> print("Selected indices:", indices)
"""

from typing import List, Tuple, Union
from dscribe.descriptors import SOAP
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def kmeans_selector(features: np.ndarray, num_select: int, seed: int) -> List[int]:
    """
    Select representative samples using KMeans clustering.

    Parameters
    ----------
    features : ndarray of shape (n_samples, n_features)
        Feature matrix of all MOFs.
    num_select : int
        Number of clusters (and thus samples) to select.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    selected : list of int
        Indices of selected samples (one per cluster).
    """
    print("Clustering ...")
    kmeans = KMeans(n_clusters=num_select, n_init='auto', random_state=seed).fit(features)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    selected = []
    print("Locating the center point for each cluster: \n")
    for i in tqdm(range(num_select)):
        cluster_idx = np.where(labels==i)[0]
        center = centroids[i]
        closest = min(cluster_idx, key=lambda idx: np.linalg.norm(features[idx] - center))
        selected.append(closest)
    
    return selected

def selector(
    frames: List[Atoms],
    number: int,
    sort: bool,
    species: List[str] = ["H", "C", "N", "O"],
    seed: int = 1314,
) -> Tuple[List[Atoms], List[int]]:
    """
    Select representative MOF structures from a dataset.

    Parameters
    ----------
    frames : list of ase.Atoms
        Input MOF structures.
    number : int
        Number of representative structures to select.
    sort : bool
        Whether to order selected structures using MST + DFS traversal.
    species : list of str, optional
        Atomic species included in SOAP descriptors (default: ["H", "C", "N", "O"]).
    seed : int, optional
        Random seed for KMeans clustering (default: 1314).

    Returns
    -------
    selected_frames : list of ase.Atoms
        Selected representative MOF structures.
    selected_indices : list of int
        Indices of the selected structures in the original list.
    """
    print("Featurizing MOFs ...")
    desc = SOAP(species=species, r_cut=5.0, n_max=5, l_max=5, sigma=0.2, periodic=True, compression={"mode":"mu2"}, sparse=False)
    all_feats = [desc.create(f) for f in frames]
    flt_feats = np.array([np.mean(x, axis=0) for x in all_feats])
    print(flt_feats.shape)
    selected = kmeans_selector(flt_feats, number, seed)
    
    selected_frames = [frames[i] for i in selected]
    
    if not sort:
        return selected_frames, selected
    else:
        G, _ = build_graph(flt_feats[selected])
        order = mst_dfs_tsp(flt_feats[selected])
        ordered_frames = [selected_frames[i] for i in order]
        ordered_selected = [selected[i] for i in order]
        return ordered_frames, ordered_selected


def build_graph(points: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
    """
    Build a fully connected weighted graph from feature points.

    Parameters
    ----------
    points : ndarray of shape (n_points, n_features)
        Feature vectors.

    Returns
    -------
    G : networkx.Graph
        Graph with edge weights equal to pairwise distances.
    dist_matrix : ndarray
        Pairwise distance matrix.
    """
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    G = nx.Graph()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    return G, dist_matrix

def mst_dfs_tsp(points: np.ndarray, start_node: int = 0) -> List[int]:
    """
    Generate an ordering of points using MST + DFS heuristic for TSP.

    Parameters
    ----------
    points : ndarray of shape (n_points, n_features)
        Feature vectors.
    start_node : int, optional
        Starting node for DFS traversal (default: 0).

    Returns
    -------
    visited : list of int
        Order of nodes visited during DFS traversal.
    """
    G, _ = build_graph(points)
    T = nx.minimum_spanning_tree(G, weight='weight')
    visited = list(nx.dfs_preorder_nodes(T, source=start_node))
    return visited
