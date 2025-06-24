from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def kmeans_selector(features, num_select, seed):
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

def selector(frames, number, sort, species=['H', 'C', 'N', 'O'], seed=1314):
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


def build_graph(points):
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    G = nx.Graph()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    return G, dist_matrix

def mst_dfs_tsp(points, start_node=0):
    G, _ = build_graph(points)
    # 构造最小生成树
    T = nx.minimum_spanning_tree(G, weight='weight')
    # DFS遍历生成树
    visited = list(nx.dfs_preorder_nodes(T, source=start_node))
    return visited
