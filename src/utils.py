import numpy as np
from networkx.algorithms.cuts import cut_size, volume


def get_conductance(G, pred_cluster):
    def conductance(G, S, T=None, weight=None):
        if T is None:
            T = set(G) - set(S)
        num_cut_edges = cut_size(G, S, T, weight=weight)
        volume_S = volume(G, S, weight=weight)
        volume_T = volume(G, T, weight=weight)
        return num_cut_edges / max(min(volume_S, volume_T), 1)

    return conductance(G, pred_cluster)


def get_k_means_error(X, pred_cluster):
    mean = np.mean(X[pred_cluster], axis=0)
    return np.sqrt(sum(np.linalg.norm(X[u] - mean) for u in pred_cluster) / len(pred_cluster))
