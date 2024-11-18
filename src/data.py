import math
import sys
import networkx as nx
import numpy as np
from pyrfm.random_feature import OrthogonalRandomFeature
from scipy.sparse import csr_matrix, load_npz, diags
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd


def load_data(args):
    print("Loading node labels...")
    map_node_label = {}
    labels_file = args.folder + args.data + "/labels.txt"
    with open(labels_file, "r") as fin:
        for line in fin:
            vals = line.split()
            v = vals[0]
            ls = vals[1:]

            v = int(v)
            if len(ls) >= 1:
                map_node_label[v] = set([])
                for l in ls:
                    l = int(l)
                    map_node_label[v].add(l)

    map_label_node = {}
    clusters_file = args.folder + args.data + "/clusters.txt"
    with open(clusters_file, "r") as fin:
        fin.readline()
        for line in fin:
            vals = line.split()
            c = int(vals[0])
            vs = set([int(v) for v in vals[1:]])
            map_label_node[c] = vs

    if args.algo in ["laca_c", "laca_e"]:
        IJ = np.fromfile(args.folder + args.data + "/edgelist.txt", sep=" ").reshape(-1, 2)
        row = IJ[:, 0].astype(np.int32)
        col = IJ[:, 1].astype(np.int32)
        rows = np.hstack([row, col])
        cols = np.hstack([col, row])
        vals = [1] * len(rows)

        A = csr_matrix((vals, (rows, cols)))
        D = np.array(A.sum(axis=0)).reshape(-1)
        A = preprocessing.normalize(A, norm="l1", axis=1)

        Z = construct_TNAM(args)
        G = (A, Z, D)
    else:
        sys.exit("Unknown Algorithm Name!")

    # Load NetworkX Graph and X for computing conductance and WCSS.
    G_nx = nx.read_edgelist(args.folder + args.data + "/edgelist.txt", nodetype=int)
    X = load_npz(args.folder + args.data + "/attrs.npz")
    X = preprocessing.normalize(X, norm="l2", axis=1)
    return map_node_label, map_label_node, G, G_nx, X


def construct_TNAM(args):
    X = load_npz(args.folder + args.data + "/attrs.npz")

    if 0 < args.dim < X.shape[1]:
        U, Sigma, _ = randomized_svd(X, n_components=args.dim, n_iter=10, random_state=42)
        X = csr_matrix(U).dot(diags(Sigma))
    elif args.dim == -1:
        args.dim = X.shape[1]
    X = preprocessing.normalize(X, norm="l2", axis=1)  # Get normalized attribute matrix.
    if args.algo == "laca_e":
        orf = OrthogonalRandomFeature(n_components=2 * args.dim)
        delta = 4
        X = np.sqrt(math.exp(1 / delta)) * csr_matrix(orf.fit_transform(X))
    sum = np.sum(X, axis=0)
    X = X / np.sqrt(np.maximum(X.dot(sum.T), 1e-6))
    Z = csr_matrix(X)
    return Z


def load_seeds(args, num_batch, map_node_label, map_label_node):
    print("Loading seed nodes and ground-truth local clusters...")
    batch_nodes = {}
    with open(args.folder + args.data + "/seeds_" + str(num_batch + 1) + ".txt", "r") as fin:
        fin.readline()
        for line in fin:
            v, csize = line.split()
            v, csize = int(v), int(csize)
            cluster = set([])
            for l in map_node_label[v]:
                vs = map_label_node[l]
                cluster.update(vs)
            batch_nodes[v] = cluster

    return batch_nodes
