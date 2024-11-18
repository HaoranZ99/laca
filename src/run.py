import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import localgraphclustering as lgc
import numpy as np
from scipy.sparse import csr_matrix
from data import load_data, load_seeds
from utils import get_conductance, get_k_means_error


def run_laca(args, v, csize, G):
    A, X, D = G

    n = A.shape[0]
    x = csr_matrix(([1.0], ([0], [v])), shape=(1, n))  # Init a one-hot vector.

    # 1. Estimate PPR pi.
    x = adaptive_propagation(args, n, A, D, args.e, x)

    # 2. Compute phi.
    indices = x.nonzero()[1]
    x_tmp = x.dot(X)

    x = x.tolil()
    x[:, indices] = x_tmp.dot(X[indices, :].T)
    x = x.tocsr()
    x.eliminate_zeros()

    x = x.multiply(D)

    # 3. Estimate BDD rho.
    x = adaptive_propagation(args, n, A, D, args.e * x.sum(), x)
    x = x.multiply(1.0 / D)

    arr = x.todense()
    idx = (-arr).argsort()[:csize]

    pred_cluster = set(idx.tolist()[0][:csize])
    return pred_cluster


def adaptive_propagation(args, n, A, D, epsilon, r):
    p = np.zeros(n)
    Dinv = 1.0 / D
    overall_cost = 0
    cost_bound = abs(r).sum() / ((1 - args.alpha) * epsilon)

    while True:
        r_hat = r.copy()
        mask_data = (r_hat.multiply(Dinv)).data
        mask = mask_data < epsilon
        r_hat.data[mask] = 0
        r_hat.eliminate_zeros()
        indices = r.nonzero()[1]
        overall_cost += np.sum(D[indices])
        if r_hat.nnz > args.sigma * r.nnz and overall_cost < cost_bound:
            p += (1 - args.alpha) * r
            r = args.alpha * r.dot(A)
        elif r_hat.nnz != 0:
            mask_above = mask_data >= epsilon
            r.data[mask_above] = 0
            r.eliminate_zeros()
            p += (1 - args.alpha) * r_hat
            r_hat = r_hat.dot(A)
            r += args.alpha * r_hat
        else:
            break
    p = csr_matrix(p)
    return p


def run_algo(args, v, csize, G):
    if args.algo in ["laca_c", "laca_e"]:
        pred_cluster = run_laca(args, v, csize, G)
    else:
        sys.exit("Unknown Algorithm Name!")

    return pred_cluster


if __name__ == "__main__":
    parser = ArgumentParser("Our",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler="resolve")

    parser.add_argument("--data", default="cora", type=str, help="Dataset name.")
    parser.add_argument("--folder", default="./attr_graphs/", type=str, help="Path to data folder.")
    parser.add_argument("--algo", default="laca_c", type=str, help="Algorithm name.")
    parser.add_argument("--batch", default=5, type=int, help="# batches.")
    parser.add_argument("--k", default=100, type=int, help="# seeds per batch.")
    parser.add_argument("--dim", default=-1, type=int, help="Reduced dimension.")
    parser.add_argument("--alpha", default=0.9, type=float, help="Decay factor.")
    parser.add_argument("--e", default=1e-6, type=float, help="Error threshold.")
    parser.add_argument("--sigma", default=0.0, type=float, help="Proportion of greedy operations.")
    parser.add_argument("--examine", action="store_true", help="Whether to examine conductance and WCSS.")

    args = parser.parse_args()

    print(args)

    map_node_label, map_label_node, G, G_nx, X = load_data(args)

    overall_avg_time = 0
    overall_avg_precision = 0
    overall_avg_conductance = 0
    overall_avg_attribute_distance = 0
    for num_batch in range(args.batch):
        avg_precision = 0
        avg_time = 0
        avg_conductance = 0
        avg_attribute_distance = 0
        batch_nodes = load_seeds(args, num_batch, map_node_label, map_label_node)
        running_time = 0
        for (v, cluster) in list(batch_nodes.items())[0:args.k]:
            csize = len(cluster)
            stime = time.time()
            pred_cluster = run_algo(args, v, csize, G)
            etime = time.time()
            running_time += etime - stime

            precision = len(cluster.intersection(pred_cluster)) * 1.0 / csize
            avg_precision += precision
            if args.examine:
                conduct = get_conductance(G_nx, pred_cluster)
                attribute_distance = get_k_means_error(X, list(pred_cluster))
                avg_conductance += conduct
                avg_attribute_distance += attribute_distance

        avg_time = running_time / args.k
        avg_precision /= args.k
        avg_conductance /= args.k
        avg_attribute_distance /= args.k
        overall_avg_precision += avg_precision
        overall_avg_time += avg_time
        overall_avg_conductance += avg_conductance
        overall_avg_attribute_distance += avg_attribute_distance

        print("Batch: %d, Avg Precision: %f, Avg Time: %f." % (num_batch, avg_precision, avg_time))

    overall_avg_precision = overall_avg_precision / args.batch
    print("Overall Avg Precision: %f." % (overall_avg_precision))
    overall_avg_time = overall_avg_time / args.batch
    print("Overall Avg Query Time: %f." % (overall_avg_time))
    if args.examine:
        overall_avg_conductance = overall_avg_conductance / args.batch
        print("Overall Avg Conductance : %f." % (overall_avg_conductance))
        overall_avg_attribute_distance = overall_avg_attribute_distance / args.batch
        print("Overall Avg Attribute Distance : %f." % (overall_avg_attribute_distance))
