from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import random

if __name__ == '__main__':
    parser = ArgumentParser("Our",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='default', help='data name.')
    # parser.add_argument('--ratio', default=0.1, type=float, help='#seeds')
    parser.add_argument('--k', default=100, type=int, help='#seeds')

    args = parser.parse_args()

    print(args)

    classes = {}
    nodes = {}
    with open(args.data + "/labels.txt", "r") as fin:
        for line in fin:
            vs = line.split()
            v = vs[0]
            ls = vs[1:]

            v = int(v)
            if len(ls) >= 1:
                nodes[v] = set([])
                for l in ls:
                    l = int(l)
                    if l not in classes:
                        classes[l] = set([])

                    classes[l].add(v)
                    nodes[v].add(l)

    print(classes.keys(), [len(vs) for vs in classes.values()])

    V = nodes.keys()
    n = len(V)

    num_batch = 5
    for i in range(num_batch):
        seeds = random.sample(V, args.k)  # int(args.ratio*n))
        print(i + 1, len(seeds))
        with open(args.data + "/seeds_" + str(i + 1) + ".txt", "w") as fout:
            fout.write("%SeedNode-ID Local_Cluster_Size\n")
            for s in seeds:
                size_clu = 0
                for c in nodes[s]:
                    size_clu += len(classes[c])

                fout.write(str(s) + " " + str(size_clu) + "\n")

    with open(args.data + "/clusters.txt", "w") as fout:
        fout.write("%cluster-ID Node-IDs\n")
        for (k, vs) in sorted(classes.items()):
            fout.write(str(k))
            for v in vs:
                fout.write(" " + str(v))

            fout.write("\n")

