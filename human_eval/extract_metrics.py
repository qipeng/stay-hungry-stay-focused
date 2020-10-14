"""
Extract metrics predicted by the teacher model from the output of the prediction run.
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import numpy as np
import os
import sys

EPSILON = 2e-2

def read_metrics(filename):
    res = []
    with open(filename) as f:
        for line in f:
            if not line.startswith('Metrics:'):
                continue
            line = line.strip().split()
            res.append([float(x) for x in line[1:]])
    return res

def main(args):
    sys1 = read_metrics(args.system1_metrics_file)
    sys2 = read_metrics(args.system2_metrics_file)

    ids = []

    with open(args.record_file) as f:
        for line in f:
            line = line.rstrip().split(',')
            ids.append(int(line[0]))

    bn1 = os.path.basename(args.system1_dump_file)
    bn2 = os.path.basename(args.system2_dump_file)
    metrics = {'human': [x[0] for x in sys1], bn1: [x[1] for x in sys1], bn2: [x[1] for x in sys2]}

    print({k: len(metrics[k]) for k in metrics})
    keys = list(sorted(metrics.keys()))
    print(keys)

    for i in ids:
        for i1 in range(len(keys) - 1):
            for i2 in range(i1+1, len(keys)):
                k1 = keys[i1]
                k2 = keys[i2]

                diff = metrics[k1][i] - metrics[k2][i]
                #if np.abs(diff) < EPSILON:
                #    diff = 0
                #print(np.sign(diff))
                print(diff)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('record_file', type=str)
    parser.add_argument('system1_metrics_file', type=str)
    parser.add_argument('system2_metrics_file', type=str)
    parser.add_argument('system1_dump_file', type=str)
    parser.add_argument('system2_dump_file', type=str)

    args = parser.parse_args()

    main(args)
