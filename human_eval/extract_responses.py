"""
Extract human eval survey results from CSV files from Qualtrics, and compare it to system output.
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import numpy as np
import sys

def main(args):
    response = []
    with open(args.response_file) as f:
        for line in f:
            response.append(line.rstrip().split(','))

    idx = 0
    while idx < len(response[0]):
        if response[0][idx].startswith('QID'):
            break
        idx += 1

    responses = defaultdict(list)
    responses_info = defaultdict(list)
    responses_spec = defaultdict(list)

    with open(args.record_file) as f:
        for line in f:
            line = line.strip().split(',')
            for system in line[1:]:
                responses[system].append(int(response[-1][idx]))
                idx += 1
            for system in line[1:]:
                responses_info[system].append(int(response[-1][idx]))
                idx += 1
            for system in line[1:]:
                responses_spec[system].append(int(response[-1][idx]))
                idx += 1

    for print_func in [print_comp, print_raw, print_stats]:
        print('Overall')
        print_func(responses)
        print()
        print('Informativeness')
        print_func(responses_info)
        print()
        print('Specificity')
        print_func(responses_spec)
        print()

def print_raw(responses):
    keys = list(sorted(responses.keys()))
    length = len(responses[keys[0]])
    print(' '.join(keys))
    for i in range(length):
        print(' '.join([str(responses[key][i]) for key in keys]))

def print_comp(responses):
    keys = list(sorted(responses.keys()))
    length = len(responses[keys[0]])
    for i in range(length):
        for j1 in range(len(keys)-1):
            for j2 in range(j1+1, len(keys)):
                print(np.sign(responses[keys[j2]][i] - responses[keys[j1]][i])) # j1 better: > 0, equal: =0, j2 better: < 0


def print_stats(responses):
    for k in responses:
        responses[k] = np.array(responses[k])

    print(". left equal right")
    keys = list(sorted(responses.keys()))
    for i in range(len(keys) - 1):
        k1 = keys[i]
        for j in range(i+1, len(keys)):
            k2 = keys[j]
            print(f"{k1}<=>{k2} {np.sum(responses[k1] < responses[k2])} {np.sum(responses[k1] == responses[k2])} {np.sum(responses[k1] > responses[k2])}")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('record_file', type=str)
    parser.add_argument('response_file', type=str)

    args = parser.parse_args()

    main(args)
