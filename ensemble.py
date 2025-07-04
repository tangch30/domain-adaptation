import math
import numpy as np


class EnsembleTrainer(object):
    def __init__(self):
        pass

    def data_generator(self, mix_rate):
        pass

    def test_ensemble_with_known_mix(self):
        pass

    def train_ensemble_with_hedge(self):
        pass


def find_eps_cover(eps, d):
    # return unnormalized epsilon cover
    N = math.ceil(1/eps)
    all_partitions_list = find_coverings(N, d)
    return all_partitions_list, N


def find_coverings(N_remainder, d_remainder):
    if d_remainder == 1:
        return list(list(N_remainder))
    else:
        new_partitions_list = list()
        for n in range(1, N_remainder+1):
            partitions_list = find_coverings(N_remainder-n, d_remainder-1)
            for partitions in partitions_list:
                partitions.insert(0, n)
                new_partitions_list.append(partitions)

    return new_partitions_list
