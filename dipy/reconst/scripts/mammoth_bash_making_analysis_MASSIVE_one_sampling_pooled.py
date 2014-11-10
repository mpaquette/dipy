from __future__ import division, print_function

import numpy as np

import itertools

import os


#python /home/paquette/dipy/dipy/reconst/scripts/analysis_MASSIVE_one_sampling.py 
#-data /mnt/parallel_scratch_mp2_wipe_on_april_2015/descotea/paquette/massive/ismrm/simu/data/interpolate/ 
#-grad /mnt/parallel_scratch_mp2_wipe_on_april_2015/descotea/paquette/massive/ismrm/simu/grad/interpolate/ 
#-out /mnt/parallel_scratch_mp2_wipe_on_april_2015/descotea/paquette/massive/ismrm/simu/output/ 
#-tag interpolate -s 2 -n 60 -shells 0 1 1 0 0 -order 0 -it 0 -sigma 0 0.05 -trial 1

def make_name(nS, N, shells, order, it):
    # nS: int, number of shells
    # N: int, number of points
    # shells: list of bool, shell selection
    # order: float, q-weigthing exponant used
    # it: int, ID of specific sampling scheme generation
    bvals = [500, 1000, 2000, 3000, 4000]
    name = 'S-{}_N-{}_b'.format(nS, N)
    for b in [bvals[i] for i, j in enumerate(shells) if j == 1]:
        name += '-{}'.format(b)
    name += '_Ord-{}_it-{}'.format(order, it)
    return name

def main():
    # Ns = [35]
    Ns = [35, 60, 90, 120, 305]

    # Ss = [1]
    Ss = [1, 2, 3, 4, 5]

    list_order = [0, 1, 2]

    its = range(10)

    command = 'python /home/paquette/dipy/dipy/reconst/scripts/analysis_MASSIVE_one_sampling_pooled.py'
    datapath = '/mnt/parallel_scratch_mp2_wipe_on_april_2015/descotea/paquette/massive/ismrm/simu/data/interpolate/'
    gradpath = '/mnt/parallel_scratch_mp2_wipe_on_april_2015/descotea/paquette/massive/ismrm/simu/grad/interpolate/'
    output = '/mnt/parallel_scratch_mp2_wipe_on_april_2015/descotea/paquette/massive/ismrm/simu/output/'
    tag = 'interpolate'
    sigma = [0, 1/30., 1/20., 1/10.]
    trial = 3
    skip = 0



    # Get data from MASSIVE or simulations
    # Example of loop over all samplings
    for S in Ss:
        for N in Ns:
            shell_permu = [list(i) for i in set([i for i in itertools.permutations(S*[1] + (5-S)*[0])])]
            for shells in shell_permu:
                for order in list_order:
                    for it in its:
                        fname = make_name(S, N, shells, order, it)
                        if os.path.exists(gradpath + tag + '_' + fname + '.bvals'):
                            sigma_str = ''
                            for sig in sigma:
                                sigma_str += ' {} '.format(sig)
                            bash_command = command + ' -data ' + datapath + ' -grad ' + gradpath + ' -out ' + output + ' -tag ' + tag + ' -s {} -n {} -shells {} {} {} {} {} -order {} -it {} -sigma {} -trial {} -skip {}'.format(S, N, shells[0], shells[1], shells[2], shells[3], shells[4], order, it, sigma_str, trial, skip)
                            print(bash_command)

if __name__ == "__main__":
    main()
