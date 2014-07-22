from __future__ import division, print_function

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_data, get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf, single_tensor, all_tensor_evecs
from dipy.viz import fvtk
from dipy.core.ndindex import ndindex

import numpy as np
import nibabel as nib

# Example using the CSD to process the data

from sparc_dmri.load_data import get_data, get_mask
from sparc_dmri.output import compute_npeaks_and_angles, screenshot_odf

# Load the dataset
filename = "grad_60_dirs_3000"
ndir = 60
shell = 3000
denoised = 'nlm'

data, affine, gtab = get_data(ndir, shell=shell, denoised=denoised)
mask = get_mask()
mask = mask[:, :, None]


def simulation():

    SNR = 20
    S0 = 1

    fdwi, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals) * 3
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (60, 0)]

    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=[50, 50], snr=SNR)

    return S, gtab

#sphere = get_sphere('symmetric362')
sphere = get_sphere('symmetric724')
sphere = sphere.subdivide(1)

def richardson_lucy(data, mask, gtab, sphere):

    S0 = 1

    # Dictionary creation

    num_dir = len(gtab.bvals) - 1
    num_dir_fod = len(sphere.vertices)
    response = (np.array([0.0020, 0.000, 0.000]), S0)
    num_iter = 400

    H = np.zeros((num_dir, num_dir_fod))

    for i in range(num_dir_fod):

        evals = response[0]
        evecs = all_tensor_evecs(sphere.vertices[i])

        signal_tmp = single_tensor(gtab, S0, evals, evecs)

        H[:, i] = signal_tmp[1:]

    #print(H.shape)

    H_t = H.T

    FOD = np.zeros(data.shape[:-1] + (len(sphere.vertices),))

    for index in ndindex(data.shape[:-1]):
        S = data[index]

        if mask[index] > 0:

            H_t_s = np.dot(H_t, S[1:])

            print(H_t_s.shape)

            fod = np.ones(num_dir_fod)/float(num_dir)

            for i in range(num_iter):

                H_f = np.dot(H, fod)

                fod = fod * (H_t_s / np.dot(H_t, H_f))

            FOD[index] = fod

    return FOD

#data, gtab = simulation()
#data = data[None, None, None, :]
#mask = np.ones(data.shape[:-1])

FOD = richardson_lucy(data, mask, gtab, sphere)

ren = fvtk.ren()

fvtk.add(ren, fvtk.sphere_funcs(FOD, sphere))

fvtk.show(ren)

