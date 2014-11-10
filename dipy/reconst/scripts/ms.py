from __future__ import division, print_function

import numpy as np
import nibabel as nib

from numpy import indices


from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel)
from dipy.reconst.peaks import PeaksAndMetrics
from dipy.core.gradients import gradient_table

from dipy.reconst.scripts.peak import pfm, peak_extract


def max_abs(shm_coeff):
    ind = np.argmax(np.abs(shm_coeff), axis=0)
    x, y, z, w = indices(shm_coeff.shape[1:])
    new_shm_coeff = shm_coeff[(ind, x, y, z, w)]
    return new_shm_coeff


def csd_ms(gtab, data, aff, mask, response, sphere, min_angle=15.0,
           relative_peak_th=0.35):

    gd1, gd2, gd3, gd4, gd5 = prepare_data_for_multi_shell(gtab, data)

    coeffs = []
    Bs = []

    Ns = np.array([gd1[0].gradients.shape[0], gd2[0].gradients.shape[0],
                   gd3[0].gradients.shape[0], gd4[0].gradients.shape[0], gd5[0].gradients.shape[0]])
    N = np.min(Ns[Ns > 0])

    if N >= 45:
        sh_order = 8
    elif N >= 28 and N < 45:
        sh_order = 6
    elif N >= 15 and N < 28:
        sh_order = 4
    else:
        return False

    for gd in [gd1, gd2, gd3, gd4, gd5]:
        if gd[0].gradients.shape[0] > 0:
            model = ConstrainedSphericalDeconvModel(gd[0], response, sh_order=sh_order)

            peaks = pfm(model,
                        gd[1],
                        mask,
                        sphere,
                        min_angle=min_angle,
                        relative_peak_th=relative_peak_th,
                        parallel=False, #parallel=False,
                        sh_order=sh_order)

            coeffs.append(peaks.shm_coeff)
            Bs.append(peaks.B)

    coeffs3 = np.array(coeffs)
    best_coeffs = max_abs(coeffs3)

    odf = np.dot(best_coeffs, peaks.B)

    new_peaks = PeaksAndMetrics()
    new_peaks.B = peaks.B
    new_peaks.shm_coeff = best_coeffs
    new_peaks.odf = odf

    # Custom peaks extraction
    new_peaks = peak_extract(new_peaks, odf, mask, sphere, npeaks=5, normalize_peaks=True,
                             relative_peak_th=relative_peak_th, min_angle=min_angle)

    return new_peaks


def sdt_ms(gtab, data, aff, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1):

    gd1, gd2, gd3, gd4, gd5 = prepare_data_for_multi_shell(gtab, data)

    coeffs = []
    Bs = []

    Ns = np.array([gd1[0].gradients.shape[0], gd2[0].gradients.shape[0],
                   gd3[0].gradients.shape[0], gd4[0].gradients.shape[0], gd5[0].gradients.shape[0]])
    N = np.min(Ns[Ns > 0])

    if N >= 45:
        sh_order = 8
    elif N >= 28 and N < 45:
        sh_order = 6
    elif N >= 15 and N < 28:
        sh_order = 4
    else:
        return False

    for gd in [gd1, gd2, gd3, gd4, gd5]:
        if gd[0].gradients.shape[0] > 0:
            model = ConstrainedSDTModel(gd[0], ratio, sh_order=sh_order)
            peaks = pfm(model, gd[1], mask,
                        sphere,
                        min_angle=min_angle,
                        relative_peak_th=relative_peak_th,
                        parallel=False, #parallel=False,
                        sh_order=sh_order)

            coeffs.append(peaks.shm_coeff)
            Bs.append(peaks.B)

    coeffs3 = np.array(coeffs)
    best_coeffs = max_abs(coeffs3)

    odf = np.dot(best_coeffs, peaks.B)#.squeeze()
    new_peaks = PeaksAndMetrics()
    new_peaks.B = peaks.B
    new_peaks.shm_coeff = best_coeffs
    new_peaks.odf = odf

    # Custom peaks extraction
    new_peaks = peak_extract(new_peaks, odf, mask, sphere, npeaks=5, normalize_peaks=True,
                             relative_peak_th=relative_peak_th, min_angle=min_angle)

    return new_peaks


def prepare_data_for_multi_shell(gtab, data):
    ind500 = ((gtab.bvals < 600) & (gtab.bvals > 400))
    ind1000 = ((gtab.bvals < 1100) & (gtab.bvals > 900))
    ind2000 = ((gtab.bvals < 2100) & (gtab.bvals > 1900))
    ind3000 = ((gtab.bvals < 3100) & (gtab.bvals > 2900))
    ind4000 = ((gtab.bvals < 4100) & (gtab.bvals > 3900))

    S500 = data[..., ind500]
    S1000 = data[..., ind1000]
    S2000 = data[..., ind2000]
    S3000 = data[..., ind3000]
    S4000 = data[..., ind4000]

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    gtab500 = gradient_table(bvals[ind500], bvecs[ind500, :], b0_threshold=10)
    gtab1000 = gradient_table(bvals[ind1000], bvecs[ind1000, :], b0_threshold=10)
    gtab2000 = gradient_table(bvals[ind2000], bvecs[ind2000, :], b0_threshold=10)
    gtab3000 = gradient_table(bvals[ind3000], bvecs[ind3000, :], b0_threshold=10)
    gtab4000 = gradient_table(bvals[ind4000], bvecs[ind4000, :], b0_threshold=10)

    return (gtab500, S500), (gtab1000, S1000), (gtab2000, S2000), (gtab3000, S3000), (gtab4000, S4000)