from __future__ import division, print_function

import numpy as np
import nibabel as nib

from numpy import indices
from itertools import product

# Example using the CSD to process the data

from sparc_dmri.load_data import get_data, get_mask
from sparc_dmri.output import (compute_npeaks_and_angles,
                               screenshot_odf, screenshot_peaks,
                               signal_from_peaks)

from dipy.reconst.csdeconv import auto_response, ConstrainedSphericalDeconvModel, ConstrainedSDTModel
from dipy.data import get_sphere
from dipy.reconst.peaks import (peaks_from_model, reshape_peaks_for_visualization, PeaksAndMetrics,
                                peak_directions)
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
from dipy.core.ndindex import ndindex


def prepare_data_for_multi_shell(gtab, data):
    ind1000 = (gtab.bvals < 10) | ((gtab.bvals < 1100) & (gtab.bvals > 900))
    ind2000 = (gtab.bvals < 10) | ((gtab.bvals < 2100) & (gtab.bvals > 1900))
    ind3000 = (gtab.bvals < 10) | ((gtab.bvals < 3100) & (gtab.bvals > 2900))

    S1000 = data[..., ind1000]
    S2000 = data[..., ind2000]
    S3000 = data[..., ind3000]

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    gtab1000 = gradient_table(bvals[ind1000], bvecs[ind1000, :], b0_threshold=10)
    gtab2000 = gradient_table(bvals[ind2000], bvecs[ind2000, :], b0_threshold=10)
    gtab3000 = gradient_table(bvals[ind3000], bvecs[ind3000, :], b0_threshold=10)

    return (gtab1000, S1000), (gtab2000, S2000), (gtab3000, S3000)

def pfm(model, data, mask, sphere, parallel=True, min_angle=25.0, relative_peak_th=0.1, sh_order=8):

    peaks = peaks_from_model(model=model,
                             data=data,
                             mask=mask,
                             sphere=sphere,
                             relative_peak_threshold=relative_peak_th,
                             min_separation_angle=min_angle,
                             return_odf=False,
                             return_sh=True,
                             normalize_peaks=False,
                             sh_order=sh_order,
                             sh_basis_type='mrtrix',
                             npeaks=3,
                             parallel=parallel)
    return peaks

def max_abs(shm_coeff):
    ind = np.argmax(np.abs(shm_coeff), axis=0)
    x, y, z, w = indices(shm_coeff.shape[1:])
    new_shm_coeff = shm_coeff[(ind, x, y, z, w)]
    return new_shm_coeff

def csd_ms(gtab, data, affine, mask, response, sphere, min_angle=15.0,
           relative_peak_th=0.35, sh_order=8):

    gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data)

    coeffs = []
    Bs = []

    for gd in [gd1, gd2, gd3]:
        model = ConstrainedSphericalDeconvModel(gd[0], response, sh_order=sh_order)

        peaks = pfm(model,
                    gd[1],
                    mask,
                    sphere,
                    min_angle=min_angle,
                    relative_peak_th=relative_peak_th,
                    parallel=True, #parallel=False,
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
    new_peaks = peak_extract(new_peaks, odf, mask, npeaks=3, normalize_peaks=True,
                             relative_peak_th=relative_peak_th, min_angle=min_angle)

    return new_peaks


def peak_extract(new_peaks, odf, mask, npeaks=3, normalize_peaks=True, relative_peak_th=0.35, min_angle=15):

    global_max = -np.inf

    peak_dirs = np.zeros(odf.shape[:3] + (npeaks, 3), dtype=np.float64)
    peak_values = np.zeros(odf.shape[:3] + (npeaks,), dtype=np.float64)

    for idx in ndindex(odf.shape[:3]):

        if not mask[idx]:
            continue

        # Get peaks of odf
        direction, pk, ind = peak_directions(odf[idx], sphere, relative_peak_th, min_angle)

        # Calculate peak metrics
        if pk.shape[0] != 0:
            global_max = max(global_max, pk[0])

            n = min(npeaks, pk.shape[0])
           # qa_array[idx][:n] = pk[:n] - odf.min()

            peak_dirs[idx][:n] = direction[:n]
         #   peak_indices[idx][:n] = ind[:n]
            peak_values[idx][:n] = pk[:n]

            if normalize_peaks:
                peak_values[idx][:n] /= pk[0]
                peak_dirs[idx] *= peak_values[idx][:, None]

        new_peaks.peak_dirs = peak_dirs#[:, :, None, :, :]
        new_peaks.peak_values = peak_values#[:, :, None, :]

    return new_peaks


def sdt_ms(gtab, data, affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1, sh_order=8):

    gd1, gd2, gd3 = prepare_data_for_multi_shell(gtab, data)

    coeffs = []
    Bs = []
    for gd in [gd1, gd2, gd3]:
        model = ConstrainedSDTModel(gd[0], ratio, sh_order=sh_order)
        peaks = pfm(model, gd[1], mask,
                    sphere,
                    min_angle=min_angle,
                    relative_peak_th=relative_peak_th,
                    parallel=True, #parallel=False,
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
    new_peaks = peak_extract(new_peaks, odf, mask, npeaks=3, normalize_peaks=True,
                             relative_peak_th=relative_peak_th, min_angle=min_angle)

    return new_peaks


# Load the dataset
ndir = 60
shell = None
denoised = None

for ndir, shell, denoised in product([20], [None], [None, 'nlm', 'nlsam']):
    print(ndir, shell, denoised)

    filename = 'grad_' + str(ndir) + '_dirs_' + str(shell) + '_' + str(denoised) + '_'

    data, affine, gtab = get_data(ndir, shell=shell, denoised=denoised)
    mask = get_mask()
    mask = mask[:, :, None]

    response, ratio = auto_response(gtab, data, roi_radius=3, fa_thr=0.8)

    if ndir == 60:
        sh_order = 8
    elif ndir == 30:
        sh_order = 6
    else:
        sh_order = 4

    # Compute tensors
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    sphere = get_sphere('symmetric724').subdivide()

    # Compute CSD
    S0 = response[1]
    print("Mean S0 is: ", S0)


    ## CSD part
    (a, b) = (30, 2)
    response = (1e-4 * np.array([a, b, b]), S0)
    peaks_csd = csd_ms(gtab, data, affine, mask, response, sphere, min_angle=25.0, relative_peak_th=0.35,  sh_order=sh_order)
    screenshot_odf(peaks_csd.odf, sphere, filename + str(a) + "_" + str(b) + "_msCSDodf.png")

    print(peaks_csd.peak_dirs.shape, peaks_csd.peak_values.shape)
    screenshot_peaks(peaks_csd.peak_dirs, filename + str(a) + "_" + str(b) + "_msCSDpeaks.png", peaks_csd.peak_values)

    # Get the number of fiber and angles in the required format
    angles_csd = compute_npeaks_and_angles(peaks_csd.peak_dirs)

    np.savetxt(filename + 'CSD_angles.txt', angles_csd[:1], fmt="%i")
    f_handle = file(filename + 'CSD_angles.txt', 'a')

    np.savetxt(f_handle, angles_csd[1:], fmt="%.4f")
    f_handle.close()

    # Get the signal estimation
    indices_evals = np.where(angles_csd[0].reshape(13, 16) == 1)
    lambdas = tenfit.evals[indices_evals][:, :2]
    l01 = np.mean(lambdas, axis=0).squeeze()
    evals = np.array([l01[0], l01[1], l01[1]])

    # Restore original B0
    data_clean, _, _ = get_data(ndir, shell=shell, denoised=None)
    data[..., 0] = data_clean[..., 0]

    signal = signal_from_peaks(data, peaks_csd.peak_dirs, peaks_csd.peak_values, evals, shell)
    np.savetxt(filename + 'CSD_signal.txt', signal, fmt="%.4f")


    ## SDT part
    response = (1e-4 * np.array([a, b, b]), S0)
    ratio = b/a
    peaks_sdt = sdt_ms(gtab, data, affine, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1, sh_order=sh_order)

    screenshot_odf(peaks_sdt.odf, sphere, filename + str(a) + "_" + str(b) + "_msSDTodf.png")
    screenshot_peaks(peaks_sdt.peak_dirs, filename + str(a) + "_" + str(b) + "_msSDTpeaks.png", peaks_sdt.peak_values)

    # Get the number of fiber and angles in the required format
    angles_sdt = compute_npeaks_and_angles(peaks_sdt.peak_dirs)

    np.savetxt(filename + 'SDT_angles.txt', angles_sdt[:1], fmt="%i")
    f_handle = file(filename + 'SDT_angles.txt', 'a')

    np.savetxt(f_handle, angles_sdt[1:], fmt="%.4f")
    f_handle.close()

    # Get the signal estimation
    indices_evals = np.where(angles_sdt[0].reshape(13, 16) == 1)
    lambdas = tenfit.evals[indices_evals][:, :2]
    l01 = np.mean(lambdas, axis=0).squeeze()
    evals = np.array([l01[0], l01[1], l01[1]])

    # Restore original B0
    data_clean, _, _ = get_data(ndir, shell=shell, denoised=None)
    data[..., 0] = data_clean[..., 0]

    signal = signal_from_peaks(data, peaks_sdt.peak_dirs, peaks_sdt.peak_values, evals, shell)
    np.savetxt(filename + 'SDT_signal.txt', signal, fmt="%.4f")
