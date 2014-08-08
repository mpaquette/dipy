from __future__ import division, print_function

import numpy as np
import nibabel as nib

from numpy import indices

from itertools import product
from dipy.reconst.csdeconv import (auto_response, ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel)
from dipy.reconst.shm import (QballModel, CsaOdfModel, lazy_index)
from dipy.data import get_sphere
from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization, PeaksAndMetrics, peak_directions
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk

import itertools
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.ndindex import ndindex



def make_name(nS, N, shells, order, it):
	# nS: int, number of shells
	# N: int, number of points
	# shells: list of bool, shell selection
	# order: float, q-weigthing exponant used
	# it: int, ID of specific sampling scheme generation
	bvals = [500, 1000, 2000, 3000, 4000]
	name = 'S-{}_N-{}_b'.format(nS,N)
	for b in [bvals[i] for i, j in enumerate(shells) if j == 1]:
		name += '-{}'.format(b)
	name += '_Ord-{}_it-{}'.format(order, it)
	return name

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

def screenshot_odf(odf, sphere, filename, show=False):
    """Takes a screenshot of the odfs, saved as filename.png"""

    if ".png" not in filename:
        filename += '.png'

    ren = fvtk.ren()
    fodf_spheres = fvtk.sphere_funcs(odf, sphere, scale=1.8, norm=True)
    fvtk.add(ren, fodf_spheres)
 #   fvtk.add(ren, fvtk.axes())

    fodf_spheres.RotateZ(90)
    fodf_spheres.RotateX(180)
    fodf_spheres.RotateY(180)

    if show:
        fvtk.show(ren, size=(600, 600))

    fvtk.record(ren, out_path=filename, size=(1000, 1000))
    print('Saved illustration as', filename)


def screenshot_peaks(peaks_dirs, filename, peaks_values=None, show=False):
    """Takes a screenshot of the peaks, saved as filename.png"""

    if ".png" not in filename:
        filename += '.png'

    ren = fvtk.ren()
    fodf_peaks = fvtk.peaks(peaks_dirs, peaks_values, scale=1.8)
    fvtk.add(ren, fodf_peaks)
  #  fvtk.add(ren, fvtk.axes())

    fodf_peaks.RotateZ(90)
    fodf_peaks.RotateX(180)
    fodf_peaks.RotateY(180)

    if show:
        fvtk.show(ren, size=(600, 600))

    fvtk.record(ren, out_path=filename, size=(1000, 1000))
    print('Saved illustration as', filename)

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

    Ns = np.array([gd1[0].gradients.shape[0], gd2[0].gradients.shape[0], gd3[0].gradients.shape[0], gd4[0].gradients.shape[0], gd5[0].gradients.shape[0]])
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
    new_peaks = peak_extract(new_peaks, odf, mask, npeaks=3, normalize_peaks=True,
                             relative_peak_th=relative_peak_th, min_angle=min_angle)

    return new_peaks

def sdt_ms(gtab, data, aff, mask, ratio, sphere, min_angle=25.0, relative_peak_th=0.1):

    gd1, gd2, gd3, gd4, gd5 = prepare_data_for_multi_shell(gtab, data)

    coeffs = []
    Bs = []

    Ns = np.array([gd1[0].gradients.shape[0], gd2[0].gradients.shape[0], gd3[0].gradients.shape[0], gd4[0].gradients.shape[0], gd5[0].gradients.shape[0]])
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
    new_peaks = peak_extract(new_peaks, odf, mask=mask, npeaks=3, normalize_peaks=True,
                             relative_peak_th=relative_peak_th, min_angle=min_angle)

    return new_peaks

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



# Get data from MASSIVE or simulations
# Example of loop over all samplings
for S in [1, 2, 3, 4, 5]:
    for N in [30, 60, 90]:
        shell_permu = [list(i) for i in set([i for i in itertools.permutations(S*[1] + (5-S)*[0])])]
        for shells in shell_permu:
            list_order = [0, 1, 2]
            if S == 1:
                list_order = [0]
            for order in list_order:
                for it in [0]:
                    wd = 'D:/H_schijf/Data/MASSIVE/Processed/Analysis/GradSamplingB0/'
                    fname = make_name(S, N, shells, order, it)
                    bvals, bvecs = read_bvals_bvecs(wd + 'grad/grad_' + fname + '.bval',
                    					 					   wd + 'grad/grad_' + fname + '.bvec')
                    vol = nib.load(wd + 'data/data_' + fname + '.nii.gz')
                    data = vol.get_data()
                    aff  = vol.get_affine()
                    hdr  = vol.get_header()
                    gtab = gradient_table(bvals=bvals, bvecs=bvecs, big_delta=51.6, small_delta=32.8, b0_threshold=0.5)
                    tag = 'img/'
                    mask = np.ones([1, 1, 1], dtype='bool')


                    # Set parameters
                    _where_b0s = lazy_index(gtab.b0s_mask)
                    S0 = np.mean(data[:, :, :, _where_b0s])
                    sphere = get_sphere('symmetric724').subdivide()

                    # Compute DTI LLS
                    tenmodel = TensorModel(gtab, fit_method="LS")
                    tenfit = tenmodel.fit(data)
                    filename = wd + tag + fname + '_DTILLS'
                    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
                    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
                    # Save everything
                    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff), '_evecs.nii.gz')
                    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff), '_evals.nii.gz')
                    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + 'fa.nii.gz')

                    # Compute DTI WLLS
                    tenmodel = TensorModel(gtab, fit_method="WLS")
                    tenfit = tenmodel.fit(data)
                    filename = wd + tag + fname + '_DTIWLLS'
                    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
                    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
                    # Save everything
                    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff), '_evecs.nii.gz')
                    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff), '_evals.nii.gz')
                    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + 'fa.nii.gz')

                    # Compute DTI NLLS
                    tenmodel = TensorModel(gtab, fit_method="NLLS")
                    tenfit = tenmodel.fit(data)
                    filename = wd + tag + fname + '_DTINLLS'
                    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
                    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
                    # Save everything
                    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff), '_evecs.nii.gz')
                    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff), '_evals.nii.gz')
                    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + 'fa.nii.gz')

                    #    # Compute DTI RESTORE
                    #    tenmodel = TensorModel(gtab, fit_method="RT")
                    #    tenfit = tenmodel.fit(data)
                    #    filename = wd + tag + fname + '_DTIRESTORE'
                    #    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
                    #    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
                    #    # Save everything
                    #    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff), '_evecs.nii.gz')
                    #    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff), '_evals.nii.gz')
                    #    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + 'fa.nii.gz')

                    # Compute DTI REKINDLE


                    if S == 1:
                        if N >= 45:
                            CSD_sh_order = 8
                            Qball_sh_order = 8
                            CSA_sh_order = 4
                        elif N >= 28 and N < 45:
                            CSD_sh_order = 6
                            Qball_sh_order = 6
                            CSA_sh_order = 4
                        elif N >= 15 and N < 28:
                            CSD_sh_order = 4
                            Qball_sh_order = 4
                            CSA_sh_order = 4
                        # Compute CSD
                        # Calibrate RF on whole or partial data?
                        response = (np.array([0.0015, 0.0003, 0.0003]), S0)

                        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=CSD_sh_order)
                        peaks_csd = peaks_from_model(model=csd_model,
                                     data=data,
                                     sphere=sphere,
                                     relative_peak_threshold=.1,
                                     min_separation_angle=25,
                                     parallel=False,
                                     npeaks=5,
                                     return_sh=True,
                                     normalize_peaks=False,
                                     return_odf=True,
                                     sh_order=CSD_sh_order)
                        filename = wd + tag + fname + '_CSD'

                        nfib = np.sum(np.sum(np.abs(peaks_csd.peak_dirs), axis=-1) > 0, axis=-1).ravel()
                        print("1 fiber", np.sum(nfib==1), "2 fibers", np.sum(nfib==2), "3 fibers", np.sum(nfib==3))

                        screenshot_odf(peaks_csd.odf, sphere, filename + "_odf.png", show=True)
                        screenshot_peaks(peaks_csd.peak_dirs, filename + "_peaks.png", peaks_csd.peak_values, show=True)
                        # Save everything
                        nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'), aff), filename + 'fodf.nii.gz')
                        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd), aff), filename + 'peaks.nii.gz')
                        nib.save(nib.Nifti1Image(peaks_csd.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')


                        # Compute Qball
                        qball_model = QballModel(gtab, sh_order=Qball_sh_order)
                        peaks_qball = peaks_from_model(model=qball_model,
                                       data=data,
                                       sphere=sphere,
                                       relative_peak_threshold=.1,
                                       min_separation_angle=25,
                                       parallel=False,
                                       npeaks=5,
                                       return_sh=True,
                                       normalize_peaks=False,
                                       return_odf=True,
                                       sh_order=Qball_sh_order)
                        filename = wd + tag + fname + '_Qball'
                        screenshot_odf(peaks_qball.odf, sphere, filename + "_odf.png", show=True)
                        screenshot_peaks(peaks_qball.peak_dirs, filename + "_peaks.png", peaks_qball.peak_values,
                         show=True)
                        # Save everything
                        nib.save(nib.Nifti1Image(peaks_qball.shm_coeff.astype('float32'), aff), filename + 'fodf.nii.gz')
                        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_qball), aff), filename + 'peaks.nii.gz')
                        nib.save(nib.Nifti1Image(peaks_qball.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')


                        # Compute SDT
                        # Calibrate RF on whole or partial data?
                        ratio = 0.21197

                        sdt_model = ConstrainedSDTModel(gtab, ratio=ratio, sh_order=Qball_sh_order)
                        peaks_sdt = peaks_from_model(model=sdt_model,
                                       data=data,
                                       sphere=sphere,
                                       relative_peak_threshold=.1,
                                       min_separation_angle=25,
                                       parallel=False,
                                       npeaks=5,
                                       return_sh=True,
                                       normalize_peaks=False,
                                       return_odf=True,
                                       sh_order=Qball_sh_order)
                        filename = wd + tag + fname + '_SDT'
                        screenshot_odf(peaks_sdt.odf, sphere, filename + "_odf.png", show=True)
                        screenshot_peaks(peaks_sdt.peak_dirs, filename + "_peaks.png", peaks_sdt.peak_values,
                         show=True)
                        # Save everything
                        nib.save(nib.Nifti1Image(peaks_sdt.shm_coeff.astype('float32'), aff), filename + 'fodf.nii.gz')
                        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_sdt), aff), filename + 'peaks.nii.gz')
                        nib.save(nib.Nifti1Image(peaks_sdt.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')


                        # Compute CSA
                        csa_model = CsaOdfModel(gtab, sh_order=CSA_sh_order)
                        peaks_csa = peaks_from_model(model=csa_model,
                                     data=data,
                                     sphere=sphere,
                                     relative_peak_threshold=.1,
                                     min_separation_angle=25,
                                     parallel=False,
                                     npeaks=5,
                                     return_sh=True,
                                     normalize_peaks=False,
                                     return_odf=True,
                                     sh_order=CSA_sh_order)
                        filename = wd + tag + fname + '_CSA'
                        screenshot_odf(peaks_csa.odf, sphere, filename + "_odf.png", show=True)
                        screenshot_peaks(peaks_csa.peak_dirs, filename + "_peaks.png", peaks_csa.peak_values, show=True)
                        # Save everything
                        nib.save(nib.Nifti1Image(peaks_csa.shm_coeff.astype('float32'), aff), filename + 'fodf.nii.gz')
                        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csa), aff), filename + 'peaks.nii.gz')
                        nib.save(nib.Nifti1Image(peaks_csa.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')


                    if S >= 1:
                        # Compute DKI LLS

                        # Compute DKI WLLS

                        # Compute DKI NLLS

                        # Compute DKI RESTORE

                        # Compute DKI REKINDLE


                        # Compute multi shell CSD
                        response = (np.array([0.0015, 0.0003, 0.0003]), S0)
                        peaks_csd = csd_ms(gtab=gtab, data=data, aff=aff, mask=mask, response=response, sphere=sphere, min_angle=25.0, relative_peak_th=0.1)
                        if peaks_csd:
                            filename = wd + tag + fname + '_msCSD'

                            nfib = np.sum(np.sum(np.abs(peaks_csd.peak_dirs), axis=-1) > 0, axis=-1).ravel()
                            print("1 fiber", np.sum(nfib==1), "2 fibers", np.sum(nfib==2), "3 fibers", np.sum(nfib==3))

                            screenshot_odf(peaks_csd.odf, sphere, filename + "_odf.png", show=True)
                            screenshot_peaks(peaks_csd.peak_dirs, filename + "_peaks.png", peaks_csd.peak_values, show=True)
                            # Save everything
                            nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'), aff), filename + 'fodf.nii.gz')
                            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd), aff), filename + 'peaks.nii.gz')
#                            nib.save(nib.Nifti1Image(peaks_csd.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')


                        # Compute multi shell SDT
                        ratio = 0.21197
                        peaks_sdt = sdt_ms(gtab=gtab, data=data, aff=aff, mask=mask, ratio=ratio, sphere=sphere, min_angle=25.0, relative_peak_th=0.1)
                        if peaks_sdt:
                            filename = wd + tag + fname + '_msSDT'

                            nfib = np.sum(np.sum(np.abs(peaks_sdt.peak_dirs), axis=-1) > 0, axis=-1).ravel()
                            print("1 fiber", np.sum(nfib==1), "2 fibers", np.sum(nfib==2), "3 fibers", np.sum(nfib==3))

                            screenshot_odf(peaks_sdt.odf, sphere, filename + "_odf.png", show=True)
                            screenshot_peaks(peaks_sdt.peak_dirs, filename + "_peaks.png", peaks_sdt.peak_values, show=True)
                            # Save everything
                            nib.save(nib.Nifti1Image(peaks_sdt.shm_coeff.astype('float32'), aff), filename + 'fodf.nii.gz')
                            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_sdt), aff), filename + 'peaks.nii.gz')
#                            nib.save(nib.Nifti1Image(peaks_sdt.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')

