from __future__ import division, print_function

import numpy as np
import nibabel as nib

from itertools import product
# Example using the CSD to process the data

from sparc_dmri.load_data import get_data, get_mask
from sparc_dmri.output import compute_npeaks_and_angles, screenshot_odf, screenshot_peaks, get_crop_peaks, signal_from_peaks

from dipy.reconst.csdeconv import auto_response, ConstrainedSphericalDeconvModel
from dipy.data import get_sphere
from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization
from dipy.reconst.dti import TensorModel

# Load the dataset
ndir = 60
shell = None
denoised = None

for ndir, shell, denoised in product([20], [2000], [None, 'nlm', 'nlsam']):
    print(ndir, shell, denoised)
    if shell is None:
        filename = 'CSD_' + str(ndir) + '_dirs_multi_' + str(denoised) + '_'
    else:
        filename = 'CSD_' + str(ndir) + '_dirs_' + str(shell) + '_' + str(denoised) + '_'

    data, affine, gtab = get_data(ndir, shell=shell, denoised=denoised)
    mask = get_mask()
    response, ratio = auto_response(gtab, data, roi_radius=3, fa_thr=0.9)

    if ndir == 60:
        sh_order = 8
    elif ndir == 30:
        sh_order = 6
    else:
        sh_order = 4

    print("SH order is", sh_order)

    # Compute tensors
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    sphere = get_sphere('symmetric724').subdivide()

    # Compute CSD
    S0 = response[1]
    for (a, b) in [(30., 2.)]: #(15., 4.), (19., 2.), (25., 2.), (25., 1.), (25, 2.5), (21., 2.), (19., 2.), (17., 3.), (19., 5.), (14., 2.), (17., 2.)]:

        response = (1e-4 * np.array([a, b, b]), S0)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        peaks_csd = peaks_from_model(model=csd_model,
                                     data=data,
                                     sphere=sphere,
                                     mask=mask,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     parallel=True,
                                     npeaks=3,
                                     return_sh=True,
                                     normalize_peaks=True,
                                     return_odf=True,
                                     sh_order=sh_order)

        nfib = np.sum(np.sum(np.abs(peaks_csd.peak_dirs), axis=-1) > 0, axis=-1).ravel()
        print("1 fiber", np.sum(nfib==1), "2 fibers", np.sum(nfib==2), "3 fibers", np.sum(nfib==3))
        screenshot_odf(peaks_csd.odf, sphere, filename + str(a) + "_" + str(b) + "_odf.png")
        screenshot_peaks(peaks_csd.peak_dirs, filename + str(a) + "_" + str(b) + "_peaks.png", peaks_csd.peak_values)

    # Save everything
    nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'), affine), filename + 'fodf_CSD.nii.gz')
    nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd), affine), filename + 'peaks_CSD.nii.gz')
    nib.save(nib.Nifti1Image(peaks_csd.peak_indices, affine), filename + 'fodf_CSD_peak_indices.nii.gz')
    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), affine), filename + 'fa.nii.gz')

    # Take a screenshot
    #screenshot_odf(peaks_csd.odf, sphere, filename + "odf.png")

    # Get the number of fiber and angles in the required format
    angles = compute_npeaks_and_angles(peaks_csd.peak_dirs)

    np.savetxt(filename + 'angles.txt', angles[:1], fmt="%i")
    f_handle = file(filename + 'angles.txt', 'a')

    np.savetxt(f_handle, angles[1:], fmt="%.4f")
    f_handle.close()

    # Get the signal estimation
    indices = np.where(angles[0].reshape(13, 16) == 1)
    lambdas = tenfit.evals[indices][:, :2]
    l01 = np.mean(lambdas, axis=0).squeeze()
    evals = np.array([l01[0], l01[1], l01[1]])

    # Restore original B0
    data_clean, _, _ = get_data(ndir, shell=shell, denoised=None)
    data[..., 0] = data_clean[..., 0]

    signal = signal_from_peaks(data, peaks_csd.peak_dirs, peaks_csd.peak_values, evals, shell)
    np.savetxt(filename + 'signal.txt', signal, fmt="%.4f")
