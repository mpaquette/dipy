from __future__ import division, print_function

import numpy as np
import nibabel as nib

from dipy.reconst.peaks import peaks_from_model, peak_directions

from dipy.core.ndindex import ndindex


def pfm(model, data, mask, sphere, parallel=True, min_angle=25.0, relative_peak_th=0.1, sh_order=8):

    peaks = peaks_from_model(model=model,
                             data=data,
                             mask=mask,
                             sphere=sphere,
                             relative_peak_threshold=relative_peak_th,
                             min_separation_angle=min_angle,
                             return_odf=True,
                             return_sh=True,
                             normalize_peaks=False,
                             sh_order=sh_order,
                             sh_basis_type='mrtrix',
                             npeaks=3,
                             parallel=parallel)
    return peaks


def peak_extract(new_peaks, odf, mask, sphere, npeaks=3, normalize_peaks=True,
                 relative_peak_th=0.35, min_angle=15):

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