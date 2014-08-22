from __future__ import division, print_function

import numpy as np
import nibabel as nib

from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel, odf_sh_to_sharp)
from dipy.reconst.shm import (QballModel, CsaOdfModel, lazy_index)
from dipy.data import get_sphere
from dipy.reconst.peaks import reshape_peaks_for_visualization
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
from dipy.reconst.shore import ShoreModel

import itertools
from dipy.io.gradients import read_bvals_bvecs

from dipy.reconst.scripts.ms import csd_ms, sdt_ms
from dipy.reconst.scripts.peak import pfm
from dipy.reconst.scripts.screenshot import screenshot_odf, screenshot_peaks


import argparse

DESCRIPTION = """
    MASSIVE analysis
    """

sphere = get_sphere('symmetric724').subdivide()
show = True


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('input', action='store', metavar='input', type=str,
                   help='Path of the input folder containing folders grad/ data/ and img/.')

    p.add_argument('output', action='store', metavar='output', type=str,
                   help='Path of the output folder.')

#    p.add_argument('-o', action='store', dest='savename',
#                   metavar='savename', required=False, default=None, type=str,
#                   help='Path and prefix for the saved metrics files. The name is always appended \
#                   with _(metric_name).nii.gz, where (metric_name) if the name of the computed metric.')

    return p


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


def decision(N, model):
    if model == 'CSD' or model == 'Qball' or model == 'SDT':
        if N >= 15 and N < 28:
            sh_order = 4
        elif N >= 28 and N < 45:
            sh_order = 6
        elif N >= 45:
            sh_order = 8
    elif model == 'CSA':
        sh_order = 4
    elif model == 'SHORE':
        if N >= 35 and N < 84:
            sh_order = 4
        elif N >= 84 and N < 165:
            sh_order = 6
        elif N >= 165:
            sh_order = 8
        else:
            return False
    return sh_order


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    inputfolder = args.input
    outputfolder = args.output

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
                        fname = make_name(S, N, shells, order, it)
                        bvals, bvecs = read_bvals_bvecs(inputfolder + 'grad/grad_' + fname + '.bval',
                                                        inputfolder + 'grad/grad_' + fname + '.bvec')
                        vol = nib.load(inputfolder + 'data/data_' + fname + '.nii.gz')
                        data = vol.get_data()
                        aff  = vol.get_affine()
                        hdr  = vol.get_header()
                        gtab = gradient_table(bvals=bvals, bvecs=bvecs,
                                              big_delta=51.6, small_delta=32.8, b0_threshold=0.5)
                        mask = np.ones([1, 1, 1], dtype='bool')


                        # Set parameters
                        _where_b0s = lazy_index(gtab.b0s_mask)
                        S0 = np.mean(data[:, :, :, _where_b0s])

                        # Compute DTI LLS
                        tenmodel = TensorModel(gtab, fit_method="LS")
                        tenfit = tenmodel.fit(data)
                        filename = outputfolder + fname + '_DTILLS'
                        screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=show)
                        screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png",
                                         tenfit.evals[:, :, :, 0], show=show)
                        nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff),
                        '_evecs.nii.gz')
                        nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff),
                        '_evals.nii.gz')
                        nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + 'fa.nii.gz')

                        # Compute DTI WLLS
                        tenmodel = TensorModel(gtab, fit_method="WLS")
                        tenfit = tenmodel.fit(data)
                        filename = outputfolder + fname + '_DTIWLLS'
                        screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png",
                                       show=show)
                        screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png",
                                         tenfit.evals[:, :, :, 0], show=show)
                        nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff),
                        '_evecs.nii.gz')
                        nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff),
                        '_evals.nii.gz')
                        nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff),
                                 filename + 'fa.nii.gz')

                        # Compute DTI NLLS
                        tenmodel = TensorModel(gtab, fit_method="NLLS")
                        tenfit = tenmodel.fit(data)
                        filename = outputfolder + fname + '_DTINLLS'
                        screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=show)
                        screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png",
                                         tenfit.evals[:, :, :, 0], show=show)
                        nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff),
                        '_evecs.nii.gz')
                        nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff),
                        '_evals.nii.gz')
                        nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + 'fa.nii.gz')

                        #    # Compute DTI RESTORE
                        #    tenmodel = TensorModel(gtab, fit_method="RT")
                        #    tenfit = tenmodel.fit(data)
                        #    filename = outputfolder + fname + '_DTIRESTORE'
                        #    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=show)
                        #    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=show)
                        #    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff), '_evecs.nii.gz')
                        #    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff), '_evals.nii.gz')
                        #    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + 'fa.nii.gz')

                        # Compute DTI REKINDLE


                        if S == 1:
                            # Compute Qball
                            sh_order = decision(N, 'Qball')
                            qball_model = QballModel(gtab, sh_order=sh_order)
                            peaks_qball = pfm(model=qball_model, data=data, mask=mask,
                                            sphere=sphere, parallel=False, sh_order=sh_order)
                            filename = outputfolder + fname + '_Qball'
                            screenshot_odf(peaks_qball.odf, sphere, filename + "_odf.png",
                                           show=show)
                            screenshot_peaks(peaks_qball.peak_dirs, filename + "_peaks.png",
                                             peaks_qball.peak_values, show=show)
                            nib.save(nib.Nifti1Image(peaks_qball.shm_coeff.astype('float32'),
                                                     aff), filename + 'fodf.nii.gz')
                            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_qball),
                                                     aff), filename + 'peaks.nii.gz')
                            nib.save(nib.Nifti1Image(peaks_qball.peak_indices, aff),
                                     filename + 'fodf_peak_indices.nii.gz')


                            # Compute SDT
                            # Calibrate RF on whole or partial data?
                            ratio = 0.21197
                            sh_order = decision(N, 'SDT')
                            sdt_model = ConstrainedSDTModel(gtab, ratio=ratio, sh_order=sh_order)
                            peaks_sdt = pfm(model=sdt_model, data=data, mask=mask,
                                            sphere=sphere, parallel=False, sh_order=sh_order)
                            filename = outputfolder + fname + '_SDT'
                            screenshot_odf(peaks_sdt.odf, sphere, filename + "_odf.png",
                                           show=show)
                            screenshot_peaks(peaks_sdt.peak_dirs, filename + "_peaks.png",
                                             peaks_sdt.peak_values, show=show)
                            nib.save(nib.Nifti1Image(peaks_sdt.shm_coeff.astype('float32'), aff),
                                     filename + 'fodf.nii.gz')
                            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_sdt), aff),
                                     filename + 'peaks.nii.gz')
                            nib.save(nib.Nifti1Image(peaks_sdt.peak_indices, aff),
                                     filename + 'fodf_peak_indices.nii.gz')


                            # Compute CSA
                            sh_order = decision(N, 'CSA')
                            csa_model = CsaOdfModel(gtab, sh_order=sh_order)
                            peaks_csa = pfm(model=csa_model, data=data, mask=mask,
                                            sphere=sphere, parallel=False, sh_order=sh_order)
                            filename = outputfolder + fname + '_CSA'
                            screenshot_odf(peaks_csa.odf, sphere, filename + "_odf.png",
                                           show=show)
                            screenshot_peaks(peaks_csa.peak_dirs, filename + "_peaks.png",
                                             peaks_csa.peak_values, show=show)
                            nib.save(nib.Nifti1Image(peaks_csa.shm_coeff.astype('float32'),
                                                     aff), filename + 'fodf.nii.gz')
                            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csa),
                                                     aff), filename + 'peaks.nii.gz')
                            nib.save(nib.Nifti1Image(peaks_csa.peak_indices, aff),
                                     filename + 'fodf_peak_indices.nii.gz')


                        if S > 1:
                            # Compute DKI LLS

                            # Compute DKI WLLS

                            # Compute DKI NLLS

                            # Compute DKI RESTORE

                            # Compute DKI REKINDLE


                            # Compute multi shell CSD max_abs
                            response = (np.array([0.0015, 0.0003, 0.0003]), S0)
                            peaks_csd = csd_ms(gtab=gtab, data=data, aff=aff,
                                               mask=mask, response=response,
                                               sphere=sphere, min_angle=25.0, relative_peak_th=0.1)
                            if peaks_csd:
                                filename = outputfolder + fname + '_msCSD'
                                screenshot_odf(peaks_csd.odf, sphere, filename + "_odf.png",
                                               show=show)
                                screenshot_peaks(peaks_csd.peak_dirs, filename + "_peaks.png",
                                                 peaks_csd.peak_values, show=show)
                                nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'),
                                                         aff), filename + 'fodf.nii.gz')
                                nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd),
                                                         aff), filename + 'peaks.nii.gz')
    #                            nib.save(nib.Nifti1Image(peaks_csd.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')


                            # Compute single/multi shell CSD
                            # Calibrate RF on whole or partial data?
                            response = (np.array([0.0015, 0.0003, 0.0003]), S0)
                            sh_order = decision(N, 'CSD')
                            csd_model = ConstrainedSphericalDeconvModel(gtab, response,
                                                                        sh_order=sh_order)
                            peaks_csd = pfm(model=csd_model, data=data, mask=mask,
                                            sphere=sphere, parallel=False, sh_order=sh_order)
                            filename = outputfolder + fname + '_CSD'
                            screenshot_odf(peaks_csd.odf, sphere, filename + "_odf.png",
                                           show=show)
                            screenshot_peaks(peaks_csd.peak_dirs, filename + "_peaks.png",
                                             peaks_csd.peak_values, show=show)
                            nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'),
                                                     aff), filename + 'fodf.nii.gz')
                            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd),
                                                     aff), filename + 'peaks.nii.gz')
                            nib.save(nib.Nifti1Image(peaks_csd.peak_indices, aff),
                                     filename + 'fodf_peak_indices.nii.gz')


                            # Compute multi shell SDT max_abs
                            ratio = 0.21197
                            peaks_sdt = sdt_ms(gtab=gtab, data=data, aff=aff,
                                               mask=mask, ratio=ratio, sphere=sphere, min_angle=25.0, relative_peak_th=0.1)
                            if peaks_sdt:
                                filename = outputfolder + fname + '_msSDT'
                                screenshot_odf(peaks_sdt.odf, sphere, filename + "_odf.png",
                                               show=show)
                                screenshot_peaks(peaks_sdt.peak_dirs, filename + "_peaks.png",
                                                 peaks_sdt.peak_values, show=show)
                                nib.save(nib.Nifti1Image(peaks_sdt.shm_coeff.astype('float32'),
                                                         aff), filename + 'fodf.nii.gz')
                                nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_sdt),
                                                         aff), filename + 'peaks.nii.gz')
    #                            nib.save(nib.Nifti1Image(peaks_sdt.peak_indices, aff), filename + 'fodf_peak_indices.nii.gz')

                            # Compute SHORE
                            sh_order = decision(N, 'SHORE')
                            if sh_order:
                                zeta = 700
                                lambdaN = 1e-8
                                lambdaL = 1e-8
                                shore_model = ShoreModel(gtab, radial_order=sh_order,
                                                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
                                peaks_shore = pfm(model=shore_model, data=data, mask=mask,
                                                sphere=sphere, parallel=False, sh_order=sh_order)
                                filename = outputfolder + fname + '_SHORE'
                                screenshot_odf(peaks_shore.odf, sphere, filename + "_odf.png",
                                               show=show)
                                screenshot_peaks(peaks_shore.peak_dirs, filename + "_peaks.png",
                                                 peaks_shore.peak_values, show=show)
                                nib.save(nib.Nifti1Image(peaks_shore.shm_coeff.astype('float32'),
                                                         aff), filename + 'fodf.nii.gz')
                                nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_shore),
                                                         aff), filename + 'peaks.nii.gz')
                                nib.save(nib.Nifti1Image(peaks_shore.peak_indices, aff),
                                         filename + 'fodf_peak_indices.nii.gz')

                                # Compute SHORED
#                                fodf_sh = odf_sh_to_sharp(peaks_shore.shm_coeff, sphere,
#                                                          basis='mrtrix', ratio=ratio,
#                                                          sh_order=sh_order, lambda_=1., tau=0.1,
#                                                          r2_term=True)


if __name__ == "__main__":
    main()