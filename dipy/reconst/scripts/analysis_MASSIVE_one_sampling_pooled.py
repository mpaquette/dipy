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
show = False
ss = False


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('-data', action='store', metavar='input', type=str,
                   help='Path of the input folder containing folders data/.')

    p.add_argument('-grad', action='store', metavar='input', type=str,
                   help='Path of the input folder containing folders grad/.')

    p.add_argument('-output', action='store', metavar='output', type=str,
                   help='Path of the output folder.')

    p.add_argument('-tag', action='store', metavar='output', type=str,
                   help='Name of sampling technique.')

    p.add_argument('-s', action='store', metavar='S', type=int,
                   help='Numbers of shells in sampling scheme')

    p.add_argument('-n', action='store', metavar='N', type=int,
                   help='Numbers of samples in sampling scheme')

    p.add_argument('-shells', action='store', metavar='shells', type=int, nargs='+',
                   help='Shell pattern.')

    p.add_argument('-order', action='store', metavar='order', type=int,
                   help='Order of the sampling scheme points distribution.')

    p.add_argument('-it', action='store', metavar='it', type=int,
                   help='Marker for distinct sampling scheme with same parameters.')

    p.add_argument('-sigma', action='store', metavar='sigma', type=float, nargs='+',
                   help='Noise sigmas.')

    p.add_argument('-trial', action='store', metavar='n_trial', type=int,
                   help='Marker for distinct noise trial.')

    p.add_argument('-skip', action='store', metavar='skip_trial', type=int,
               help='Noise trial to skip.')



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
    # elif model == 'SHORE':
    #     if N >= 35 and N < 84:
    #         sh_order = 4
    #     elif N >= 84 and N < 165:
    #         sh_order = 6
    #     elif N >= 165:
    #         sh_order = 8
    #     else:
    #         return False
    elif model == 'SHORE':
      if N >= 35 and N < 60:
          sh_order = 6
      elif N >= 60:
          sh_order = 8
      else:
          return False
    return sh_order

def add_rician_noise(signal, sigma, seed = 0):
    # sigma == 1/SNR
    if sigma == 0:
        return signal
    else:
        np.random.seed(seed)
        n1 = np.random.randn(*signal.shape)
        n2 = np.random.randn(*signal.shape)
        return np.sqrt((signal + sigma * n1)**2 + (sigma * n2)**2)

def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    datafolder = args.data
    gradfolder = args.grad
    outputfolder = args.output

    sampName = args.tag + '_'

    S = args.s
    N = args.n
    shells = args.shells
    order = args.order
    it = args.it

    sigma = args.sigma
    n_trial = args.trial
    skip_trial = args.skip


    fname = make_name(S, N, shells, order, it)
    bvals, bvecs = read_bvals_bvecs(gradfolder + sampName + fname + '.bvals',
                                    gradfolder + sampName + fname + '.bvecs')
    vol = nib.load(datafolder + sampName + fname + '.nii.gz')
    data_noiseless = vol.get_data()
    aff  = vol.get_affine()
    hdr  = vol.get_header()
    gtab = gradient_table(bvals=bvals, bvecs=bvecs,
                          big_delta=51.6, small_delta=32.8, b0_threshold=0.5)
    # mask = np.ones([1, 1, 1], dtype='bool')
    mask = np.ones_like(data_noiseless[..., 0], dtype='bool')


    # Set parameters
    _where_b0s = lazy_index(gtab.b0s_mask)


    hardcoded_npeaks_max = 5

    DTILLS_evecs = np.zeros((21, len(sigma), n_trial - skip_trial, 3, 3))
    DTILLS_evals = np.zeros((21, len(sigma), n_trial - skip_trial, 3))
    DTILLS_fa = np.zeros((21, len(sigma), n_trial - skip_trial))

    DTIWLLS_evecs = np.zeros((21, len(sigma), n_trial - skip_trial, 3, 3))
    DTIWLLS_evals = np.zeros((21, len(sigma), n_trial - skip_trial, 3))
    DTIWLLS_fa = np.zeros((21, len(sigma), n_trial - skip_trial))

    DTINLLS_evecs = np.zeros((21, len(sigma), n_trial - skip_trial, 3, 3))
    DTINLLS_evals = np.zeros((21, len(sigma), n_trial - skip_trial, 3))
    DTINLLS_fa = np.zeros((21, len(sigma), n_trial - skip_trial))

    sh_order_csd = decision(N, 'CSD')
    coef_csd = round( 0.5 * (sh_order_csd + 1) * (sh_order_csd + 2))

    CSD_fodf = np.zeros((21, len(sigma), n_trial - skip_trial, coef_csd))
    CSD_peaks = np.zeros((21, len(sigma), n_trial - skip_trial, 3*hardcoded_npeaks_max))
    CSD_ind = np.zeros((21, len(sigma), n_trial - skip_trial, hardcoded_npeaks_max))

    if S == 1:
        Qball_fodf = np.zeros((21, len(sigma), n_trial - skip_trial, coef_csd))
        Qball_peaks = np.zeros((21, len(sigma), n_trial - skip_trial, 3*hardcoded_npeaks_max))
        Qball_ind = np.zeros((21, len(sigma), n_trial - skip_trial, hardcoded_npeaks_max))

        SDT_fodf = np.zeros((21, len(sigma), n_trial - skip_trial, coef_csd))
        SDT_peaks = np.zeros((21, len(sigma), n_trial - skip_trial, 3*hardcoded_npeaks_max))
        SDT_ind = np.zeros((21, len(sigma), n_trial - skip_trial, hardcoded_npeaks_max))

        sh_order_csa = decision(N, 'CSA')
        coef_csa = round( 0.5 * (sh_order_csd + 1) * (sh_order_csd + 2))
        CSA_fodf = np.zeros((21, len(sigma), n_trial - skip_trial, coef_csa))
        CSA_peaks = np.zeros((21, len(sigma), n_trial - skip_trial, 3*hardcoded_npeaks_max))
        CSA_ind = np.zeros((21, len(sigma), n_trial - skip_trial, hardcoded_npeaks_max))

    if S > 1:
        msCSD_fodf = np.zeros((21, len(sigma), n_trial - skip_trial, coef_csd))
        msCSD_peaks = np.zeros((21, len(sigma), n_trial - skip_trial, 3*hardcoded_npeaks_max))
        # msCSD_ind = np.zeros((21, len(sigma), n_trial - skip_trial, hardcoded_npeaks_max))

        msSDT_fodf = np.zeros((21, len(sigma), n_trial - skip_trial, coef_csd))
        msSDT_peaks = np.zeros((21, len(sigma), n_trial - skip_trial, 3*hardcoded_npeaks_max))
        # msSDT_ind = np.zeros((21, len(sigma), n_trial - skip_trial, hardcoded_npeaks_max))





    for (i_sig, sig) in enumerate(sigma):
        if sig == 0:
            n_tr = [0]
        else:
            n_tr = range(skip_trial, n_trial)
        for (i_seed, seed) in enumerate(n_tr):

            noisetag = '_sig-{:.3f}_tr-{}'.format(sig, seed)

            data = add_rician_noise(data_noiseless, sig, seed)

            S0 = np.mean(data[:, :, :, _where_b0s])
            min_signal = 0.001 * S0

            # Compute DTI LLS
            tenmodel = TensorModel(gtab, fit_method="LS", min_signal = min_signal)
            tenfit = tenmodel.fit(data)
            filename = outputfolder + fname + '_DTILLS'
            if ss:
                screenshot_odf(tenfit.odf(sphere), sphere, filename + noisetag + "_odf.png", show=show)
                # screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + noisetag + "_peaks.png",
                #                  tenfit.evals[:, :, :, 0], show=show)
                screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + noisetag + "_peaks.png", show=show)
            # nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff),
            # filename + noisetag + '_evecs.nii.gz')
            # nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff),
            # filename + noisetag + '_evals.nii.gz')
            # nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + noisetag + '_fa.nii.gz')

            DTILLS_evecs[:, i_sig, i_seed] = tenfit.evecs.squeeze()
            DTILLS_evals[:, i_sig, i_seed] = tenfit.evals.squeeze()
            DTILLS_fa[:, i_sig, i_seed] = tenfit.fa.squeeze()

            # Compute DTI WLLS
            tenmodel = TensorModel(gtab, fit_method="WLS", min_signal = min_signal)
            tenfit = tenmodel.fit(data)
            filename = outputfolder + fname + '_DTIWLLS'
            if ss:
                screenshot_odf(tenfit.odf(sphere), sphere, filename + noisetag + "_odf.png",
                                show=show)
                # screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + noisetag + "_peaks.png",
                #                  tenfit.evals[:, :, :, 0], show=show)
                screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + noisetag + "_peaks.png", show=show)
            # nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff),
            # filename + noisetag + '_evecs.nii.gz')
            # nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff),
            # filename + noisetag + '_evals.nii.gz')
            # nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff),
            #          filename + noisetag + '_fa.nii.gz')

            DTIWLLS_evecs[:, i_sig, i_seed] = tenfit.evecs.squeeze()
            DTIWLLS_evals[:, i_sig, i_seed] = tenfit.evals.squeeze()
            DTIWLLS_fa[:, i_sig, i_seed] = tenfit.fa.squeeze()

            # Compute DTI NLLS
            tenmodel = TensorModel(gtab, fit_method="NLLS", min_signal = min_signal)
            tenfit = tenmodel.fit(data)
            filename = outputfolder + fname + '_DTINLLS'
            if ss:
                screenshot_odf(tenfit.odf(sphere), sphere, filename + noisetag + "_odf.png", show=show)
                # screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + noisetag + "_peaks.png",
                #                  tenfit.evals[:, :, :, 0], show=show)
                screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + noisetag + "_peaks.png", show=show)
            # nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff),
            # filename + noisetag + '_evecs.nii.gz')
            # nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff),
            # filename + noisetag + '_evals.nii.gz')
            # nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + noisetag + '_fa.nii.gz')

            DTINLLS_evecs[:, i_sig, i_seed] = tenfit.evecs.squeeze()
            DTINLLS_evals[:, i_sig, i_seed] = tenfit.evals.squeeze()
            DTINLLS_fa[:, i_sig, i_seed] = tenfit.fa.squeeze()

            #    # Compute DTI RESTORE
            #    tenmodel = TensorModel(gtab, fit_method="RT")
            #    tenfit = tenmodel.fit(data)
            #    filename = outputfolder + fname + '_DTIRESTORE'
            #    screenshot_odf(tenfit.odf(sphere), sphere, filename + noisetag + "_odf.png", show=show)
            #    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + noisetag + "_peaks.png", tenfit.evals[:, :, :, 0], show=show)
            #    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), aff), filename + noisetag + '_evecs.nii.gz')
            #    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), aff), filename + noisetag + '_evals.nii.gz')
            #    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), aff), filename + noisetag + 'fa.nii.gz')

            # Compute DTI REKINDLE


            if S == 1:
                # Compute Qball
                sh_order = decision(N, 'Qball')
                qball_model = QballModel(gtab, sh_order=sh_order, min_signal = min_signal)
                peaks_qball = pfm(model=qball_model, data=data, mask=mask,
                                sphere=sphere, parallel=False, sh_order=sh_order)
                filename = outputfolder + fname + '_Qball'
                if ss:
                    screenshot_odf(peaks_qball.odf, sphere, filename + noisetag + "_odf.png",
                                 show=show)
                    # screenshot_peaks(peaks_qball.peak_dirs, filename + noisetag + "_peaks.png",
                    #                  peaks_qball.peak_values, show=show)
                #     screenshot_peaks(peaks_qball.peak_dirs, filename + noisetag + "_peaks.png", show=show)
                # nib.save(nib.Nifti1Image(peaks_qball.shm_coeff.astype('float32'),
                #                          aff), filename + noisetag + '_fodf.nii.gz')
                # nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_qball),
                #                          aff), filename + noisetag + '_peaks.nii.gz')
                # nib.save(nib.Nifti1Image(peaks_qball.peak_indices, aff),
                #          filename + noisetag + '_fodf_peak_indices.nii.gz')

                Qball_fodf[:, i_sig, i_seed] = peaks_qball.shm_coeff.squeeze()
                Qball_peaks[:, i_sig, i_seed] = reshape_peaks_for_visualization(peaks_qball).squeeze()
                Qball_ind[:, i_sig, i_seed] = peaks_qball.peak_indices.squeeze()


                # Compute SDT
                # Calibrate RF on whole or partial data?
                ratio = 0.21197
                sh_order = decision(N, 'SDT')
                sdt_model = ConstrainedSDTModel(gtab, ratio=ratio, sh_order=sh_order)
                peaks_sdt = pfm(model=sdt_model, data=data, mask=mask,
                                sphere=sphere, parallel=False, sh_order=sh_order)
                filename = outputfolder + fname + '_SDT'
                if ss:
                    screenshot_odf(peaks_sdt.odf, sphere, filename + noisetag + "_odf.png",
                                 show=show)
                    # screenshot_peaks(peaks_sdt.peak_dirs, filename + noisetag + "_peaks.png",
                    #                  peaks_sdt.peak_values, show=show)
                #     screenshot_peaks(peaks_sdt.peak_dirs, filename + noisetag + "_peaks.png", show=show)
                # nib.save(nib.Nifti1Image(peaks_sdt.shm_coeff.astype('float32'), aff),
                #          filename + noisetag + '_fodf.nii.gz')
                # nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_sdt), aff),
                #          filename + noisetag + '_peaks.nii.gz')
                # nib.save(nib.Nifti1Image(peaks_sdt.peak_indices, aff),
                #          filename + noisetag + '_fodf_peak_indices.nii.gz')

                SDT_fodf[:, i_sig, i_seed] = peaks_sdt.shm_coeff.squeeze()
                SDT_peaks[:, i_sig, i_seed] = reshape_peaks_for_visualization(peaks_sdt).squeeze()
                SDT_ind[:, i_sig, i_seed] = peaks_sdt.peak_indices.squeeze()


                # Compute CSA
                sh_order = decision(N, 'CSA')
                csa_model = CsaOdfModel(gtab, sh_order=sh_order, min_signal = min_signal)
                peaks_csa = pfm(model=csa_model, data=data, mask=mask,
                                sphere=sphere, parallel=False, sh_order=sh_order)
                filename = outputfolder + fname + '_CSA'
                if ss:
                    screenshot_odf(peaks_csa.odf, sphere, filename + noisetag + "_odf.png",
                                 show=show)
                    # screenshot_peaks(peaks_csa.peak_dirs, filename + noisetag + "_peaks.png",
                    #                  peaks_csa.peak_values, show=show)
                #     screenshot_peaks(peaks_csa.peak_dirs, filename + noisetag + "_peaks.png", show=show)
                # nib.save(nib.Nifti1Image(peaks_csa.shm_coeff.astype('float32'),
                #                          aff), filename + noisetag + '_fodf.nii.gz')
                # nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csa),
                #                          aff), filename + noisetag + '_peaks.nii.gz')
                # nib.save(nib.Nifti1Image(peaks_csa.peak_indices, aff),
                #          filename + noisetag + '_fodf_peak_indices.nii.gz')

                CSA_fodf[:, i_sig, i_seed] = peaks_csa.shm_coeff.squeeze()
                CSA_peaks[:, i_sig, i_seed] = reshape_peaks_for_visualization(peaks_csa).squeeze()
                CSA_ind[:, i_sig, i_seed] = peaks_csa.peak_indices.squeeze()


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
                    if ss:
                        screenshot_odf(peaks_csd.odf, sphere, filename + noisetag + "_odf.png",
                                 show=show)
                        # screenshot_peaks(peaks_csd.peak_dirs, filename + noisetag + "_peaks.png",
                        #                  peaks_csd.peak_values, show=show)
                    #     screenshot_peaks(peaks_csd.peak_dirs, filename + noisetag + "_peaks.png", show=show)
                    # nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'),
                    #                          aff), filename + noisetag + '_fodf.nii.gz')
                    # nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd),
                    #                          aff), filename + noisetag + '_peaks.nii.gz')
                    # # nib.save(nib.Nifti1Image(peaks_csd.peak_indices, aff), filename + noisetag + '_fodf_peak_indices.nii.gz')


                msCSD_fodf[:, i_sig, i_seed] = peaks_csd.shm_coeff.squeeze()
                msCSD_peaks[:, i_sig, i_seed] = reshape_peaks_for_visualization(peaks_csd).squeeze()
                # msCSD_ind[:, i_sig, i_seed] = peaks_csd.peak_indices

                # Compute multi shell SDT max_abs
                ratio = 0.21197
                peaks_sdt = sdt_ms(gtab=gtab, data=data, aff=aff,
                                   mask=mask, ratio=ratio, sphere=sphere, min_angle=25.0, relative_peak_th=0.1)
                if peaks_sdt:
                    filename = outputfolder + fname + '_msSDT'
                    if ss:
                      screenshot_odf(peaks_sdt.odf, sphere, filename + noisetag + "_odf.png",
                                     show=show)
                      # screenshot_peaks(peaks_sdt.peak_dirs, filename + noisetag + "_peaks.png",
                      #                  peaks_sdt.peak_values, show=show)
                      screenshot_peaks(peaks_sdt.peak_dirs, filename + noisetag + "_peaks.png", show=show)
                    # nib.save(nib.Nifti1Image(peaks_sdt.shm_coeff.astype('float32'),
                    #                          aff), filename + noisetag + '_fodf.nii.gz')
                    # nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_sdt),
                    #                          aff), filename + noisetag + '_peaks.nii.gz')
                    # nib.save(nib.Nifti1Image(peaks_sdt.peak_indices, aff), filename + noisetag + '_fodf_peak_indices.nii.gz')

                msSDT_fodf[:, i_sig, i_seed] = peaks_sdt.shm_coeff.squeeze()
                msSDT_peaks[:, i_sig, i_seed] = reshape_peaks_for_visualization(peaks_sdt).squeeze()
                # msSDT_ind[:, i_sig, i_seed] = peaks_sdt.peak_indices


            # Compute single/multi shell CSD
            # Calibrate RF on whole or partial data?
            response = (np.array([0.0015, 0.0003, 0.0003]), S0)
            sh_order = decision(N, 'CSD')
            csd_model = ConstrainedSphericalDeconvModel(gtab, response,
                                                        sh_order=sh_order)
            peaks_csd = pfm(model=csd_model, data=data, mask=mask,
                            sphere=sphere, parallel=False, sh_order=sh_order)
            filename = outputfolder + fname + '_CSD'
            if ss:
                screenshot_odf(peaks_csd.odf, sphere, filename + noisetag + "_odf.png",
                             show=show)
                # screenshot_peaks(peaks_csd.peak_dirs, filename + noisetag + "_peaks.png",
                #                  peaks_csd.peak_values, show=show)
            #     screenshot_peaks(peaks_csd.peak_dirs, filename + noisetag + "_peaks.png", show=show)
            # nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'),
            #                          aff), filename + noisetag + '_fodf.nii.gz')
            # nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd),
            #                          aff), filename + noisetag + '_peaks.nii.gz')
            # nib.save(nib.Nifti1Image(peaks_csd.peak_indices, aff),
            #          filename + noisetag + '_fodf_peak_indices.nii.gz')

            CSD_fodf[:, i_sig, i_seed] = peaks_csd.shm_coeff.squeeze()
            CSD_peaks[:, i_sig, i_seed] = reshape_peaks_for_visualization(peaks_csd).squeeze()
            # CSD_ind[:, i_sig, i_seed] = peaks_csd.peak_indices.squeeze()




                # # Compute SHORE
                # sh_order = decision(N, 'SHORE')
                # if sh_order:
                #     zeta = 700
                #     lambdaN = 1e-8
                #     lambdaL = 1e-8
                #     shore_model = ShoreModel(gtab, radial_order=sh_order,
                #                      zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL, constrain_e0=True)
                #     peaks_shore = pfm(model=shore_model, data=data, mask=mask,
                #                     sphere=sphere, parallel=False, sh_order=sh_order)
                #     filename = outputfolder + fname + '_SHORE'
                #     if ss:
                #         screenshot_odf(peaks_shore.odf, sphere, filename + noisetag + "_odf.png",
                #                        show=show)
                #         # screenshot_peaks(peaks_shore.peak_dirs, filename + noisetag + "_peaks.png",
                #         #                  peaks_shore.peak_values, show=show)
                #         screenshot_peaks(peaks_shore.peak_dirs, filename + noisetag + "_peaks.png", show=show)
                #     nib.save(nib.Nifti1Image(peaks_shore.shm_coeff.astype('float32'),
                #                              aff), filename + noisetag + '_fodf.nii.gz')
                #     nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_shore),
                #                              aff), filename + noisetag + '_peaks.nii.gz')
                #     nib.save(nib.Nifti1Image(peaks_shore.peak_indices, aff),
                #              filename + noisetag + '_fodf_peak_indices.nii.gz')



                    # Compute SHORED
        #                                fodf_sh = odf_sh_to_sharp(peaks_shore.shm_coeff, sphere,
        #                                                          basis='mrtrix', ratio=ratio,
        #                                                          sh_order=sh_order, lambda_=1., tau=0.1,
        #                                                          r2_term=True)

    filename = outputfolder + fname + '_DTILLS'
    nib.save(nib.Nifti1Image(DTILLS_evecs.astype(np.float32), aff), filename + '_evecs.nii.gz')
    nib.save(nib.Nifti1Image(DTILLS_evals.astype(np.float32), aff), filename + '_evals.nii.gz')
    nib.save(nib.Nifti1Image(DTILLS_fa.astype(np.float32), aff), filename + '_fa.nii.gz')

    filename = outputfolder + fname + '_DTIWLLS'
    nib.save(nib.Nifti1Image(DTIWLLS_evecs.astype(np.float32), aff), filename + '_evecs.nii.gz')
    nib.save(nib.Nifti1Image(DTIWLLS_evals.astype(np.float32), aff), filename + '_evals.nii.gz')
    nib.save(nib.Nifti1Image(DTIWLLS_fa.astype(np.float32), aff), filename + '_fa.nii.gz')

    filename = outputfolder + fname + '_DTINLLS'
    nib.save(nib.Nifti1Image(DTINLLS_evecs.astype(np.float32), aff), filename + '_evecs.nii.gz')
    nib.save(nib.Nifti1Image(DTINLLS_evals.astype(np.float32), aff), filename + '_evals.nii.gz')
    nib.save(nib.Nifti1Image(DTINLLS_fa.astype(np.float32), aff), filename + '_fa.nii.gz')

    filename = outputfolder + fname + '_CSD'
    nib.save(nib.Nifti1Image(CSD_fodf.astype('float32'), aff), filename + '_fodf.nii.gz')
    nib.save(nib.Nifti1Image(CSD_peaks.astype('float32'), aff), filename + '_peaks.nii.gz')
    nib.save(nib.Nifti1Image(CSD_ind, aff), filename + '_fodf_peak_indices.nii.gz')

    if S == 1:
        filename = outputfolder + fname + '_Qball'
        nib.save(nib.Nifti1Image(Qball_fodf.astype('float32'), aff), filename + '_fodf.nii.gz')
        nib.save(nib.Nifti1Image(Qball_peaks.astype('float32'), aff), filename + '_peaks.nii.gz')
        nib.save(nib.Nifti1Image(Qball_ind, aff), filename + '_fodf_peak_indices.nii.gz')

        filename = outputfolder + fname + '_SDT'
        nib.save(nib.Nifti1Image(SDT_fodf.astype('float32'), aff), filename + '_fodf.nii.gz')
        nib.save(nib.Nifti1Image(SDT_peaks.astype('float32'), aff), filename + '_peaks.nii.gz')
        nib.save(nib.Nifti1Image(SDT_ind, aff), filename + '_fodf_peak_indices.nii.gz')

        filename = outputfolder + fname + '_CSA'
        nib.save(nib.Nifti1Image(CSA_fodf.astype('float32'), aff), filename + '_fodf.nii.gz')
        nib.save(nib.Nifti1Image(CSA_peaks.astype('float32'), aff), filename + '_peaks.nii.gz')
        nib.save(nib.Nifti1Image(CSA_ind, aff), filename + '_fodf_peak_indices.nii.gz')

    if S > 1:

        filename = outputfolder + fname + '_msSDT'
        nib.save(nib.Nifti1Image(msSDT_fodf.astype('float32'), aff), filename + '_fodf.nii.gz')
        nib.save(nib.Nifti1Image(msSDT_peaks.astype('float32'), aff), filename + '_peaks.nii.gz')
        nib.save(nib.Nifti1Image(msSDT_ind, aff), filename + '_fodf_peak_indices.nii.gz')

        filename = outputfolder + fname + '_msCSD'
        nib.save(nib.Nifti1Image(msCSD_fodf.astype('float32'), aff), filename + '_fodf.nii.gz')
        nib.save(nib.Nifti1Image(msCSD_peaks.astype('float32'), aff), filename + '_peaks.nii.gz')
        nib.save(nib.Nifti1Image(msCSD_ind, aff), filename + '_fodf_peak_indices.nii.gz')


if __name__ == "__main__":
    main()
