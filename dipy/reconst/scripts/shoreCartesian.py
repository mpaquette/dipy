from __future__ import division, print_function

import numpy as np
import nibabel as nib

# Example using the Cartesian Shore to process the data

from sparc_dmri.load_data import get_data, get_mask
from sparc_dmri.output import compute_npeaks_and_angles, screenshot_odf, screenshot_peaks

from dipy.reconst.shore_cart import ShoreCartModel
from dipy.data import get_sphere
from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization
from dipy.reconst.dti import TensorModel


# Load the dataset
# filename = "grad_60_dirs"
ndir = 60
shell = None
# denoised = None
denoised = 'nlm'
# denoised = 'nlsam'
filename = 'grad_' + str(ndir) + '_dirs_' + str(shell) + '_'

data, affine, gtab = get_data(ndir, shell=shell, denoised=denoised)
mask = get_mask()
mask = mask[:, :, None]
print('Data loaded')
print(data.shape)

# Data correction
for vox in np.ndindex(mask.shape):
    if mask[vox]:
        toFix = data[vox][:]>data[vox][0]
        data[vox][toFix] = data[vox][0]

if ndir == 20:
    # 60 q-points, 50 coeffs
    radial_order = 6
elif ndir == 30:
    # 90 q-points, 95 coeffs
    # sparse reconstruction wink, wink!
    radial_order = 8
elif ndir == 60:
    # 180 q-points, 95 coeffs
    radial_order = 8
# elif ndir == 60:
#     # 180 q-points, 161 coeffs
#     radial_order = 10


zeta = 700.
mu = 1/ (2 * np.pi * np.sqrt(zeta))
lambd = 0.005
constrain_e0 = True
positive_constraint = False
tau = 0.07

# Shore Cartesian Model
shore_cart_model = ShoreCartModel(gtab,
                                  radial_order = radial_order,
                                  mu = mu,
                                  lambd = lambd,
                                  e0_cons = constrain_e0,
                                  eap_cons = positive_constraint,
                                  tau = tau)

# Nice big sphere
# sphere = get_sphere('symmetric724')
sphere = get_sphere('symmetric724').subdivide(1)
print(sphere.vertices.shape)

# ODF computation and peak extraction, smoment SHOULD be 1 in shore_cart.py
peaks = peaks_from_model(model=shore_cart_model,
                         data=data,
                         sphere=sphere,
                         mask=mask,
                         relative_peak_threshold=.5,
                         min_separation_angle=25,
                         parallel=False,
                         return_sh=True,
                         normalize_peaks=True,
                         npeaks=3,
                         return_odf=True)
print('peaks_from_model finished')
print(peaks.peak_dirs.shape)


# Fit again for signal estimation because urrrhghg.. don't get me started
shore_fit = shore_cart_model.fit(data=data,
                                 mask=mask)
print('Shore Fitted')
print(shore_fit.shore_coeff.shape)


nfib = np.sum(np.sum(np.abs(peaks.peak_dirs), axis=-1) > 0, axis=-1).ravel()
print("1 fiber", np.sum(nfib==1), "2 fibers", np.sum(nfib==2), "3 fibers", np.sum(nfib==3))

# kill negative lobe of the sharpening
odfs = peaks.odf
odfs[odfs<0] = 0
screenshot_odf(odfs, sphere, filename + "odf.png", 2.)
screenshot_peaks(peaks.peak_dirs, filename + "peaks.png", 2.)

# Save everything
nib.save(nib.Nifti1Image(peaks.shm_coeff.astype('float32'), affine), filename + 'ODF_ShoreCart.nii.gz')
nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks), affine), filename + 'peaks_ShoreCart.nii.gz')
nib.save(nib.Nifti1Image(peaks.peak_indices, affine), filename + 'odf_ShoreCart_peak_indices.nii.gz')


# Get the number of fiber and angles in the required format
angles = compute_npeaks_and_angles(peaks.peak_dirs)

np.savetxt(filename + 'angles.txt', angles[:1], fmt="%i")
f_handle = file(filename + 'angles.txt', 'a')

np.savetxt(f_handle, angles[1:], fmt="%.4f")
f_handle.close()



# Signal estimation
print('Signal Estimation')

from scipy.optimize import curve_fit
from dipy.reconst.shore_cart import shore_evaluate_E_cheapFaster as shore_evaluate_E
from dipy.reconst.shore_cart import shore_phi_matrix_E


def func(b, D):
    return np.exp(-b * np.abs(D))

def func2(b, f, D1, D2):
    f1 = np.exp(f) / (np.exp(f) + 1)
    return f1 * np.exp(-b * np.abs(D1)) + (1-f1) * np.exp(-b * np.abs(D2))

# Load original dataset for isotropic voxels
denoised = None
data_orig, _, _ = get_data(ndir, shell=shell, denoised=denoised)
print('Data loaded')
print(data.shape)

# Data correction
for vox in np.ndindex(mask.shape):
    if not mask[vox]:
        toFix = data_orig[vox][:]>data_orig[vox][0]
        data_orig[vox][toFix] = data_orig[vox][0]

shore_fit_iso = shore_cart_model.fit(data=data_orig,
                                 mask=(1-mask).astype(np.bool))


qlist = np.genfromtxt('data/EstimatedSignal_qvec.txt')

KK = shore_phi_matrix_E(radial_order, mu, qlist, tau)

signal_estimated = np.zeros(mask.shape + (qlist.shape[0],))
signal_estimated2 = np.zeros(mask.shape + (qlist.shape[0],))
signal_estimated3 = np.zeros(mask.shape + (qlist.shape[0],))

for vox in np.ndindex(mask.shape):
    if mask[vox]:
        print('Voxel {}'.format(vox))
        # WM voxel, we use SHORE to estimate signal
        est_signal = shore_evaluate_E(radial_order = radial_order,
                                      coeff = shore_fit.shore_coeff[vox],
                                      qlist = qlist,
                                      mu = mu,
                                      K = KK)


        est_signal /= data[vox][0]
        est_signal[est_signal > 1] = 1
        est_signal[est_signal < 0] = 0
        signal_estimated[vox] = est_signal
        signal_estimated2[vox] = est_signal
        signal_estimated3[vox] = est_signal

    else:
        print('Voxel {} is Iso'.format(vox))
        #Isotropic voxel, we use exponential fit

        signal_b = np.zeros(4)
        signal_b[0] = 1.
        for i in range(3):
            signal_b[i+1] = data_orig[vox][i*ndir + 1 : (i+1)*ndir + 1].mean() / data_orig[vox][0]

        # signal_b[1] = data[vox][1:61].mean()
        # signal_b[2] = data[vox][61:121].mean()
        # signal_b[3] = data[vox][121:181].mean()

        b = np.array([0, 1, 2, 3])
        try:
            popt, pcov = curve_fit(func, b, signal_b)
            b = np.array([1, 2, 3, 4, 5])
            values = func(b,popt)
            values[values > 1] = 1
            values[values < 0] = 0
            print(values)
        except RuntimeError:
            print("Error - curve_fit failed")
        # popt, pcov = curve_fit(func2, b, signal_b)


        # values = func2(b,*popt)
        # print(np.exp([popt[0]]) / (np.exp(popt[0]) + 1))
        # print(values)

        for i in range(5):
            signal_estimated[vox][i*81 : (i+1)*81] = values[i]

        ##################
        signal_b = np.zeros(4)
        signal_b[0] = 1.
        for i in range(3):
            signal_b[i+1] = data_orig[vox][i*ndir + 1 : (i+1)*ndir + 1].mean() / data_orig[vox][0]

        # signal_b[1] = data[vox][1:61].mean()
        # signal_b[2] = data[vox][61:121].mean()
        # signal_b[3] = data[vox][121:181].mean()

        b = np.array([0, 1, 2, 3])
        # popt, pcov = curve_fit(func, b, signal_b)
        try:
            popt, pcov = curve_fit(func2, b, signal_b)
            b = np.array([1, 2, 3, 4, 5])
            # values = func(b,popt)
            values = func2(b,*popt)
            print(np.exp([popt[0]]) / (np.exp(popt[0]) + 1))
            values[values > 1] = 1
            values[values < 0] = 0
            print(values)
        except RuntimeError:
            print("Error - curve_fit failed")

        # print(np.exp([popt[0]]) / (np.exp(popt[0]) + 1))
        # print(values)

        for i in range(5):
            signal_estimated2[vox][i*81 : (i+1)*81] = values[i]

        ##################
        est_signal = shore_evaluate_E(radial_order = radial_order,
                              coeff = shore_fit_iso.shore_coeff[vox],
                              qlist = qlist,
                              mu = mu,
                              K = KK)

        est_signal /= data_orig[vox][0]
        est_signal[est_signal > 1] = 1
        est_signal[est_signal < 0] = 0
        signal_estimated3[vox] = est_signal


np.savetxt(filename + 'signal.txt', signal_estimated3.reshape(208,405), fmt="%.4f")

# nib.save(nib.Nifti1Image(signal_estimated.astype('float32'), affine), filename + 'Signal_monoexp_ShoreCart.nii.gz')
# nib.save(nib.Nifti1Image(signal_estimated2.astype('float32'), affine), filename + 'Signal_biexp_ShoreCart.nii.gz')
nib.save(nib.Nifti1Image(signal_estimated3.astype('float32'), affine), filename + 'Signal_shore_ShoreCart.nii.gz')

