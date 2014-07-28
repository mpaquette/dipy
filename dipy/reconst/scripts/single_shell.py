from __future__ import division, print_function

import numpy as np
import nibabel as nib

from itertools import product
# Example using the CSD to process the data

from dipy.reconst.csdeconv import (auto_response, ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel)
from dipy.reconst.shm import (QballModel, CsaOdfModel, lazy_index)
from dipy.data import get_sphere
from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk

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



# Load the dataset
ndir = 60
shell = None

# need to write a get_data function

#img = nib.load('Gradient_60.nii')
directory = 'D:/H_schijf/Data/MASSIVE/Processed/'
img = nib.load(directory + 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells_48_40_39.nii')
data = img.get_data()
affine = img.get_affine()

bvals = np.loadtxt(directory + 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells.bval')
bvecs = np.loadtxt(directory + 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells.bvec')

ind1000 = bvals < 1100
S1000 = data[..., ind1000]
bvals = bvals[ind1000]
bvecs = bvecs[:,ind1000]

gtab = gradient_table(bvals=bvals, bvecs=bvecs, big_delta=51.6, small_delta=32.8,
                      b0_threshold=0.5)
_where_b0s = lazy_index(gtab.b0s_mask)
S0 = np.mean(data[:, :, :, _where_b0s])

sh_order = 8
print("SH order is", sh_order)
#response, ratio = auto_response(gtab, S1000, roi_radius=1, fa_thr=0.9)
response = (np.array([0.0015, 0.0003, 0.0003]), S0)

# Compute tensors
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(S1000)
sphere = get_sphere('symmetric724').subdivide()

filename = directory + 'test'

# Compute CSD
S0 = response[1]
(a, b) = (17., 3.)
#response = (1e-4 * np.array([a, b, b]), S0)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
peaks_csd = peaks_from_model(model=csd_model,
                             data=S1000,
                             sphere=sphere,
                             relative_peak_threshold=.25,
                             min_separation_angle=25,
                             parallel=False,
                             npeaks=5,
                             return_sh=True,
                             normalize_peaks=False,
                             return_odf=True,
                             sh_order=sh_order)

nfib = np.sum(np.sum(np.abs(peaks_csd.peak_dirs), axis=-1) > 0, axis=-1).ravel()
print("1 fiber", np.sum(nfib==1), "2 fibers", np.sum(nfib==2), "3 fibers", np.sum(nfib==3))

screenshot_odf(peaks_csd.odf, sphere, filename + str(a) + "_" + str(b) + "_CSDodf.png", show=True)
screenshot_peaks(peaks_csd.peak_dirs, filename + str(a) + "_" + str(b) +
                 "_CSDpeaks.png", peaks_csd.peak_values, show=True)
# Save everything
nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'), affine), filename + 'fodf_CSD.nii.gz')
nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd), affine), filename + 'peaks_CSD.nii.gz')
nib.save(nib.Nifti1Image(peaks_csd.peak_indices, affine), filename + 'fodf_CSD_peak_indices.nii.gz')
nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), affine), filename + 'fa.nii.gz')

# Compute Qball
qball_model = QballModel(gtab, sh_order=sh_order)
peaks_qball = peaks_from_model(model=qball_model,
                               data=S1000,
                               sphere=sphere,
                               relative_peak_threshold=.25,
                               min_separation_angle=25,
                               parallel=False,
                               npeaks=5,
                               return_sh=True,
                               normalize_peaks=False,
                               return_odf=True,
                               sh_order=sh_order)
screenshot_odf(peaks_qball.odf, sphere, filename + "_Qballodf.png", show=True)
screenshot_peaks(peaks_qball.peak_dirs, filename + "_Qballpeaks.png", peaks_qball.peak_values,
                 show=True)
# Compute Qball
csa_model = CsaOdfModel(gtab, sh_order=4)
peaks_csa = peaks_from_model(model=csa_model,
                             data=S1000,
                             sphere=sphere,
                             relative_peak_threshold=.25,
                             min_separation_angle=25,
                             parallel=False,
                             npeaks=5,
                             return_sh=True,
                             normalize_peaks=False,
                             return_odf=True,
                             sh_order=sh_order)
screenshot_odf(peaks_csa.odf, sphere, filename + "_CSAodf.png", show=True)
screenshot_peaks(peaks_csa.peak_dirs, filename + "_CSApeaks.png", peaks_csa.peak_values, show=True)


