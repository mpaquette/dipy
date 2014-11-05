from __future__ import division, print_function

from dipy.viz import fvtk

import numpy as np

import os

def screenshot_odf(odf, sphere, filename, show=False):
    """Takes a screenshot of the odfs, saved as filename.png"""

    if ".png" not in filename:
        filename += '.png'

    ren = fvtk.ren()
    fodf_spheres = fvtk.sphere_funcs(odf, sphere, scale=1.8, norm=True)
    fvtk.add(ren, fodf_spheres)
 #   fvtk.add(ren, fvtk.axes())

    # fodf_spheres.RotateZ(90)
    fodf_spheres.RotateX(90)
    fodf_spheres.RotateY(90)

    if show:
        fvtk.show(ren, size=(1000, 1000))

    fvtk.record(ren, out_path=filename, size=(1000, 1000))
    print('Saved illustration as', filename)


def screenshot_peaks(peaks_dirs, filename, peaks_values=None, show=False):
    """Takes a screenshot of the peaks, saved as filename.png"""

    if ".png" not in filename:
        filename += '.png'

    print(peaks_dirs.shape)

    if peaks_values is None:
        if peaks_dirs.ndim == 4:
            peaks_dirs = peaks_dirs[..., None, :]
        peaks_values = np.ones(peaks_dirs.shape[:-1])

    ren = fvtk.ren()
    fodf_peaks = fvtk.peaks(peaks_dirs, peaks_values, scale=2.8)
    fvtk.add(ren, fodf_peaks)
  #  fvtk.add(ren, fvtk.axes())

    # fodf_peaks.RotateZ(90)
    fodf_peaks.RotateX(90)
    fodf_peaks.RotateY(90)

    if show:
        fvtk.show(ren, size=(1000, 1000))

    fvtk.record(ren, out_path=filename, size=(1000, 1000))
    i = 0
    while (os.path.getsize(filename) < 4000) and i < 20:
        fvtk.record(ren, out_path=filename, size=(1000, 1000))
        i += 1
    print('Saved illustration as', filename)