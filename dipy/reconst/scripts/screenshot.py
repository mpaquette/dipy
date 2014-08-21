from __future__ import division, print_function

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