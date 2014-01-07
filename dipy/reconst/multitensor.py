import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
from lmfit import minimize, Parameters
from dipy.sims.voxel import multi_tensor

class MultiTensorModel():

    def __init__(self, gtab, iso = False, maxN = 2, selectType = 0):

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        self.iso = iso
        self.maxN = maxN
        self.selectType = selectType

        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0


    @multi_voxel_fit
    def fit(self, data):

    ...

    return MultiTensorFit(self, coef)

class MultiTensorFit():

    def __init__(self, model, coef):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        coef : 2d ndarray,
            multitensor coefficients Nx6
        """

        self.model = model
        self._coef = coef
        self.gtab = model.gtab
        self.N = coef.shape[0]

    @property
    def shore_coeff(self):
        """The SHORE coefficients
        """
        return self._shore_co
