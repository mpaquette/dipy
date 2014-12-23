import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
# from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.core.gradients import gradient_table
from scipy.optimize import fmin_l_bfgs_b as l_bfgs_b

from dipy.core.geometry import sphere2cart
from dipy.sims.voxel import all_tensor_evecs


class FreeWaterTensorModel():

    def __init__(self, gtab, init_type = 0, init_N = [10, 10, 10, 10, 10]]):

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        
        self.init_type = init_type
        self.init_N = init_N

        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0

    @multi_voxel_fit
    def fit(self, data):

        def _model(gtab, param):
            # Simulated Q-space signal with a prolate and an isotropic tensor.

            S_prolate, stick = _single_tensor_prolate(gtab, param[1:5])

            S_iso = _isotropic_tensor(gtab, param[0])

            S = param[5] * S_iso + (1 - param[5]) * S_prolate

            return S, stick

        def residual(param):
            model, _ = _model(self.gtab, param)
            return np.sum((data - model)**2)

        def bounds():
            bounds = []
            
            # isotropic tensor D in [eps, 1 - eps] * DfreeWater
            bounds += [(1.0e-10, 1.0 - 1.0e-10)]

            # prolate tensor eval lambda_1 in [eps, 1 - eps] * DfreeWater
            bounds += [(1.0e-10, 1.0 - 1.0e-10)]

            # prolate tensor eval lambda_2_3 in [eps, 1 - eps] * lambda_1
            bounds += [(1.0e-10, 1.0 - 1.0e-10)]

            # angles within the half-sphere
            bounds += [(0.0, 180.0)]
            bounds += [(0.0, 180.0)]

            #free water fraction in [eps, 1 - eps]
            bounds += [(1e-10, 1 - 1e-10)]

            return bounds


        if self.init_type == 0:
            # Fixed parameters initialization
            x0 = np.array([0.8, 0.8, 0.2, 90.0, 90.0, 0.2])

        elif self.init_type == 1:
            # loop over a grid of configuration
            x0s = grid_search(N = self.init_N, bounds = bounds())

            best_x0 = np.array([0.8, 0.8, 0.2, 90.0, 90.0, 0.2])
            best_res = residual(best_x0)

            for x in x0s:
                res = residual(x)
                if res < best_res:
                    best_res = res
                    best_x0 = x

            x0 = best_x0

        else :
            raise NotImplementedError, 'Initialization type = {} Not Implemented'.format(self.init_type)

        param, fitQual, _ = l_bfgs_b(residual, 
                            x0 = x0, 
                            fprime=None, 
                            args=(), 
                            approx_grad=True,  
                            bounds=bounds(), 
                            m=10, 
                            factr=1e0, 
                            pgtol=1e-16, 
                            epsilon=1e-8, 
                            iprint=1, 
                            # iprint=-1, 
                            maxfun=1000, 
                            maxiter=1000, 
                            disp=None, 
                            callback=None)

        est_data, stick = _model(self.gtab, param)

        return FreeWaterTensorFit(self, param, fitQual, est_data, stick)

class FreeWaterTensorFit():
    def __init__(self, model, param, fitQual, est_data, stick):
        self.model = model
        self.gtab = model.gtab

        self.init_type = model.init_type
        self.init_N = model.init_N

        self.param = param
        self.fitQual = fitQual
        self.dir = stick
        self.est_data = est_data

    def dir(self):
        return self.dir

    def fitQual(self):
        return self.fitQual

    def est_data(self):
        return self.est_data

    def param(self):
        return self.param

    def prolate_evals(self):
        Dfree = 0.002299
        lam1 = self.param[1] * Dfree
        lam2 = self.param[2] * lam1
        return np.array([lam1, lam2, lam2])

    def iso_evals(self):
        Dfree = 0.002299
        lam1 = self.param[0] * Dfree
        return np.array([lam1, lam1, lam1])

    def angles(self):
        return self.param[3:5]

    def free_water_fraction(self):
        return self.param[5]


def _single_tensor_prolate(gtab, param):
    # Simulated Q-space signal with a single prolate tensor.

    # Free Water  mm^2/s
    Dfree = 0.002299

    lam1 = param[0] * Dfree
    lam2 = param[1] * lam1
    evals = np.array([lam1, lam2, lam2])

    sticks = sphere2cart(1, np.deg2rad(param[2]), np.deg2rad(param[3]))
    sticks = np.array(sticks)
    evecs=all_tensor_evecs(sticks).T

    out_shape = gtab.bvecs.shape[:gtab.bvecs.ndim - 1]
    gradients = gtab.bvecs.reshape(-1, 3)

    R = np.asarray(evecs)
    S = np.zeros(len(gradients))
    D = np.dot(np.dot(R, np.diag(evals)), R.T)

    for (i, g) in enumerate(gradients):
        S[i] = np.exp(-gtab.bvals[i] * np.dot(np.dot(g.T, D), g))

    return S.reshape(out_shape), sticks



def _isotropic_tensor(gtab, param):
    # Simulated Q-space signal with a single isotropic tensor.
    Dfree = 0.002299

    D = param * Dfree

    out_shape = gtab.bvecs.shape[:gtab.bvecs.ndim - 1]
    gradients = gtab.bvecs.reshape(-1, 3)

    S = np.zeros(len(gradients))

    for (i, g) in enumerate(gradients):
        S[i] = np.exp(-gtab.bvals[i] * D)

    return S.reshape(out_shape)


def grid_search(N, bounds):

    x0s = []
    for x_iso in np.linspace(bounds[0][0], bounds[0][1], N[0]):
        for x_lam1 in np.linspace(bounds[1][0], bounds[1][1], N[1]):
            for x_lam2 in np.linspace(bounds[2][0], bounds[2][1], N[2]):
                for ang1 in np.linspace(bounds[3][0], bounds[3][1], N[3]):
                    for ang2 in np.linspace(bounds[4][0], bounds[4][1], N[4]):
                        for fw_f in np.linspace(bounds[5][0], bounds[5][1], N[5]):
                            x0s.append([x_iso, x_lam1, x_lam2, ang1, ang2, fw_f])

    return np.array(x0s)
