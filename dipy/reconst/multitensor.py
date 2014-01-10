import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
from lmfit import minimize, Parameters
from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import gradient_table
from lmfit import report_errors

class MultiTensorModel():

    def __init__(self, gtab, iso = False, maxN = 2, selectType = 0, fixMevals = False, fixFractions = False):

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        self.iso = iso
        self.maxN = maxN
        self.selectType = selectType
        self.fixFractions = fixFractions
        self.fixMevals = fixMevals

        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0


    @multi_voxel_fit
    def fit(self, data):

        # model selection
        if self.selectType == 0:
            N = self.maxN
        else:
            raise NotImplementedError, 'Model selection type {} not there'.format(sef.selectType)

        params = build_param(self.bvals, self.bvecs, lam1_init = 1.7e-3, lam2_init = 0.3e-3, ang1_init = 90, ang2_init = 0, N = N, iso = self.iso, fixMevals = self.fixMevals, fixFractions = self.fixFractions)


        # def residual(params, data):
        def residual(params):
            bs = params['bvals'].value
            gs = params['gvals'].value
            gtab = gradient_table(bs,gs)

            mevals, angles, fractions = params2tensor(params, N, self.iso)

            model, _ = multi_tensor(gtab, mevals, S0 = 1, angles=angles,
                             fractions=fractions, snr=None)
            return data - model
    
        # argmin = minimize(residual, params, method = 'leastsq', args = (data))
        # argmin = minimize(residual, params, method = 'leastsq')
        argmin = minimize(residual, params, method = 'leastsq', xtol = 1e-16, ftol = 1e-16, maxfev = int(1e9))
        print(argmin.message)
        coef = params2tensor(params, N, self.iso)

        return MultiTensorFit(self, coef)

class MultiTensorFit():

    def __init__(self, model, coef):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        coef : (mevals, angles, fractions),
            multitensor coefficients
        """

        self.model = model
        self._mevals = coef[0]
        self._angles = coef[1]
        self._fractions = coef[2]
        self.gtab = model.gtab
        self.iso = model.iso
        self._N = coef[0].shape[0] - self.iso

    @property
    def multitensor_mevals(self):
        return self._mevals

    @property
    def multitensor_angles(self):
        return self._angles

    @property
    def multitensor_fractions(self):
        return self._fractions

    @property
    def multitensor_N(self):
        return self._N



def build_param(bs, gs, lam1_init, lam2_init, ang1_init, ang2_init, N, iso, fixMevals, fixFractions):
    params = Parameters()

    params.add('bvals', value=bs, vary = False)
    params.add('gvals', value=gs, vary = False)
    
    for n in range(N):
        params.add('lam1{}'.format(n), value = lam1_init, min = 1e-10, vary = not fixMevals)
        params.add('lam2{}'.format(n), value = lam2_init, min = 1e-10, vary = not fixMevals)
        params.add('ang1{}'.format(n), value = ang1_init, min = 0, max = 90)
        # params.add('ang2{}'.format(n), value = ang2_init, min = 0, max = 360)
        params.add('ang2{}'.format(n), value = n*180/np.float64(N), min = 0, max = 360)
        params.add('frac{}'.format(n), value = 1/np.float64(N), min = 0, max = 1, vary = not fixFractions)

    if iso:
        params.add('lam', value = lam1_init, min = 1e-10)
        params.add('frac', value = 0.5, min = 0, max = 1)

    return params

def params2tensor(params, N, iso):

    mevals = np.zeros((N + iso, 3))
    angles = []
    fractions = np.zeros(N + iso)

    for n in range(N):
        mevals[n,0] = params['lam1{}'.format(n)].value
        mevals[n,1] = params['lam2{}'.format(n)].value
        mevals[n,2] = params['lam2{}'.format(n)].value

        a1 = params['ang1{}'.format(n)].value
        a2 = params['ang2{}'.format(n)].value
        angles.append((a1,a2))

        fractions[n] = params['frac{}'.format(n)].value

    fractions /= fractions.sum()

    if iso:
        mevals[N,0] = params['lam'].value
        mevals[N,1] = params['lam'].value
        mevals[N,2] = params['lam'].value

        angles.append((0, 0))

        fractions *= (1 - params['frac'].value)
        fractions[N] = params['frac'].value

    fractions *= 100

    return (mevals, angles, fractions)

