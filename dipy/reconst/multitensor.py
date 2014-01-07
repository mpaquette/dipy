import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
from lmfit import minimize, Parameters
from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import gradient_table

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

        # model selection
        if self.selectType == 0:
            N = self.maxN

        params = build_param(self.bvals, self.bvecs, lam1_init = 1.7e-3, lam2_init = 0.3e-3, ang1_init = 0, ang2_init = 0, N = N, iso = self.iso)


        def residual(params, data):
            bs = params['bvals'].value
            gs = params['gvals'].value
            gtab = gradient_table(bs,gs)

            mevals, angles, fractions = params2tensor(params, N, iso)

            model, _ = multi_tensor(gtab, mevals, S0 = 1, angles=angles,
                             fractions=fractions, snr=None)
            return data - model
    
        argmin = minimize(residual, params, method = 'leastsq', args = (data))

        coef = params2tensor(params, N, iso)

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



def build_param(bs, gs, lam1_init, lam2_init, ang1_init, ang2_init, N, iso):
    params = Parameters()

    params.add('bvals', value=bs, vary = False)
    params.add('gvals', value=gs, vary = False)
    
    for n in range(N):
        param.add('lam1{}'.format(n), value = lam1_init, min = 1e-10)
        param.add('lam2{}'.format(n), value = lam2_init, min = 1e-10)
        param.add('ang1{}'.format(n), value = ang1_init, min = 0, max = 90)
        param.add('ang2{}'.format(n), value = ang2_init, min = 0, max = 360)
        param.add('frac{}'.format(n), value = 1/float(N), min = 0, max = 1)

    if iso:
        param.add('lam', value = lam1_init, min = 1e-10)
        param.add('frac', value = 0.5, min = 0, max = 1)

    return params

def params2tensor(params, N, iso)

    mevals = np.zeros((N + iso, 3))
    angles = []
    fractions = []

    for n in range(N):
        mevals[n,0] = params['lam1{}'.format(n)].value
        mevals[n,1] = params['lam2{}'.format(n)].value
        mevals[n,2] = params['lam2{}'.format(n)].value

        a1 = params['ang1{}'.format(n)].value
        a2 = params['ang2{}'.format(n)].value
        angles.append((a1,a2))

        # fractions.append(np.exp(params['frac{}'.format(n)].value))
        fractions.append(params['frac{}'.format(n)].value)

    fractions /= sum(fractions)

    if iso:
        mevals[N,0] = params['lam'].value
        mevals[N,1] = params['lam'].value
        mevals[N,2] = params['lam'].value

        angles.append((0, 0))

        fractions *= (1 - params['frac'].value)
        fractions.append(params['frac'].value)

    fractions *= 100

return (mevals, angles, fractions)

