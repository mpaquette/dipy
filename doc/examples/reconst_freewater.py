import numpy as np
from dipy.reconst.freewater_scipyBFGSb import FreeWaterTensorModel as fwModel
from dipy.viz import fvtk
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

from time import time


fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()
data = img.get_data()
data_small = data[10:40, 22, None, 10:40]
# data_small = data[20:30, 22, None, 20:30]
# data_small = data[30:33, 22, None, 20:26]
del data
del img

# Normalization
data_small = data_small / data_small[..., 0, None]


sphere = get_sphere('symmetric724')


#DTI WLS
started = time()
dti_wls = dti.TensorModel(gtab)
fit_wls = dti_wls.fit(data_small)
finished = time()
print('DTI WLS elapsed = {} seconds'.format(finished - started))

fa_dti = fit_wls.fa
evals_dti = fit_wls.evals
evecs_dti = fit_wls.evecs
cfa_dti = dti.color_fa(fa_dti, evecs_dti)





started = time()
fw_model = fwModel(gtab, init_type = 2, init_N = [3, 3, 3, 6, 6, 5])
# fw_model = fwModel(gtab, init_type = 2, init_N = [3, 3, 3, 20, 20, 20])
fw_fit = fw_model.fit(data_small)
finished = time()
print('FW init = 2 elapsed = {} seconds'.format(finished - started))



evecs_fw = fw_fit.prolate_evecs()
evals_fw = fw_fit.prolate_evals()
fwf_fw = fw_fit.free_water_fraction()
cfa_fw = dti.color_fa(1 - fwf_fw, evecs_fw)

fitQ_fw = fw_fit.fitQual







print('DTI WLS')
ren = fvtk.ren()
tt = fvtk.tensor(evals_dti, evecs_dti, cfa_dti, sphere)
tt.RotateX(-90)
tt.RotateZ(180)
fvtk.add(ren, tt)
fvtk.show(ren)



print('FW init = 2')
ren = fvtk.ren()
tt = fvtk.tensor(evals_fw, evecs_fw, cfa_fw, sphere)
tt.RotateX(-90)
tt.RotateZ(180)
fvtk.add(ren, tt)
fvtk.show(ren)




import pylab as pl

pl.figure()
pl.imshow((1 - fwf_fw[::-1,0,::-1]).T, interpolation = 'nearest')
pl.colorbar()
pl.title('Anisotropic compartement volume fraction for FWmodel init=2')
pl.show()



pl.figure()
pl.imshow((fitQ_fw[::-1,0,::-1]).T, interpolation = 'nearest')
pl.colorbar()
pl.title('LS residual of FWmodel init=2')
pl.show()



