import numpy as np
from dipy.reconst.freewater_scipyBFGSb import FreeWaterTensorModel as fwModel
from dipy.viz import fvtk
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti


fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()
data = img.get_data()
data_small = data[10:40, 22, 10:40]
del data


sphere = get_sphere('symmetric724')


dti_wls = dti.TensorModel(gtab)
fit_wls = dti_wls.fit(data_small)

fa1 = fit_wls.fa
evals1 = fit_wls.evals
evecs1 = fit_wls.evecs
cfa1 = dti.color_fa(fa1, evecs1)


ren = fvtk.ren()
fvtk.add(ren, fvtk.tensor(evals1, evecs1, cfa1, sphere))
fvtk.show()
