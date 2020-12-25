#/usr/bin/python
# script to load fragment of overview plot for SWD paper

import pyspedas
from pytplot import *
from pyspedas.mms import *


probe = "1"


trange_zoom = [1470734360.0, 1470734420.0]
trange_zoom = [1470734380.0, 1470734400.0]
trange_zoom = ['2016-08-09 09:19:46','2016-08-09 09:19:47']
trange_zoom = [1470734386.3, 1470734386.6]
trange = trange_zoom
mms_edp = edp(trange=trange, probe=probe, data_rate='brst', datatype='dce', level='l2')
tplot_options('wsize', [850,300])
brst_edp_name2 = "mms" + str(probe) + "_edp_dce_par_epar_brst_l2"
options(brst_edp_name2, 'ytitle', 'MMS1 Epara \ (mV/m)')
ylim(brst_edp_name2, -30, 30)
tplot(brst_edp_name2)
