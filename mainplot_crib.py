#/usr/bin/python
# script to load overview plot for SWD paper

# modified 2021/1/5 to run in python rather than IDL (idl is garbage)

import pyspedas
from pytplot import *
from pyspedas.mms import *


probe = "1"

trange=['2016-08-09 09:13:00','2016-08-09 09:47:00']
trange2 = [1470734460.0, 1470734460.05]
trange3 = [1470734460.05, 1470734460.2]
#trange3=['2016-08-09 09:17:00','2016-08-09 09:38:00']
#trange5=['2016-08-09 09:19:30','2016-08-09 09:19:31']

#trange = trange2

tplot_options('wsize', [850,1400])

# plot main figure

tplot_options('x_range', pyspedas.time_double(trange))

#trange=['2016-08-09 09:20:00','2016-08-09 09:30:00']

#mms_load_edp, probe='1', trange=trange, data_rate='fast', datatype='dce', level='l2'
mms_edp = edp(trange=trange, probe=probe, data_rate='fast', datatype='dce', level='l2')
#mms_edp_brst = edp(trange=trange, probe=probe, data_rate='brst', datatype='dce', level='l2')
edp_name = "mms" + str(probe) + "_edp_dce_gse_fast_l2"
options(edp_name, 'ytitle', " Epara [mV/m]")
brst_edp_name1 = "mms" + str(probe) + "_edp_dce_par_epar_brst_l2"
options(brst_edp_name1, 'ytitle', "MMS" + str(probe)+ " Burst E|| [mV/m]")

#mms_load_dsp, probe=probe, trange=trange, data_rate='fast', datatype='swd', level='l2'
mms_dsp = dsp(trange=trange, data_rate='fast', datatype='swd', level='l2')
swd = "mms1_dsp_swd_E12_Counts"
options(swd, 'spec', False)
options(swd, 'ytitle', "SWD [1/s]")
options(swd, 'labflag', -1)
options(swd, "legend_names", ['0.5-3 mV/m', '3-12 mV/m', '12-50 mV/m', '50+ mV/m'])
ylim(swd, 0, 170)


#mms_load_fpi, probe=probe, trange=trange, data_rate='brst', datatype='dis-moms'
mms_fpi1 = fpi(trange=trange, probe=probe, data_rate='brst', datatype='dis-moms')
#mms_load_fpi, probe=probe, trange=trange, data_rate='fast', datatype='dis-moms'
mms_fpi2 = fpi(trange=trange, probe=probe, data_rate='fast', datatype='dis-moms')
#dis_bulkv = "mms" + str(probe) + "_dis_bulkv_gse_brst"
dis_bulkv = "mms" + str(probe) + "_dis_bulkv_gse_fast"
dis_energy = "mms" + str(probe) + "_dis_energyspectr_omni_brst"

#dis_bulkv = "mms1_dis_bulkv_gse_fast"
#dis_ni = "mms1_dis_numberdensity_fast"
#dis_energy = "mms1_dis_energyspectr_omni_fast"
dis_ni = "mms" + str(probe) + "_dis_numberdensity_fast"

options(dis_bulkv, 'ytitle', "Ion V GSE \  [km/s]")

options(dis_ni, 'legend_names', ["Ni, ions"])
options(dis_ni, 'ytitle', 'Ions \ Density [cm^-3]')
options(dis_energy, 'spec', True)
options(dis_energy, 'ytitle', 'Ions \ Energy [eV]')
options(dis_energy, 'ztitle', 'MeV/(cm^2 s sr eV)')
ylim(dis_energy, 2.0, 20000.0)

    # DIVIDE Zvalues



# Fast Plasma Investigations instruments for electrons
#mms_load_fpi, probe=probe, trange=trange, data_rate='brst', datatype='des-moms'
mms_fpi3 = fpi(trange=trange, probe=probe, data_rate='brst', datatype='des-moms')
des_energy = "mms" + str(probe) + "_des_energyspectr_omni_brst"
des_ni = "mms" + str(probe) + "_des_numberdensity_brst"
# Electron velocity is noisy in low density
options(des_ni, 'ytitle', 'Electrons \ Density [1/cm3]')
options(des_energy, 'spec', True)
options(des_energy, 'ytitle', 'Electrons \  Energy [eV]')
options(des_energy, 'ztitle', 'MeV/(cm^2 s sr eV)')



# Fluxgate magnetometer
#mms_load_fgm, probe=probe, trange=trange, data_rate='srvy'
mms_fgm = fgm(trange=trange, data_rate='srvy', probe=probe)
fgm_name = "mms" + str(probe) + "_fgm_b_gse_srvy_l2"
options(fgm_name, "legend_names", ['Bx GSE', 'By GSE', 'Bz GSE'])
options(fgm_name, 'ytitle', 'B [nT]')
ylim(fgm_name, -150, 150)


# MEC ephemeris data (GSE)
#mms_load_mec, probe=probe, trange=trange, data_rate='srvy'
mms_mec1 = mec(trange=trange, probe=1, data_rate='srvy')
mec_name = "mms" + str(probe) + "_mec_r_gse"

options(mec_name, "legend_names", ['X GSE', 'Y GSE', 'Z GSE'])
options(mec_name, 'ytitle', "Position [km]")
options(mec_name, 'labflag', '-1')




#ylim, dis_energy, 2, 10e3
#xlim(trange[0], trange[1])


# remove
#tplot_options('roi', trange2)
#tplot_options('title_size', 14)
#options(swd, 'ylog', True)



tplot([dis_energy, des_energy, dis_ni, swd, dis_bulkv,fgm_name, edp_name],vert_spacing=4)


# Drop vector E field
def plotit():
    mms_brst_edp = edp(trange=trange, probe=probe, data_rate='brst', datatype='dce', level='l2')
    tplot([dis_energy, des_energy, dis_ni, swd, dis_bulkv, brst_edp_name1,fgm_name],vert_spacing=2)



def plotagain():
    trange=trange2
    mainplot_init()
    tplot([dis_energy, des_energy, dis_ni, swd, dis_bulkv, brst_edp_name1,fgm_name],vert_spacing=2)


# Put the small zoom plot below the main plot?


#mms_load_feeps, probe='1', trange=trange, level='l2', data_rate='srvy', datatype='ion'
#feeps_ion = "mms1_epd_feeps_srvy_l2_ion_"

#x = -50649.641930699385
#y = 22563.089427813047
#z = 2600.7478602880660

#x = -7.9500299
#y = 3.5415304
#z = 0.40821657
