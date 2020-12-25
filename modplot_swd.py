# Modifies SWD maps to show statistical bow shock and Earth

import sysdd
import pyMMSmap as pm
dirdat = sys.argv[1]
dmap,bins,datatype,coordsx,coordsz = pm.readHDF5(dirdat+"/map.h5")
pm.plot_swd_mean_extras(dmap,coordsx,dirdat+"/newer/",True,False,False,"inferno","gsm")
