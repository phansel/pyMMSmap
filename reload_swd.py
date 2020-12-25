import sys
import pyMMSmap as pm

dirdat = sys.argv[1]

dmap,bins,datatype,coordsx,coordsz = pm.readHDF5(dirdat+"/map.h5")
pm.plot_swd_mean_extras(dmap,coordsx,dirdat+"/new/",True,False,"inferno",'gse')
