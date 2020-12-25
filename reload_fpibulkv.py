# /usr/bin/python3

from mms_map_pyrewrite import *
from mms_map_functions import *
import os


datadir = "/home/paul/data/maps/python/fpi_bulkv/20158-20203_2020-06-20-02-41/"
map,_bins,datatype,coordsx,coordsz = readHDF5(datadir+"/map.h5")

fpi_bulkv_bins = np.linspace(-1.5e3,1.5e3,30,dtype='float32')
bins = fpi_bulkv_bins
print("Plotting to " + datadir)
plot_fpi_bulkv_mean(map,bins,coordsx,datadir,False)
plot_fpi_bulkv_mode(map,bins,coordsx,datadir,False)

