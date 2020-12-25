# /usr/bin/python3

from mms_map_pyrewrite import *
from mms_map_functions import *
import os

datadir = "/home/paul/data/maps/python/fpi_ni/20158-20203_2020-06-20-02-30/"
map,_bins,datatype,coordsx,coordsz = readHDF5(datadir+"/map.h5")
bins = np.linspace(-7.,8.,256,dtype='float32')

os.system("mkdir -p "+datadir+"/xz/")
os.system("mkdir -p "+datadir+"/xy/")

plot_fpi_ni_mean(map,bins,coordsx,datadir+"/xz/",False, True)
plot_fpi_ni_mode(map,bins,coordsx,datadir+"/xz/",False,True)

plot_fpi_ni_mean(map,bins,coordsx,datadir+"/xy/",False, False)
plot_fpi_ni_mode(map,bins,coordsx,datadir+"/xy/",False,False)

