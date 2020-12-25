from mms_map_functions import *
from mms_map_pyrewrite import *
# imagedim defined initially, but coordsx is not?
import os

import datetime

nowstr=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

y1 = 2015
y2 = 2020
m1 = 8
m2 = 3

map,bins = mapMultipleMonths(1,'fpi_ni',y1,y2,m1,m2,imagedim)
datadir = "/home/paul/data/maps/python/fpi_ni/allsc/" + str(y1)+str(m1)+ "-" +str(y2)+str(m2) +"_" + nowstr + "/"
os.system("mkdir -p "+datadir)
print("Writing to " + datadir)



writeHDF5(datadir+"map.h5",map,"fpi_ni",bins,coordsx,coordsz)
plot_fpi_ni_mean(map,bins,coordsx,datadir+"xz/",False, True)
plot_fpi_ni_mode(map,bins,coordsx,datadir+"/xz/",False,True)
