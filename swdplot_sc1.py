from mms_map_functions import *
from mms_map_pyrewrite import *
import os

import datetime

nowstr=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

coord_system = "gse"


y1 = 2015
m1 = 12

y2 = 2019
m2 = 6

map,bins = mapMultipleMonths(1, 'swd',y1,y2,m1,m2,imagedim)
dirdat = "/home/paul/data/maps/python/swd/somespacecraft/sc1/" + str(y1)+str(m1)+ \
                "-" +str(y2)+str(m2) +"_" + nowstr + coord_system + "/"
os.system("mkdir -p "+dirdat)
print("Writing to " + dirdat)

writeHDF5(dirdat+"map.h5",map,'swd',bins, coordsx, coordsz)
plot_swd_mean2(map,coordsx,dirdat,True,False, "inferno", coord_system)
