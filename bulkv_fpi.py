from mms_map_functions import *
from mms_map_pyrewrite import *
import os

import datetime

nowstr=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

y1 = 2015
y2 = 2020
m1 = 8
m2 = 3


# five minutes
#desired_pixel_dim = np.array([6371.,6371.,6371.*5])

#maxdims = [2e5, 2e5, 2e5]
#mindims = [-2e5, -2e5, -2e5]

#rangex = [mindims[0],maxdims[0]]
#rangey = [mindims[1],maxdims[1]]
#rangez = [mindims[2],maxdims[2]]

#totaldims = np.subtract(maxdims, mindims)

#npixels = np.round(totaldims / desired_pixel_dim, decimals=0)
#print(npixels)
#imagedim = np.int64(npixels + 5)

# definition of spatial bins!
coordsx = np.linspace(start=mindims[0],stop=maxdims[0],num=imagedim[0],
                   dtype=np.float64)
coordsz = np.linspace(start=mindims[0],stop=maxdims[0],num=imagedim[2],
                   dtype=np.float64)


map,bins = mapMultipleMonths(1,'fpi_bulkv',y1,y2,m1,m2,imagedim)
datadir = "/home/paul/data/maps/python/fpi_bulkv/allsc/" + str(y1)+str(m1)+ "-" +str(y2)+str(m2) +"_" + nowstr + "/"

os.system("mkdir -p " + datadir)

print("writing to " + datadir)

writeHDF5(datadir+"map.h5",map,"fpi_bulkv",bins,coordsx,coordsz)
plot_fpi_ni_mean(map,bins,coordsx,datadir,False, True)
plot_fpi_bulkv_median(map,bins,coordsx,datadir,False,True)
