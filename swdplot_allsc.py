import pyMMSmap as pm
import os
import sys
import datetime

nowstr=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

try:
    y1 = sys.argv[1]
    m1 = sys.argv[2]
    y2 = sys.argv[3]
    m2 = sys.argv[4]

except IndexError:
    print("Missing year1 month1 year2 month2, setting to default")
    y1 = 2015 
    m1 = 8
    y2 = 2020
    m2 = 3

if ((y1 == y2) and (m2 < m1)) or (y1 > y2):
    raise ValueError("Time must flow forwards. y1m1 < y2m2")

try:
    dirdat = "/home/paul/data/maps/python/swd/allsc/" + str(y1)+str(m1)+ "-" +str(y2)+str(m2) +"_" + nowstr + "/"
    os.system("mkdir -p "+dirdat+"new/")
    print("Writing to " + dirdat)
except:
    raise ValueError("Failed to create data output directory.")

map,bins = pm.mapMultipleMonthsAllSC('swd',y1,y2,m1,m2,imagedim)
pm.writeHDF5(dirdat+"map.h5",map,'swd',bins, coordsx, coordsz)
pm.plot_swd_mean_extras(map,coordsx,dirdat+"/new/",True,False,"inferno","gse")
