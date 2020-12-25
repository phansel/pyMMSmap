

from sys import *

data_dir = "/home/paul/data/maps/python/"

if (len(sys.argv) < 4) or (sys.argv[1] is "--h"):
    print("Datatype: fpi_ni, exb, fgm, edp, swd.")
    print("Trange: YYYY-MM-DD/HH:MM:SS,YYYY-MM-DD/HH:MM:SS")
    print("Maptype: vector, flat, ")
    raise IOError("You must pass datatype,"+
                    " trange, and maptype.")

if len(sys.argv) > 4:
    raise IOError("Too many command line args.")



script, datatype, trange, maptype = sys.argv

# time format:
# "YYYY-MM-DD/HH:MM:SS,YYYY-MM-DD/HH:MM:SS"
#  012345678901234567890123456789012345678
date1 = trange[0:10]
date2 = trange[20:30]
years = (date1[0:4],date2[0:4])
months = (date1[5:7],date2[5:7])
days = (date1[8:10],date2[8:10])
map_dir_str = years[0]+months[0]+days[0]+
              years[1]+months[1]+days[1] + '/'

filename = data_dir + '/' + datatype + '/' +
           map_dir_str + "mapdata.h5"



map,bins,datatype = readHDF5(filename)
