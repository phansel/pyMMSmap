# do not use this
print("Do not use this function!")
ValueError("Nope")
import pyMMSmap
# imagedim has to be a 3x1 vector!
map,bins = mapMultipleMonths('1','fpi_ni',2015,2018,5,5,imagedim)
writeHDF5("/home/paul/data/maps/python/fpi_ni/201505-201805/map.h5",map,"fpi_ni",bins)
plot_fpi_ni_mean(map,bins,coordsx,"/home/paul/data/maps/python/fpi_ni/201505-201805/",False)
plot_fpi_ni_median(map,bins,coordsx,"/home/paul/data/maps/python/fpi_ni/201505-201805/",False)
