# map_vec.py

from idlpy import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#print(IDL.run('print, "hello from IDL."'))
#print(IDL.run('map_gen_vtk'))
# Credits to spacepy/astropy for help

import csv

from astropy import constants as const

plt.ion()
# using matplotlib quiver
filedir = '/home/paul/data/maps/exb/2015-10-02_2018-06-15/csvs/'

xv = np.genfromtxt(filedir+'xvel_avg.csv',delimiter=',')
yv = np.genfromtxt(filedir+'yvel_avg.csv',delimiter=',')
x = np.genfromtxt(filedir+'coordsx.csv',delimiter=',')
y = np.genfromtxt(filedir+'coordsy.csv',delimiter=',')

x /= (const.R_earth.value/1000.0) # convert m to km
y /= (const.R_earth.value/1000.0) # convert m to km

cm = np.hypot(xv,yv) / 1000.

plt.figure()
plt.figaspect(1.0)
plt.title('Velocity plot for ExB 2015-2018')
q = plt.quiver(x,y,xv,yv, cm)
cbb = plt.colorbar()
plt.show()

##
# >>> plt.figure()
# <matplotlib.figure.Figure object at 0x7fac43427fd0>
# >>> plt.figaspect(1.0)
# array([4.8, 4.8])
# >>> plt.title("lol")
# <matplotlib.text.Text object at 0x7fac3e98dbe0>
# >>> q = plt.quiver(x,y,xv,yv,cm)
# >>> cb = plt.colorbar()
# >>> cm = np.hypot(xv,yv)
# >>>
# >>> plt.figure()
# <matplotlib.figure.Figure object at 0x7fac3e940198>
# >>> plt.figaspect(1.0)
# array([4.8, 4.8])
# >>> plt.title('Velocity plot for ExB 2015-2018')
# <matplotlib.text.Text object at 0x7fac3eb89208>
# >>> q = plt.quiver(x,y,xv,yv, cm)
# >>> cb = plt.colorbar()
# >>> cm = np.hypot(xv,yv) / 1000.
# >>> plt.figure()
# <matplotlib.figure.Figure object at 0x7fac3e98a080>
# >>> plt.figaspect(1.0)
# array([4.8, 4.8])
# >>> plt.title('Velocity plot for ExB 2015-2018')
# <matplotlib.text.Text object at 0x7fac3e963b00>
# >>> q = plt.quiver(x,y,xv,yv, cm)
# >>> cbb = plt.colorbar()
# >>> datadir = '/home/paul/data/maps/swd/2015-10-02_2018-06-15/swdcsvs/'
# >>> coordsx = np.genfromtxt(datadir+'coordsx.csv',delimiter=',')
# >>>
# >>> bin1 = np.genfromtxt(datadir+'bin1.csv',delimiter=',')
# >>> coordsx /= (const.R_earth.value/1000.0) # convert m to km
# >>> bin1plt = plt.imshow(bin1, extent=[coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]],origin='lower')
# >>> plt.show()

##
