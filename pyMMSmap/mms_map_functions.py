#/usr/bin/python3

#from mms_map_pyrewrite import *

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const

# def plot_swd_mean(map, coordsx, plotpath, log, clip, xy_clip, colormap):
#     filenames= ["bin1swdmean.png", "bin2swdmean.png",
#              "bin3swdmean.png","bin4swdmean.png"]
#
#     plottitles=["Bin 1 (0.5-3 mV/m) Average SWD Count",
#                 "Bin 2 (3-12 mV/m) Average SWD Count",
#                 "Bin 3 (12-50 mV/m) Average SWD Count",
#                 "Bin 4 (50+ mV/m) Average SWD Count"]
#
#     # map.shape is (51,51,51,257,4)
#     # (xi,yi,zi,number_in_bin_at_this_countrate,bin)
#     # it is a histogram
#
# #    if xy_clip is True:
# #        # cut the map off above and below some number of Re
# #        n_off_center_z = 3
# #        center_n = int((map.shape)[2] / 2) + 1
# #        center_bins = (center_n - n_off_center_z, center_n + n_off_center_z)
# #        c_b = center_bins # alias
# #        bin1 = map[:,:,c_b[0]:c_b[1],:,0]
# #        bin2 = map[:,:,c_b[0]:c_b[1],:,1]
# #        bin3 = map[:,:,c_b[0]:c_b[1],:,2]
# #        bin4 = map[:,:,c_b[0]:c_b[1],:,3]
# #    else:
# #
#     bin1 = map[:,:,:,:,0]
#     bin2 = map[:,:,:,:,1]
#     bin3 = map[:,:,:,:,2]
#     bin4 = map[:,:,:,:,3]
#
#
#     bins = [bin1,bin2,bin3,bin4]
#     spec = np.linspace(0,256,257,dtype='int')
#     plt.ion()
#     coordsx /= (const.R_earth.value/1000.0) # convert m to km
#     # clipvals = [12.5,2.5,2,.06]
#
#     for bin in range(4):
#         plt.xlabel('Y GSE (Re)')
#         plt.ylabel('X GSE (Re)')
#
#         flatz = np.array(np.sum(bins[bin],2),dtype=np.float64)
#         totmap = np.array(np.sum(bins[bin],(2,3)),dtype=np.float64)
#
#         bintot = np.sum(flatz * spec,2)
#         print(bintot.shape)
#         binavg = bintot / totmap
#         binavg[np.isnan(binavg)] = 0 # replace NAN with 0
#         #if clip is True:
#         #    binavg = np.clip(binavg,None,clipvals[bin])
#
#         if log is True:
#             binavg = np.log(binavg)
#             filenames= ["bin1swdlogmean.png", "bin2swdlogmean.png",
#                         "bin3swdlogmean.png","bin4swdlogmean.png"]
#
#             plottitles=["Bin 1 (0.5-3 mV/m) Log Average SWD Count",
#                         "Bin 2 (3-12 mV/m) Log Average SWD Count",
#                         "Bin 3 (12-50 mV/m) Log Average SWD Count",
#                         "Bin 4 (50+ mV/m) Log Average SWD Count"]
#
#         plt.title(plottitles[bin])
#
#         b = plt.imshow(binavg, extent=[coordsx[0],coordsx[-1],
#                 coordsx[0],coordsx[-1]],origin='lower', cmap=colormap)
#
#         bar = plt.colorbar()
#
#         if log is False:
#             bar.set_label('Average SWD count (1/s)')
#         if log is True:
#             bar.set_label('Average SWD count (log 1/s)')
#
#         b.set_extent([coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]])
#         plt.savefig(plotpath+filenames[bin], dpi=400)
#         plt.clf()
# ##
# ##
# ##
# ##
# ## rewrite to remove all zero entries and maybe reproduce previous







# def plot_swd_mean2(map, coordsx, plotpath, log, clip, xy_clip, colormap,system):
#     filenames= ["bin1swdmean.png", "bin2swdmean.png",
#              "bin3swdmean.png","bin4swdmean.png"]
#
#     plottitles=["Bin 1 (0.5-3 mV/m) Average SWD Count",
#                 "Bin 2 (3-12 mV/m) Average SWD Count",
#                 "Bin 3 (12-50 mV/m) Average SWD Count",
#                 "Bin 4 (50+ mV/m) Average SWD Count"]
#
#     # map.shape is (51,51,51,257,4)
#     # (xi,yi,zi,number_in_bin_at_this_countrate,bin)
#     # it is a histogram
#
# #    if xy_clip is True:
# #        # cut the map off above and below some number of Re
# #        n_off_center_z = 3
# #        center_n = int((map.shape)[2] / 2) + 1
# #        center_bins = (center_n - n_off_center_z, center_n + n_off_center_z)
# #        c_b = center_bins # alias
# #        bin1 = map[:,:,c_b[0]:c_b[1],:,0]
# #        bin2 = map[:,:,c_b[0]:c_b[1],:,1]
# #        bin3 = map[:,:,c_b[0]:c_b[1],:,2]
# #        bin4 = map[:,:,c_b[0]:c_b[1],:,3]
# #    else:
# #
#     ### map = [rx,ry,rz,count,bin]
#     bin1 = map[:,:,:,:,0]
#     bin2 = map[:,:,:,:,1]
#     bin3 = map[:,:,:,:,2]
#     bin4 = map[:,:,:,:,3]
#
#
#     bins = [bin1,bin2,bin3,bin4]
#     spec = np.linspace(0,256,257,dtype='int') # [0,1,2,3,...,255,256,257]
#
#     ## allow plot to be shown without locking up CLI
#     plt.ion()
#     coordsx /= (const.R_earth.value/1000.0) # convert m to km
#     # clipvals = [12.5,2.5,2,.06]
#
#     for bin in range(4):
#         if system=="gse":
#             plt.xlabel('Y GSE (Re)')
#             plt.ylabel('X GSE (Re)')
#         if system=="gsm":
#             plt.xlabel('Y GSM (Re)')
#             plt.ylabel('X GSM (Re)')
#
#
#
#         flatz = np.array(np.sum(bins[bin],2),dtype=np.float64)
#         ## flatz is [[rx,ry,rz,count]...]
#
#         totmap = np.array(np.sum(bins[bin],(2,3)),dtype=np.float64)
#         ## totmap is [[rx,ry]...]
#
#         bintot = np.sum(flatz * spec,2)
#         print(bintot.shape)
#         binavg = bintot / totmap
#         binavg[np.isnan(binavg)] = 0 # replace NAN with 0
#         #if clip is True:
#         #    binavg = np.clip(binavg,None,clipvals[bin])
#
#         if log is True:
#             binavg = np.log(binavg)
#             filenames= ["bin1swdlogmean.png", "bin2swdlogmean.png",
#                         "bin3swdlogmean.png","bin4swdlogmean.png"]
#
#             plottitles=["Bin 1 (0.5-3 mV/m) Log Average SWD Count",
#                         "Bin 2 (3-12 mV/m) Log Average SWD Count",
#                         "Bin 3 (12-50 mV/m) Log Average SWD Count",
#                         "Bin 4 (50+ mV/m) Log Average SWD Count"]
#
#         plt.title(plottitles[bin])
#
#         b = plt.imshow(binavg, extent=[coordsx[0],coordsx[-1],
#                 coordsx[0],coordsx[-1]],origin='lower', cmap=colormap)
#
#         bar = plt.colorbar()
#
#         if log is False:
#             bar.set_label('Average SWD count (1/s)')
#         if log is True:
#             bar.set_label('Average SWD count (log 1/s)')
#
#         b.set_extent([coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]])
#         plt.savefig(plotpath+filenames[bin], dpi=400)
#         plt.clf()
#

# Courtesy of Joe Kington on StackOverflow
from matplotlib.patches import Wedge
def dual_half_circle(center, radius, angle=0, ax=None, colors=('b','k'),
                     **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = Wedge(center, radius, theta1, theta2, fc='teal', **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc='black', **kwargs)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]




def plot_swd_mean_extras(map, coordsx, plotpath, log, clip, colormap,system):
    filenames= ["bin1swdmean.png", "bin2swdmean.png",
             "bin3swdmean.png","bin4swdmean.png"]

    plottitles=["Bin 1 (0.5-3 mV/m) Average SWD Count",
                "Bin 2 (3-12 mV/m) Average SWD Count",
                "Bin 3 (12-50 mV/m) Average SWD Count",
                "Bin 4 (50+ mV/m) Average SWD Count"]

    # map.shape is (51,51,51,257,4)
    # (xi,yi,zi,number_in_bin_at_this_countrate,bin)
    # it is a histogram

    bin1 = map[:,:,:,:,0]
    bin2 = map[:,:,:,:,1]
    bin3 = map[:,:,:,:,2]
    bin4 = map[:,:,:,:,3]

    plt.ion()

    bins = [bin1,bin2,bin3,bin4]
    spec = np.linspace(0,256,257,dtype='int')
    coordsx /= (const.R_earth.value/1000.0) # convert m to km
    coordsx -= 1
    # clipvals = [12.5,2.5,2,.06]
    
    

    for bin in range(4):
        flatz = np.array(np.sum(bins[bin],2),dtype=np.float64)
        totmap = np.array(np.sum(bins[bin],(2,3)),dtype=np.float64)
        
        bintot = np.sum(flatz * spec,2)
        print(bintot.shape)
        binavg = bintot / totmap
        binavg[np.isnan(binavg)] = 0 # replace NAN with 0
        #if clip is True:
        #    binavg = np.clip(binavg,None,clipvals[bin])

        binavg = np.log(binavg)
        filenames= ["bin1swdlogmean.png", "bin2swdlogmean.png",
                    "bin3swdlogmean.png","bin4swdlogmean.png"]
        plottitles=["Bin 1 (0.5-3 mV/m) Log Average SWD Count",
                    "Bin 2 (3-12 mV/m) Log Average SWD Count",
                    "Bin 3 (12-50 mV/m) Log Average SWD Count",
                    "Bin 4 (50+ mV/m) Log Average SWD Count"]

        fig, ax = plt.subplots()

        ax.set_title(plottitles[bin])

        binavg[0,0] = np.log(0.0)
        #print(binavg[0,0])
        #print(binavg[-1,0])
        #print(binavg[0,-1])
        #print(binavg[-1,-1])

        

        image_out = ax.imshow(binavg, extent=[coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]],origin='lower', cmap=colormap)

        if system == "gse":
            ax.set_xlabel('Y GSE (Re)')
            ax.set_ylabel('X GSE (Re)')

        if system == "gsm":
            ax.set_xlabel('Y GSM (Re)')
            ax.set_ylabel('X GSM (Re)')

        bar = plt.colorbar(image_out)
        #bar = fig.colorbar(binavg)

        bar.set_label('Average SWD count (log 1/s)')

        #b.set_extent([coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]])

        # plot Earth

        dual_half_circle((0,0), radius=1, angle=0, ax=ax)

        # plot statistical bow shock
        # s66 parameters
        M_ms = 6.96 # 6.96
        # Bz in nanoTesla
        bz = -15e-9
        # upstream solar wind dynamic pressure Pa
        Dp = 2.5 # 2.5
        # solar wind beta
        beta = 0.03

        a1 = 11.646
        a2 = 0.2016
        a3 = 0.122
        a4 = 6.215
        a5 = 0.578
        a6 = -0.009
        a7 = 0.012

        r0 = (a1 + 8*a3 - 8*a2 + a3*bz)*Dp**(-1/a4)

        spread = 1.2

        theta = np.linspace(-np.pi/spread,np.pi/spread, 100)
        alpha = (a5 + a6*bz)*(1 + a7*Dp)
        r_magnetopause = r0 * (2 / (1 + np.cos(theta))) ** alpha

        # assuming Bz < 0
        #r0 = a1(1+a2*bz)*(1 + a9*beta)*(1+a4((a8-1)*m_ms**2+2)/((a8+1)*m_ms**2))*Dp**(-1/a11)
        #alpha = a5 * (1+a13 * bz) * (1+a7*Dp) * (1 + a10 * np.log(1+beta)))*(1+a14*M_ms)
        a1 = 11.1266
        a2 = 0.0010
        a3 = -0.0005
        a4 = 2.5966
        a5 = 0.8182
        a6 = -0.0170
        a7 = -0.0122
        a8 = 1.3007
        a9 = -0.0049
        a10 = -0.0328
        a11 = 6.047
        a12 = 1.029
        a13 =  0.0231
        a14 = -0.002

        # courtesy of J.K. Chao et al 2002
        eps = a12

        r0 = a1*(1+a3*bz)*(1+a9*beta)*(1+a4*((a8-1)*M_ms**2 + 2)/((a8 + 1)*M_ms**2))*Dp**(-1/a11)
        alpha = a5 * (1 + a6*bz) * (1 + a7*Dp) * (1 + a10 * np.log(1+beta))*(1+a14*M_ms)

        r_bowshock = r0 * ((1 + eps) / (1 + eps*np.cos(theta))) ** alpha

        #convert polar to xy so that it actually plots on top of map image
        # there must be a function for this but I can't find it
        x_bs = r_bowshock * np.sin(theta)
        y_bs = r_bowshock * np.cos(theta)
        x_mp = r_magnetopause * np.sin(theta)
        y_mp = r_magnetopause * np.cos(theta)

        ax.plot(x_bs, y_bs, '--', color='black', linewidth=1.5)
        ax.plot(x_mp, y_mp, '-', color='black', linewidth=1.5)

        plt.xlim(-31,31)
        plt.ylim(-31,31)

        plt.gca().invert_xaxis()

        plt.savefig(plotpath+filenames[bin], dpi=400)
        plt.clf()





# def plot_swd_median(map, coordsx, plotpath, log, colormap):
#     filenames= ["bin1swdmedian.png", "bin2swdmedian.png",
#              "bin3swdmedian.png","bin4swdmedian.png"]
#
#     plottitles=["Bin 1 (0.5-3 mV/m) Median SWD Count",
#                 "Bin 2 (3-12 mV/m) Median SWD Count",
#                 "Bin 3 (12-50 mV/m) Median SWD Count",
#                 "Bin 4 (50+ mV/m) Median SWD Count"]
#
#     bin1 = map[:,:,:,:,0]
#     bin2 = map[:,:,:,:,1]
#     bin3 = map[:,:,:,:,2]
#     bin4 = map[:,:,:,:,3]
#
#     if log is True:
#         print("Log does nothing yet!")
#
#     bins = [bin1,bin2,bin3,bin4]
#     spec = np.linspace(0,256,257,dtype='int')
#     spec_mat = np.tile(spec,(257,257,1))
#     plt.ion()
#     coordsx /= (const.R_earth.value/1000.0) # convert m to km
#
#     for bin in range(4):
#         plt.xlabel('Y GSE (Re)')
#         plt.ylabel('X GSE (Re)')
#         totmap = np.array(np.sum(bins[bin],(2,3)),dtype=np.float64)
#         flatz = np.array(np.sum(bins[bin],2),dtype=np.float64)
#         flatz_nozero = flatz
#         flatz_nozero[:,:,0] = 0
#         median = np.argmax(flatz,axis=2)
#         print(median.shape)
#
#         plt.title(plottitles[bin])
#
#         b = plt.imshow(median, extent=[coordsx[0],coordsx[-1],
#                 coordsx[0],coordsx[-1]],origin='lower', cmap=colormap)
#
#         bar = plt.colorbar()
#         bar.set_label('Median SWD count (1/s)')
#
#         b.set_extent([coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]])
#         plt.savefig(plotpath+filenames[bin], dpi=400)
#         plt.clf()

# def plot_swd_std(map, coordsx, plotpath, log, colormap):
#     filenames= ["bin1swd_std.png", "bin2swd_std.png",
#              "bin3swd_swd.png","bin4swd_std.png"]
#
#     plottitles=["Bin 1 (0.5-3 mV/m) SD of SWD Count",
#                 "Bin 2 (3-12 mV/m) SD of SWD Count",
#                 "Bin 3 (12-50 mV/m) SD of SWD Count",
#                 "Bin 4 (50+ mV/m) SD of SWD Count"]
#
#     bin1 = map[:,:,:,:,0]
#     bin2 = map[:,:,:,:,1]
#     bin3 = map[:,:,:,:,2]
#     bin4 = map[:,:,:,:,3]
#
#     bins = [bin1,bin2,bin3,bin4]
#     spec = np.linspace(0,256,257,dtype='int')
#     spec_mat = np.tile(spec,(257,257,1))
#     plt.ion()
#     coordsx /= (const.R_earth.value/1000.0) # convert m to km
#
#     for bin in range(4):
#         plt.xlabel('Y GSE (Re)')
#         plt.ylabel('X GSE (Re)')
#         totmap = np.array(np.sum(bins[bin],(2,3)),dtype=np.float64)
#         flatz = np.array(np.sum(bins[bin],2),dtype=np.float64)
#         flatz_nozero = flatz
#         flatz_nozero[:,:,0] = 0
#         stddev = np.std(flatz,axis=2)
#         print(stddev.shape)
#
#         plt.title(plottitles[bin])
#
#         b = plt.imshow(stddev, extent=[coordsx[0],coordsx[-1],
#                 coordsx[0],coordsx[-1]],origin='lower', cmap=colormap)
#
#         bar = plt.colorbar()
#         bar.set_label('Deviation in SWD count (1/s)')
#
#         b.set_extent([coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]])
#         plt.savefig(plotpath+filenames[bin], dpi=400)
#         plt.clf()

def plot_fpi_bulkv_mean(map, bins, coordsx, plotpath, log):
    filenames= ["fpi_bulkv_xmean.png","fpi_bulkv_ymean.png"]

    plottitles= ["FPI Mean Ion Bulk Velocity X","FPI Mean Ion Bulk Velocity Y"]

    spec = bins
    plt.ion()
    coords = coordsx/(const.R_earth.value/1000.0) # convert m to km
    coords -= 1
    print(map.shape)
    flatz = np.array(np.sum(map,(2,5)),dtype=np.float64)
    n_samp_tot = np.array(np.sum(map,(2,3,4,5)),dtype=np.float64)
    xvel_arr = np.sum(flatz,3)
    yvel_arr = np.sum(flatz,2)
    xvel_avg = np.sum(xvel_arr * spec, 2) / n_samp_tot
    yvel_avg = np.sum(yvel_arr * spec, 2) / n_samp_tot

    if log is True:
        raise ValueError("Can't have log of a negative value!")
        xvel_avg = np.log(xvel_avg)
        yvel_avg = np.log(yvel_avg)
        cbar_names = ['Log mean FPI ion bulk velocity X (km/s)',
                      'Log mean FPI ion bulk velocity Y (km/s)']
        filenames= ["fpi_bulkv_xlogmean.png","fpi_bulkv_ylogmean.png"]

    if log is False:
        cbar_names = ['Mean ion bulk velocity X (km/s)',
                      'Mean ion bulk velocity Y (km/s)']


    vel_avg = [xvel_avg,yvel_avg]

    for num in range(0,2):
        fig, ax = plt.subplots()

        ax.set_title(plottitles[num])

        ax.set_xlabel('Y GSE (Re)')
        ax.set_ylabel('X GSE (Re)')
        image_out = ax.imshow(vel_avg[num], extent=[coords[0],coords[-1],
                coords[0],coords[-1]],origin='lower', cmap='seismic')
        bar = plt.colorbar(image_out)
        bar.set_label(cbar_names[num])


        # plot Earth
        dual_half_circle((0,0), radius=1, angle=0, ax=ax)
        # plot statistical bow shock
        # s66 parameters
        M_ms = 6.96 # 6.96
        # Bz in nanoTesla
        bz = -15e-9
        # upstream solar wind dynamic pressure Pa
        Dp = 2.5 # 2.5
        # solar wind beta
        beta = 0.03
        a1 = 11.646
        a2 = 0.2016
        a3 = 0.122
        a4 = 6.215
        a5 = 0.578
        a6 = -0.009
        a7 = 0.012
        r0 = (a1 + 8*a3 - 8*a2 + a3*bz)*Dp**(-1/a4)
        spread = 1.2
        theta = np.linspace(-np.pi/spread,np.pi/spread, 100)
        alpha = (a5 + a6*bz)*(1 + a7*Dp)
        r_magnetopause = r0 * (2 / (1 + np.cos(theta))) ** alpha
        # assuming Bz < 0
        #r0 = a1(1+a2*bz)*(1 + a9*beta)*(1+a4((a8-1)*m_ms**2+2)/((a8+1)*m_ms**2))*Dp**(-1/a11)
        #alpha = a5 * (1+a13 * bz) * (1+a7*Dp) * (1 + a10 * np.log(1+beta)))*(1+a14*M_ms)
        a1 = 11.1266
        a2 = 0.0010
        a3 = -0.0005
        a4 = 2.5966
        a5 = 0.8182
        a6 = -0.0170
        a7 = -0.0122
        a8 = 1.3007
        a9 = -0.0049
        a10 = -0.0328
        a11 = 6.047
        a12 = 1.029
        a13 =  0.0231
        a14 = -0.002
        # courtesy of J.K. Chao et al 2002
        eps = a12
        r0 = a1*(1+a3*bz)*(1+a9*beta)*(1+a4*((a8-1)*M_ms**2 + 2)/((a8 + 1)*M_ms**2))*Dp**(-1/a11)
        alpha = a5 * (1 + a6*bz) * (1 + a7*Dp) * (1 + a10 * np.log(1+beta))*(1+a14*M_ms)
        r_bowshock = r0 * ((1 + eps) / (1 + eps*np.cos(theta))) ** alpha
        #convert polar to xy so that it actually plots on top of map image
        # there must be a function for this but I can't find it
        x_bs = r_bowshock * np.sin(theta)
        y_bs = r_bowshock * np.cos(theta)
        x_mp = r_magnetopause * np.sin(theta)
        y_mp = r_magnetopause * np.cos(theta)
        ax.plot(x_bs, y_bs, '--', color='black', linewidth=1.5)
        ax.plot(x_mp, y_mp, '-', color='black', linewidth=1.5)

        plt.xlim(-31,31)
        plt.ylim(-31,31)

        plt.gca().invert_xaxis()

        plt.savefig(plotpath+filenames[num], dpi=400)
        plt.clf()

# def plot_fpi_bulkv_mode(map, bins, coordsx, plotpath, log):
#     filenames= ["fpi_bulkv_xmode.png","fpi_bulkv_ymode.png"]
#
#     plottitles= ["FPI Mode Ion Bulk Velocity X","FPI Mode Ion Bulk Velocity Y"]
#
#     spec = bins
#     plt.ion()
#     coords = coordsx/(const.R_earth.value/1000.0) - 1# convert m to km
#     print(map.shape)
#     flatz = np.array(np.sum(map,(2,5)),dtype=np.float64)
#     n_samp_tot = np.array(np.sum(map,(2,3,4,5)),dtype=np.float64)
#     xvel_arr = np.sum(flatz,3)
#     yvel_arr = np.sum(flatz,2)
#
#     xvel_mode_ind = np.argmax(xvel_arr,2)
#     yvel_mode_ind = np.argmax(yvel_arr,2)
#
#     xvel_mode = spec[xvel_mode_ind]
#     yvel_mode = spec[yvel_mode_ind]
#
#     if log is True:
#         raise ValueError("Can't have log of a negative value!")
#         xvel_mode = np.log(xvel_mode)
#         yvel_mode = np.log(yvel_mode)
#         cbar_names = ['Log mode FPI ion bulk velocity X (km/s)',
#                       'Log mode FPI ion bulk velocity Y (km/s)']
#         filenames= ["fpi_bulkv_xlogmode.png","fpi_bulkv_ylogmode.png"]
#
#     if log is False:
#         cbar_names = ['Mode ion bulk velocity X (km/s)',
#                       'Mode ion bulk velocity Y (km/s)']
#
#
#     vel_mode = [xvel_mode, yvel_mode]
#
#     for num in range(0,2):
#         plt.title(plottitles[num])
#         plt.xlabel('Y GSE (Re)')
#         plt.ylabel('X GSE (Re)')
#         b = plt.imshow(vel_mode[num], extent=[coords[0],coords[-1],
#                 coords[0],coords[-1]],origin='lower', cmap='seismic')
#         bar = plt.colorbar()
#         bar.set_label(cbar_names[num])
#         b.set_extent([coords[0],coords[-1],coords[0],coords[-1]])
#         plt.savefig(plotpath+filenames[num], dpi=400)
#         plt.clf()


# def plot_vec_median(map):
#     filenames= ["fpi_bulkv_xmedian.png","fpi_bulkv_ymedian.png"]
#
#     plottitles= ["FPI Median Ion Bulk Velocity X","FPI Median Ion Bulk Velocity Y"]
#
#     spec = bins
#     plt.ion()
#     coords = coordsx/(const.R_earth.value/1000.0) -1 # convert m to km
#     print(map.shape)
#     flatz = np.array(np.sum(map,(2,5)),dtype=np.float64)
#
#     xvel_arr = np.sum(flatz,3)
#     yvel_arr = np.sum(flatz,2)
#
#     xmedianloc = np.argmax(xvel_arr,2)
#     xmedianlist = bins[np.reshape(xmedianloc,xmedianloc.size)]
#     xmedianlist[np.where(xmedianlist == bins[0])] = np.nan
#     xvel_median = np.reshape(xmedianlist,(flatz.shape[0],flatz.shape[1]))
#
#     ymedianloc = np.argmax(yvel_arr,2)
#     ymedianlist = bins[np.reshape(ymedianloc,ymedianloc.size)]
#     ymedianlist[np.where(ymedianlist == bins[0])] = np.nan
#     yvel_median = np.reshape(ymedianlist,(flatz.shape[0],flatz.shape[1]))
#
#
#     if log is True:
#         raise ValueError("Can't have log of a negative value!")
#         xvel_median = np.log(xvel_avg)
#         yvel_median = np.log(yvel_avg)
#         cbar_names = ['Log median ion bulk velocity X (km/s)',
#                       'Log median ion bulk velocity Y (km/s)']
#         filenames= ["fpi_bulkv_xlogmedian.png","fpi_bulkv_ylogmedian.png"]
#
#     if log is False:
#         cbar_names = ['Median ion bulk velocity X (km/s)',
#                       'Median FPI bulk velocity Y (km/s)']
#
#     vel_median = [xvel_median,yvel_median]
#
#     for num in range(0,2):
#         plt.title(plottitles[num])
#         plt.xlabel('Y GSE (Re)')
#         plt.ylabel('X GSE (Re)')
#         b = plt.imshow(vel_median[num], extent=[coords[0],coords[-1],
#                 coords[0],coords[-1]],origin='lower', cmap='gist_rainbow')
#         bar = plt.colorbar()
#         bar.set_label(cbar_names[num])
#         b.set_extent([coords[0],coords[-1],coords[0],coords[-1]])
#         plt.savefig(plotpath+filenames[num], dpi=400)
#         plt.clf()




def plot_fpi_ni_mean(map, bins, coordsx, plotpath, log, xz):
    filename= "fpi_ni_mean.png"

    plottitle= "FPI Mean Ion Density"

    spec = bins
    plt.ion()
    coords = coordsx/(const.R_earth.value/1000.0) - 1 # convert m to km
    print(bins.shape)
    print(map.shape)
    fig, ax = plt.subplots()

    if xz is True:
        ax.set_xlabel('Z GSE (Re)')
        ax.set_ylabel('X GSE (Re)')
        # flatz = np.array(np.sum(map,1),dtype=np.float64)
        # total_n = np.array(np.sum(map,(1,3)),dtype=np.float64)
        # does a section through Earth+XZ plane now
        flatz = np.sum(map[:,int(len(coordsx)/2-8):int(len(coordsx)/2+8),:,:],1)
        total_n = np.array(np.sum(flatz,2),dtype=np.float64)
    if xz is False:
        ax.set_xlabel('Y GSE (Re)')
        ax.set_ylabel('X GSE (Re)')
        flatz = np.array(np.sum(map,2),dtype=np.float64)
        total_n = np.array(np.sum(map,(2,3)),dtype=np.float64)
    print(flatz.shape,total_n.shape)
    avg = np.log(np.sum(flatz * np.exp(bins),2) / total_n)


    ax.set_title(plottitle)

    image_out = ax.imshow(avg, extent=[coords[0],coords[-1],
            coords[0],coords[-1]],origin='lower', cmap='CMRmap')

    bar = plt.colorbar(image_out)
    bar.set_label('Log mean FPI ion number density (1/cm^3)')

    # plot Earth
    dual_half_circle((0,0), radius=1, angle=0, ax=ax)
    # plot statistical bow shock
    # s66 parameters
    # manually fitted.
    M_ms = 6.96 # 6.96
    # Bz in nanoTesla
    bz = -15e-9
    # upstream solar wind dynamic pressure Pa
    Dp = 2.5 # 2.5
    # solar wind beta
    beta = 0.03
    a1 = 11.646
    a2 = 0.2016
    a3 = 0.122
    a4 = 6.215
    a5 = 0.578
    a6 = -0.009
    a7 = 0.012
    r0 = (a1 + 8*a3 - 8*a2 + a3*bz)*Dp**(-1/a4)
    spread = 1.2
    theta = np.linspace(-np.pi/spread,np.pi/spread, 100)
    alpha = (a5 + a6*bz)*(1 + a7*Dp)
    r_magnetopause = r0 * (2 / (1 + np.cos(theta))) ** alpha
    # assuming Bz < 0
    #r0 = a1(1+a2*bz)*(1 + a9*beta)*(1+a4((a8-1)*m_ms**2+2)/((a8+1)*m_ms**2))*Dp**(-1/a11)
    #alpha = a5 * (1+a13 * bz) * (1+a7*Dp) * (1 + a10 * np.log(1+beta)))*(1+a14*M_ms)
    a1 = 11.1266
    a2 = 0.0010
    a3 = -0.0005
    a4 = 2.5966
    a5 = 0.8182
    a6 = -0.0170
    a7 = -0.0122
    a8 = 1.3007
    a9 = -0.0049
    a10 = -0.0328
    a11 = 6.047
    a12 = 1.029
    a13 =  0.0231
    a14 = -0.002
    # courtesy of J.K. Chao et al 2002
    eps = a12
    r0 = a1*(1+a3*bz)*(1+a9*beta)*(1+a4*((a8-1)*M_ms**2 + 2)/((a8 + 1)*M_ms**2))*Dp**(-1/a11)
    alpha = a5 * (1 + a6*bz) * (1 + a7*Dp) * (1 + a10 * np.log(1+beta))*(1+a14*M_ms)
    r_bowshock = r0 * ((1 + eps) / (1 + eps*np.cos(theta))) ** alpha
    #convert polar to xy so that it actually plots on top of map image
    # there must be a function for this but I can't find it
    x_bs = r_bowshock * np.sin(theta)
    y_bs = r_bowshock * np.cos(theta)
    x_mp = r_magnetopause * np.sin(theta)
    y_mp = r_magnetopause * np.cos(theta)
    ax.plot(x_bs, y_bs, '--', color='black', linewidth=1.5)
    ax.plot(x_mp, y_mp, '-', color='black', linewidth=1.5)

    plt.xlim(-31,31)
    plt.ylim(-31,31)

    plt.gca().invert_xaxis()

    plt.savefig(plotpath+filename, dpi=400)
    plt.clf()


def plot_fpi_ni_mode(map, bins, coordsx, plotpath, log, xz):
    filename= "fpi_ni_mode.png"

    plottitle= "FPI Mode Ion Density"

    spec = bins
    plt.ion()
    coords = coordsx/(const.R_earth.value/1000.0) -1 # convert m to km

    fig, ax = plt.subplots()

    if xz is True:
        ax.set_xlabel('Z GSE (Re)')
        ax.set_ylabel('X GSE (Re)')
        # flatz = np.array(np.sum(map,1),dtype=np.float64)
        flatz = np.sum(map[:,int(len(coordsx)/2-8):int(len(coordsx)/2+8),:,:],1)
    if xz is False:
        ax.set_xlabel('Y GSE (Re)')
        ax.set_ylabel('X GSE (Re)')
        flatz = np.array(np.sum(map,2),dtype=np.float64)

    modeloc = np.argmax(flatz,2)
    modelist = bins[np.reshape(modeloc,modeloc.size)]
    modelist[np.where(modelist == bins[0])] = bins[np.clip(
                        np.max(modeloc)+1,None,len(bins)-1)]
    # median = np.zeros((flatz.shape[0],flatz.shape[1]),dtype=np.float64)
    mode = np.reshape(modelist,(flatz.shape[0],flatz.shape[1]))


    ax.set_title(plottitle)

    b = ax.imshow(mode, extent=[coords[0],coords[-1],
            coords[0],coords[-1]],origin='lower', cmap='CMRmap')

    bar = plt.colorbar(b)
    bar.set_label('Mode FPI ion number density (cm^-3)')
    if log is True:
        bar.set_label('Log mode FPI ion number density log(1/cm^3)')

    # plot Earth
    dual_half_circle((0,0), radius=1, angle=0, ax=ax)
    # plot statistical bow shock
    # s66 parameters
    M_ms = 6.96 # 6.96
    # Bz in nanoTesla
    bz = -15e-9
    # upstream solar wind dynamic pressure Pa
    Dp = 2.5 # 2.5
    # solar wind beta
    beta = 0.03
    a1 = 11.646
    a2 = 0.2016
    a3 = 0.122
    a4 = 6.215
    a5 = 0.578
    a6 = -0.009
    a7 = 0.012
    r0 = (a1 + 8*a3 - 8*a2 + a3*bz)*Dp**(-1/a4)
    spread = 1.2
    theta = np.linspace(-np.pi/spread,np.pi/spread, 100)
    alpha = (a5 + a6*bz)*(1 + a7*Dp)
    r_magnetopause = r0 * (2 / (1 + np.cos(theta))) ** alpha
    # assuming Bz < 0
    #r0 = a1(1+a2*bz)*(1 + a9*beta)*(1+a4((a8-1)*m_ms**2+2)/((a8+1)*m_ms**2))*Dp**(-1/a11)
    #alpha = a5 * (1+a13 * bz) * (1+a7*Dp) * (1 + a10 * np.log(1+beta)))*(1+a14*M_ms)
    a1 = 11.1266
    a2 = 0.0010
    a3 = -0.0005
    a4 = 2.5966
    a5 = 0.8182
    a6 = -0.0170
    a7 = -0.0122
    a8 = 1.3007
    a9 = -0.0049
    a10 = -0.0328
    a11 = 6.047
    a12 = 1.029
    a13 =  0.0231
    a14 = -0.002
    # courtesy of J.K. Chao et al 2002
    eps = a12
    r0 = a1*(1+a3*bz)*(1+a9*beta)*(1+a4*((a8-1)*M_ms**2 + 2)/((a8 + 1)*M_ms**2))*Dp**(-1/a11)
    alpha = a5 * (1 + a6*bz) * (1 + a7*Dp) * (1 + a10 * np.log(1+beta))*(1+a14*M_ms)
    r_bowshock = r0 * ((1 + eps) / (1 + eps*np.cos(theta))) ** alpha
    #convert polar to xy so that it actually plots on top of map image
    # there must be a function for this but I can't find it
    x_bs = r_bowshock * np.sin(theta)
    y_bs = r_bowshock * np.cos(theta)
    x_mp = r_magnetopause * np.sin(theta)
    y_mp = r_magnetopause * np.cos(theta)
    ax.plot(x_bs, y_bs, '--', color='black', linewidth=1.5)
    ax.plot(x_mp, y_mp, '-', color='black', linewidth=1.5)

    plt.xlim(-31,31)
    plt.ylim(-31,31)

    plt.gca().invert_xaxis()

    plt.savefig(plotpath+filename, dpi=400)
    plt.clf()
