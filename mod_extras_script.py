#### DEPRECATED, EDIT WITHIN MMS_MAP_FUNCTIONS.PY

# #def plot_swd_mean_extras(map, coordsx, plotpath, log, clip, xy_clip, colormap):
# plotpath = "./testimage"
# from mms_map_functions import *
# import sys
# from mms_map_pyrewrite import *
# dirdat = "/home/paul/data/maps/python/swd/somespacecraft/sc1/201512-20196_2020-06-08-00-30gsm/"
# dmap,bins,datatype,coordsx,coordsz = readHDF5(dirdat+"map.h5")
#
# map = dmap
# log = True
# xy_clip = False
# colormap = "inferno"
# clip = False
# filenames= ["bin1swdmean.png", "bin2swdmean.png",
#          "bin3swdmean.png","bin4swdmean.png"]
#
# plottitles=["Bin 1 (0.5-3 mV/m) Average SWD Count",
#             "Bin 2 (3-12 mV/m) Average SWD Count",
#             "Bin 3 (12-50 mV/m) Average SWD Count",
#             "Bin 4 (50+ mV/m) Average SWD Count"]
#
# # map.shape is (51,51,51,257,4)
# # (xi,yi,zi,number_in_bin_at_this_countrate,bin)
# # it is a histogram
#
#
# bin1 = map[:,:,:,:,0]
# bin2 = map[:,:,:,:,1]
# bin3 = map[:,:,:,:,2]
# bin4 = map[:,:,:,:,3]
#
#
# bins = [bin1,bin2,bin3,bin4]
# spec = np.linspace(0,256,257,dtype='int')
# coordsx /= (const.R_earth.value/1000.0) # convert m to km
# # clipvals = [12.5,2.5,2,.06]
#
# bin = 1
#
#
# flatz = np.array(np.sum(bins[bin],2),dtype=np.float64)
# totmap = np.array(np.sum(bins[bin],(2,3)),dtype=np.float64)
#
# bintot = np.sum(flatz * spec,2)
# print(bintot.shape)
# binavg = bintot / totmap
# binavg[np.isnan(binavg)] = 0 # replace NAN with 0
# #if clip is True:
# #    binavg = np.clip(binavg,None,clipvals[bin])
#
# binavg = np.log(binavg)
# filenames= ["bin1swdlogmean.png", "bin2swdlogmean.png",
#             "bin3swdlogmean.png","bin4swdlogmean.png"]
# plottitles=["Bin 1 (0.5-3 mV/m) Log Average SWD Count",
#             "Bin 2 (3-12 mV/m) Log Average SWD Count",
#             "Bin 3 (12-50 mV/m) Log Average SWD Count",
#             "Bin 4 (50+ mV/m) Log Average SWD Count"]
#
# fig, ax = plt.subplots()
#
# ax.set_title(plottitles[bin])
#
# binavg[0,0] = np.log(0.0)
# #print(binavg[0,0])
# #print(binavg[-1,0])
# #print(binavg[0,-1])
# #print(binavg[-1,-1])
#
#
# image_out = ax.imshow(binavg, extent=[coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]],origin='lower', cmap=colormap)
#
#
# ax.set_xlabel('Y GSM (Re)')
# ax.set_ylabel('X GSM (Re)')
#
#
# bar = plt.colorbar(image_out)
# #bar = fig.colorbar(binavg)
#
# bar.set_label('Average SWD count (log 1/s)')
#
# #b.set_extent([coordsx[0],coordsx[-1],coordsx[0],coordsx[-1]])
#
# # plot Earth
# # Courtesy of Joe Kington on StackOverflow
# from matplotlib.patches import Wedge
# def dual_half_circle(center, radius, angle=0, ax=None, colors=('b','k'),
#                      **kwargs):
#     """
#     Add two half circles to the axes *ax* (or the current axes) with the
#     specified facecolors *colors* rotated at *angle* (in degrees).
#     """
#     if ax is None:
#         ax = plt.gca()
#     theta1, theta2 = angle, angle + 180
#     w1 = Wedge(center, radius, theta1, theta2, fc='teal', **kwargs)
#     w2 = Wedge(center, radius, theta2, theta1, fc='black', **kwargs)
#     for wedge in [w1, w2]:
#         ax.add_artist(wedge)
#     return [w1, w2]
#
# dual_half_circle((0,0), radius=1, angle=0, ax=ax)
#
# # plot statistical bow shock
# # s66 parameters
#
# M_ms = 6.96 # 6.96
# # Bz in nanoTesla
# bz = -12e-9
# # upstream solar wind dynamic pressure Pa
# Dp = 0.85 # 2.5
# # solar wind beta
# beta = 0.02
#
#
# a1 = 11.646
# a2 = 0.2016
# a3 = 0.122
# a4 = 6.215
# a5 = 0.578
# a6 = -0.009
# a7 = 0.012
#
# r0 = (a1 + 8*a3 - 8*a2 + a3*bz)*Dp**(-1/a4)
#
# spread = 1.8
#
# theta = np.linspace(-np.pi/spread,np.pi/spread, 100)
# alpha = (a5 + a6*bz)*(1 + a7*Dp)
# r_magnetopause = r0 * (2 / (1 + np.cos(theta))) ** alpha
#
#
#
# # assuming Bz < 0
# #r0 = a1(1+a2*bz)*(1 + a9*beta)*(1+a4((a8-1)*m_ms**2+2)/((a8+1)*m_ms**2))*Dp**(-1/a11)
# #alpha = a5 * (1+a13 * bz) * (1+a7*Dp) * (1 + a10 * np.log(1+beta)))*(1+a14*M_ms)
# a1 = 11.1266
# a2 = 0.0010
# a3 = -0.0005
# a4 = 2.5966
# a5 = 0.8182
# a6 = -0.0170
# a7 = -0.0122
# a8 = 1.3007
# a9 = -0.0049
# a10 = -0.0328
# a11 = 6.047
# a12 = 1.029
# a13 =  0.0231
# a14 = -0.002
#
# # courtesy of J.K. Chao et al 2002
#
# eps = a12
#
# r0 = a1*(1+a3*bz)*(1+a9*beta)*(1+a4*((a8-1)*M_ms**2 + 2)/((a8 + 1)*M_ms**2))*Dp**(-1/a11)
# alpha = a5 * (1 + a6*bz) * (1 + a7*Dp) * (1 + a10 * np.log(1+beta))*(1+a14*M_ms)
#
#
# r_bowshock = r0 * ((1 + eps) / (1 + eps*np.cos(theta))) ** alpha
#
# #convert polar to xy so that it actually plots on top of map image
# # there must be a function for this but I can't find it
# x_bs = r_bowshock * np.sin(theta)
# y_bs = r_bowshock * np.cos(theta)
# x_mp = r_magnetopause * np.sin(theta)
# y_mp = r_magnetopause * np.cos(theta)
#
# ax.plot(x_bs, y_bs, '--', color='black', linewidth=1)
# ax.plot(x_mp, y_mp, '-', color='black', linewidth=1)
#
# plt.xlim(-31,31)
# plt.ylim(-31,31)
#
#
# # show the image
# plt.show()
#
# #plt.savefig(plotpath+filenames[bin], Dpi=400)
# #plt.clf()
