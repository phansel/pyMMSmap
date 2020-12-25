# /usr/bin/python3

# This program reads CDF files,
#   - Looks at desired quantities
#   - Looks at desired mapping area
#   - Either divides input data for diferent cores,
#   or divides between two processes and two GPUs
#
#
# Author: Paul Hansel
# Date: 9/17/2018

# imports

# import threading
import os
import numpy as np

# import numba
# from numba import cuda
# from numba import *
import matplotlib.pyplot as plt

import csv
from astropy import constants as const
import cdflib

import h5py

earthRadius = const.R_earth.value

# mypath = "/media/data/data/mms/mms1/fpi/fast/l2/dis-moms/2017/08/"

bulkv_1 = "mms1_dis_bulkv_gse_fast"
bulkv_1_l = "mms1_dis_bulkv_gse_label_fast"
epoch = "Epoch"

from os import listdir
from os.path import isfile, join

# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# coordinate system: gsm or gse
coord_system = "gse"

# default mapping parameters

desired_pixel_dim = np.array([6371.0, 6371.0, 6371.0 * 5])

maxdims = [2e5, 2e5, 2e5]
mindims = [-2e5, -2e5, -2e5]

rangex = [mindims[0], maxdims[0]]
rangey = [mindims[1], maxdims[1]]
rangez = [mindims[2], maxdims[2]]

totaldims = np.subtract(maxdims, mindims)

npixels = np.round(totaldims / desired_pixel_dim, decimals=0)
# print(npixels)
imagedim = np.int64(npixels + 4)

# definition of spatial bins!
coordsx = np.linspace(
    start=mindims[0], stop=maxdims[0], num=imagedim[0], dtype=np.float64
)
coordsz = np.linspace(
    start=mindims[0], stop=maxdims[0], num=imagedim[2], dtype=np.float64
)

# bins = np.array([])

# coordsx = np.arange(imagedim[0],start=rangex[0]-2*desired_pixel_dim[0],step=desired_pixel_dim[0])
# coordsy = findgen(imagedim[1],start=rangey[0]-2*desired_pixel_dim[1],step=desired_pixel_dim[1])
# coordsz = findgen(imagedim[2],start=rangez[0]-2*desired_pixel_dim[2],step=desired_pixel_dim[2])
#
# rangex = [coordsx[0],coordsx[-1]]
# rangey = [coordsy[0],coordsy[-1]]
# rangez = [coordsz[0],coordsz[-1]]


ep1 = "mms1_mec_r_" + coord_system
ep2 = "mms2_mec_r_" + coord_system
ep3 = "mms3_mec_r_" + coord_system
ep4 = "mms4_mec_r_" + coord_system

ephstr = [ep1, ep2, ep3, ep4]


def read_cdfs_month(spacecraft, datatype, year, month):
    if int(year) < 2015:
        print("Year was set to " + str(year) + ".")
        raise ValueError("Spacecraft wasn't even launched until 2015.")
    if int(year) > 2020:
        print("Year is > 2019.")
        raise ValueError("What are you doing with this code?")
    if int(month) < 10:
        month = "0" + str(int(month))  # yep
    if int(month) > 12:
        print("The month is" + str(month))
        raise ValueError("What calendar system are you using?")

    # thanks NASA
    cdf_var_alt = ""
    cdf_time_var_alt = ""

    # these are presets for my specific mapping goals
    if datatype == "fpi_bulkv":  # float32
        pathext = "/fpi/fast/l2/dis-moms/"
        cdf_var = "mms" + str(spacecraft) + "_dis_bulkv_gse_fast"
        cdf_err_var = "mms" + str(spacecraft) + "_dis_bulkv_err_fast"
        cdf_time_var = "Epoch"  # confirmed
        # ~4.5 seconds between samples

    elif datatype == "fpi_ni":  # float32
        pathext = "/fpi/fast/l2/dis-moms/"
        cdf_var = "mms" + str(spacecraft) + "_dis_numberdensity_fast"
        cdf_err_var = "mms" + str(spacecraft) + "_dis_numberdensity_err_fast"
        cdf_time_var = "Epoch"  # confirmed
        # ~4.5 seconds between samples

    # not sure about this! haven't downloaded data.
    elif datatype == "fpi_ne":  # float32
        pathext = "/fpi/fast/l2/des-moms/"
        cdf_var = "mms" + str(spacecraft) + "_des_numberdensity_fast"
        cdf_err_var = "mms" + str(spacecraft) + "_des_numberdensity_err_fast"
        cdf_time_var = "Epoch"  # confirmed
        # ~4.5 seconds between samples

    elif datatype == "swd":  # float32
        pathext = "/dsp/fast/l2/swd/"
        cdf_var = "mms" + str(spacecraft) + "_dsp_swd_E12_Counts"
        cdf_var_alt = "mms" + str(spacecraft) + "_dsp_swd_E34_Counts"
        cdf_err_var = ""
        cdf_time_var = "Epoch_E12"  # also  CDF_TIME_TT2000 int64
        cdf_time_var_alt = "Epoch_E34"
        # 1 per second

    elif datatype == "mec":  # float64
        pathext = "/mec/srvy/l2/ephts04d/"
        cdf_var = "mms" + str(spacecraft) + "_mec_r_" + coord_system
        cdf_err_var = ""
        cdf_time_var = "Epoch"  # units are ns since Jan 1 2000; np.int64
        # 30 s between samples

    elif datatype == "fgm":  # float32
        pathext = "/fgm/srvy/l2/"
        cdf_var = "mms" + str(spacecraft) + "_fgm_b_gse_srvy_l2"  # no bvec
        cdf_err_var = ""
        cdf_time_var = "Epoch"  # confirmed int64
        # 0.0625 s between samples 16/s

    elif datatype == "edp":  # float32
        pathext = "/edp/slow/l2/dce/"
        cdf_var = "mms" + str(spacecraft) + "_edp_dce_gse_slow_l2"
        cdf_err_var = "mms" + str(spacecraft) + "_edp_dce_err_slow_l2"
        cdf_time_var = "mms" + str(spacecraft) + "_edp_epoch_slow_l2"  # confirmed int64
        # 0.03125 s between samples 32/s

    elif datatype == "epar":  # float32
        pathext = "/edp/brst/l2/dce/"
        cdf_var = "mms" + str(spacecraft) + "_edp_dce_par_epar_brst_l2"
        cdf_err_var = "mms" + str(spacecraft) + "_edp_dce_err_slow_l2"
        cdf_time_var = "mms" + str(spacecraft) + "_edp_epoch_brst_l2"  # confirmed int64
        # 0.03125 s between samples 32/s

    else:
        raise ValueError("Data type unknown. ~fpi_ni/bulkv/swd/mec/fgm/edp")

    print(pathext + cdf_var)
    try:
        cdfpath = str(
            "/media/data/data/mms/mms"
            + str(spacecraft)
            + pathext
            + str(year)
            + "/"
            + str(month)
            + "/"
        )
        print("reading from " + cdfpath)
        cdf_files = [f for f in listdir(cdfpath) if isfile(join(cdfpath, f))]
        # print("CDF file array list:")
        # print(cdf_files)
        list_cdfs = [cdflib.CDF(cdfpath + cdffile) for cdffile in cdf_files]

        print("loaded CDFs")
        cdf_varlist = [cdf_var, cdf_time_var]  # cdf_err_var,
        cdf_varlist_alt = [cdf_var_alt, cdf_time_var_alt]

        desired_data = get_desired_vars(list_cdfs, cdf_varlist, cdf_varlist_alt)
        # save our RAM
        for cdf in list_cdfs:
            cdf.close()

        return desired_data

    except:
        print("Couldn't read CDF directory.")
        raise ValueError("Fucj")
        return []


# eliminates undesired variables to free up memory.
# @jit
def get_desired_vars(list_cdfs, cdf_varlist, cdf_varlist_alt):
    outdata = []
    for thiscdf in list_cdfs:
        thisdict = {"cdf": thiscdf.cdf_info()["CDF"]}  # gets name of thing
        try:
            for i, var in enumerate(cdf_varlist):
                thisdict[var] = thiscdf.varget(cdf_varlist[i])
            thisdict["varname"] = cdf_varlist[0]
            thisdict["timevarname"] = cdf_varlist[1]
        except:
            # for the specific case in which there is no alt
            # we'll just pretend it has the original name so nothing breaks.
            # days with half/both data will only have the first. ok.
            for i, _blah in enumerate(cdf_varlist_alt):
                thisdict[cdf_varlist[i]] = thiscdf.varget(cdf_varlist_alt[i])
            thisdict["varname"] = cdf_varlist[0]
            thisdict["timevarname"] = cdf_varlist[1]
            print("Successfully loaded and remapped alternate field names.")
        outdata.append(thisdict)
        # all this probably makes everything very memory inefficient.
    return outdata


# still doesn't work
# aligns daily E and B cdfs with each other.
# then calculates ExB velocities as list of dictionaries.
# @jit
def calc_ExB_month(spacecraft, year, month):
    bdata = read_cdfs_month(spacecraft, "fgm", year, month)
    edata = read_cdfs_month(spacecraft, "edp", year, month)
    if bdata and edata:
        bdata_interp = interpolate_b_to_e(edata, bdata)
        exb = []
        for ed, bdi in zip(edata, bdata_interp):
            print("HERE IS ED/BDI")
            print(ed)
            print(bdi)
            thisExBdict = {}
            thisExBdict["timevarname"] = ed["timevarname"]
            localdatstr = "mms" + str(spacecraft) + "_exb"
            thisExBdict[localdatstr] = calc_ExB_cdf(bdi, ed, spacecraft)
            thisExBdict["varname"] = localdatstr
            exb.append(thisExBdict)
        return exb
    else:
        return []


# this minimizes the difference between value and array.
# @cuda.jit#(nopython=True)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# @jit#(nopython=True)
# def crossp_div(e,b):
#     exb = np.cross(e,b)
#     exb = np.divide(exb, np.dot(b,b))
#     return exb
#

# bdata and edata are lists of dicts in the format returned by read_cdfs_month.
# @jit
# def interpolate_b_to_e(edata, bdata):
#     interpB = []
#     for bd,ed in zip(bdata,edata):
#         # remove the magnitude column, ok.
#         bintx = np.interp(ed[ed['timevarname']],bd[bd['timevarname']],
#                               bd[bd['varname']][:,0])
#         binty = np.interp(ed[ed['timevarname']],bd[bd['timevarname']],
#                               bd[bd['varname']][:,1])
#         bintz = np.interp(ed[ed['timevarname']],bd[bd['timevarname']],
#                               bd[bd['varname']][:,2])
#         # the above part is fine
#         bdatay_interp = np.vstack((bintx,binty,bintz))
#         bdatay_interp = np.transpose(bdatay_interp)
#         # print(bd['varname'])
#         interpBhere = { 'varname':bd['varname'],
#                         bd['varname']: bdatay_interp,
#                         'timevarname':bd['timevarname'],
#                         bd['timevarname']:bd[bd['timevarname']]}
#         interpB.append(interpBhere)
#         print(interpB)
#         print(bdatay_interp)
#     return interpB


# align data with mec data;
# given data takes priority.
# data and mec should be plain old NP array objs
def bin_mec(spacecraft, year, month, imagedim):
    mecdata_here = read_cdfs_month(spacecraft, "mec", year, month)
    # some data isn't contained in a single month.
    # also, this isn't a numpy array either.
    if not mecdata_here:
        return (np.array([]), np.array([]))

    # if we get here, mecdata_here must be properly defined
    if int(month) == 12:
        mecdata_here2 = read_cdfs_month(spacecraft, "mec", str(int(year) + 1), "1")
    else:
        mecdata_here2 = read_cdfs_month(spacecraft, "mec", year, str(int(month) + 1))
    # What the hell?
    try:
        print(spacecraft)
    except:
        print("================Spacecraft is undefined!!===============")
        ValueError("sc undefined")

    # mecdata_here is a list of CDF objects, not read in order!
    mecdata_x, mecdata_y, spatial_binned = bin_mec_internal(
        mecdata_here,
        mecdata_here2,
        spacecraft,
        year,
        month,
        rangex,
        rangey,
        rangez,
        imagedim,
    )
    return (spatial_binned, mecdata_x)


# jit-enabled function for all the bin_mec heavy lifting.
# @jit
def bin_mec_internal(
    mecdata_here,
    mecdata_here2,
    spacecraft,
    year,
    month,
    rangex,
    rangey,
    rangez,
    imagedim,
):

    for num, cdf in enumerate(mecdata_here):
        if num == 0:
            mecdata_x = cdf["Epoch"]
            mecdata_y = cdf["mms" + str(spacecraft) + "_mec_r_" + coord_system]
        else:
            # print(max(cdf['Epoch']))
            mecdata_x = np.append(mecdata_x, cdf["Epoch"], axis=0)
            mecdata_y = np.append(
                mecdata_y,
                cdf["mms" + str(spacecraft) + "_mec_r_" + coord_system],
                axis=0,
            )
    # some data isn't contained in a single month. or even two, somehow!
    for num, cdf in enumerate(mecdata_here2):
        mecdata_x = np.append(mecdata_x, cdf["Epoch"], axis=0)
        mecdata_y = np.append(
            mecdata_y, cdf["mms" + str(spacecraft) + "_mec_r_" + coord_system], axis=0
        )

    # mecdata_y and mecdata_x are arrays of 1 month of the data.
    # reminder: mecdata_x is int64 time, mecdata_y is float64 GSE radius.
    coordsx = np.linspace(rangex[0], rangex[1], imagedim[0])
    coordsy = np.linspace(rangey[0], rangey[1], imagedim[1])
    coordsz = np.linspace(rangez[0], rangez[1], imagedim[2])
    # print("coords:")
    # print(coordsx,coordsy,coordsz)
    print("length of mecdata_y:", len(mecdata_y), len(mecdata_y[0]))
    print(type(mecdata_y))

    ######### BUG HERE ##########

    mecdata_y = mecdata_y.transpose()
    sp_binnedx = np.digitize(mecdata_y[0], bins=coordsx, right=True)
    sp_binnedy = np.digitize(mecdata_y[1], bins=coordsy, right=True)
    sp_binnedz = np.digitize(mecdata_y[2], bins=coordsz, right=True)

    print(np.min(mecdata_y), mecdata_y.dtype)
    print(coordsx)

    spatial_binned = np.array([sp_binnedx, sp_binnedy, sp_binnedz], dtype="int64")
    # shape of spatial_binned is (3,n)
    return mecdata_x, mecdata_y, spatial_binned


# @jit
def align_data_to_mec(spacecraft, data_x, year, month, imagedim):
    # data coming in is aligned by time
    # mec coming in is not MEC but just bin1
    # returns a list of bins in which the input
    # non-mec data goes into
    # bin the MEC in space first, then bin inputs by space via MEC

    # mec data:rmecrange
    # x = np.array(dim = n)
    # y = np.array(dim = 3,n)
    spatial_binned, mecdata_x = bin_mec(spacecraft, year, month, imagedim)

    if not mecdata_x.any():
        print("No MEC data for ", year, month, ". Continuing")
        return np.array([])
    # 30 second clustering
    # bin by time, by proximity to each MEC datapoint
    # then propagate each MEC datapoint's bin index
    # to those that point to it via arrays.
    # make sure mec spaced appropriately here - between or at 30s segments?
    # data_x is not monotonic.
    print("spatial_binned is", spatial_binned)
    print("mecdata_x is", mecdata_x)
    print("max mdx:", max(mecdata_x))
    print("max dtx:", max(data_x))
    if max(data_x) > max(mecdata_x):
        raise ValueError(
            "MEC data does not exist for this timeseries data! at", max(data_x)
        )
    print("min mdx:", min(mecdata_x))
    print("min dtx:", min(data_x))
    print(len(spatial_binned))
    sss = np.array(mecdata_x).argsort()
    mecdata_x = mecdata_x[sss]
    print("sss is", sss)
    # we can use something other than digitize in the future for performance.
    temporal_bins = np.digitize(data_x, mecdata_x, right=False)  # maybe true?
    data_mecalign = np.zeros([3, len(data_x)])
    # I keep forgetting how this works.
    for i in range(3):
        spatial_binned[i - 1] = spatial_binned[i - 1][sss]
        data_mecalign[i - 1] = spatial_binned[i - 1][
            temporal_bins
        ]  # magic - this is 3xnd
    return data_mecalign  # this is indices of the data's locations in the map


# @jit
# loads one month of data at a time.
# @jit
def load_data_and_map(spacecraft, datatype, year, month, imagedim):
    initialized_data = False
    print(datatype, spacecraft, year, month)
    print(type(datatype), type(spacecraft), type(year), type(month))

    if datatype is "exb":  # because ExB is composed of >1 data cdf type
        data_tmp = calc_ExB_month(spacecraft, year, month)
    else:
        data_tmp = read_cdfs_month(spacecraft, datatype, year, month)
    # data_tmp is one of the only returns which *isn't* a numpy array.
    if not data_tmp:
        print("No data present.")
        return np.array([]), np.array([])
    # this process is data agnostic; it only cares about the time.
    for data_cdf in data_tmp:
        if initialized_data is False:  # only do this on the first cdf
            data_x = data_cdf[data_cdf["timevarname"]]  # laziness
            data_y = data_cdf[data_cdf["varname"]]  # magic of py
            initialized_data = True
        else:
            data_x = np.append(data_x, data_cdf[data_cdf["timevarname"]], axis=0)
            data_y = np.append(data_y, data_cdf[data_cdf["varname"]], axis=0)

    # data_x and data_y could be really large here.
    data_mecalign = align_data_to_mec(spacecraft, data_x, year, month, imagedim)

    if not data_mecalign.any():
        print("No mec-aligned data. Continuing.")
        return np.array([]), np.array([])
    # for 3D data (ExB and FPI bulkv),
    # we want to bin by 3D velocities and magnitudes
    # vector data:

    if datatype in ["exb", "fpi_bulkv", "edp"]:
        dmap, bins = map_vector_data(data_mecalign, data_y, datatype)
    # for scalar data, e.g. SWD, FPI density,
    # we only bother doing (total number of ESW) + (total samples)
    if datatype in ["fpi_ni", "swd"]:
        dmap, bins = map_scalar_data(data_mecalign, data_y, datatype)

    print(datatype)
    try:
        return (dmap, bins)
    except:
        raise ValueError("Datatype not supported")


# @jit
def map_vector_data(data_mecalign, data_y, datatype):
    # set a default scale and bin range for data
    # mapdim = np.array([imagedim]).append([specdim,specdim,specdim])
    # map = np.zeros(mapdim, dtype='int32')
    print("entered danger zone v1")

    # have to reduce map resolution
    desired_pixel_dim = np.array([6371.0, 6371.0, 6371.0 * 5])

    maxdims = [2e5, 2e5, 2e5]
    mindims = [-2e5, -2e5, -2e5]

    rangex = [mindims[0], maxdims[0]]
    rangey = [mindims[1], maxdims[1]]
    rangez = [mindims[2], maxdims[2]]

    totaldims = np.subtract(maxdims, mindims)

    npixels = np.round(totaldims / desired_pixel_dim, decimals=0)
    # print(npixels)
    imagedim = np.int64(npixels + 5)

    # definition of spatial bins!
    coordsx = np.linspace(
        start=mindims[0], stop=maxdims[0], num=imagedim[0], dtype=np.float64
    )
    coordsz = np.linspace(
        start=mindims[0], stop=maxdims[0], num=imagedim[2], dtype=np.float64
    )

    if datatype == "fpi_bulkv":
        fpi_bulkv_bins = np.linspace(-1.5e3, 1.5e3, 30, dtype="float32")
        bins = fpi_bulkv_bins
        print("imagedim is ", imagedim)
        mapdim = np.append(np.array([imagedim]), [len(bins), len(bins), len(bins)])
        mapdim = np.array(mapdim, dtype="int")
        dmap = np.zeros(mapdim, dtype="int32")
        print("data_mecalign is", data_mecalign)
        print("data_y is ", data_y)
        print("dataymax:", np.max(data_y))
        print("dataymin:", np.min(data_y))
        ptsx = np.array(data_mecalign[0, :], dtype="int64")
        ptsy = np.array(data_mecalign[1, :], dtype="int64")
        ptsz = np.array(data_mecalign[2, :], dtype="int64")
        print(np.max(ptsx), np.max(ptsy), np.max(ptsz))
        data_y = data_y.transpose()
        pvx = np.digitize(data_y[0], bins, right=True)
        pvy = np.digitize(data_y[1], bins, right=True)
        pvz = np.digitize(data_y[2], bins, right=True)
        try:
            print(dmap.shape, ptsx.shape, pvx.shape)
            print(bins, bins.shape)
            np.add.at(dmap, (ptsx, ptsy, ptsz, pvx, pvy, pvz), 1)
        except IndexError:
            raise IndexError("FPI velocity bins cap too low. Edit line 445")

    # if (datatype is 'exb'):
    #     exb_bins = np.linspace(-1e6,1e6,30,dtype='float32')
    #     bins = exb_bins
    #     print( "imagedim is ", imagedim)
    #     mapdim = np.append(np.array([imagedim]),[len(bins),len(bins),len(bins)])
    #     mapdim = np.array(mapdim, dtype='int')
    #     dmap = np.zeros(mapdim, dtype='int')
    #     print("data_mecalign is", data_mecalign)
    #     print("data_y is ", data_y)
    #     print("dataymax:",max(data_y))
    #     ptsx = np.array(data_mecalign[0,:],dtype='int64')
    #     ptsy = np.array(data_mecalign[1,:],dtype='int64')
    #     ptsz = np.array(data_mecalign[2,:],dtype='int64')
    #     data_y = data_y.transpose()
    #     pvx = np.digitize(data_y[0],bins,right=True)
    #     pvy = np.digitize(data_y[1],bins,right=True)
    #     pvz = np.digitize(data_y[2],bins,right=True)
    #     try:
    #         np.add.at(dmap,(ptsx,ptsy,ptsz,pvx,pvy,pvz),1)
    #     except IndexError:
    #         print("ExB velocity bins cap too low. Edit line 445")

    return dmap, bins


# @jit
def map_scalar_data(data_mecalign, data_y, datatype):
    # set a default scale and binrange for data
    # set a default compression structure.
    # single dimension of data; values from [bin0b] to [binNt]
    if datatype == "fpi_ni":
        bins = np.linspace(-7, 8, 256, dtype="float32")
        print("imagedim is ", imagedim)
        mapdim = np.append(np.array([imagedim]), len(bins))
        mapdim = np.array(mapdim, dtype="int")
        dmap = np.zeros(mapdim, dtype="int")
        print("data_mecalign is", data_mecalign)
        print("data_y is ", data_y)
        print("dataymax:", max(data_y))
        ptsx = np.array(data_mecalign[0, :], dtype="int64")
        ptsy = np.array(data_mecalign[1, :], dtype="int64")
        ptsz = np.array(data_mecalign[2, :], dtype="int64")
        ptsn = np.digitize(np.log(data_y), bins, right=True)
        try:
            np.add.at(dmap, (ptsx, ptsy, ptsz, ptsn), 1)
        except IndexError:
            print("FPI number density bins cap too low. Edit line 417")

    elif datatype == "swd":
        bins = np.arange(257, dtype="int")

        mapdim = np.append(np.array([imagedim]), [len(bins), 4])
        ## create new map of dimensions [x,y,z,(0,1,...256),4]
        dmap = np.zeros(mapdim, dtype="int64")

        ptsx = np.array(data_mecalign[0, :], dtype="int64")
        ptsy = np.array(data_mecalign[1, :], dtype="int64")
        ptsz = np.array(data_mecalign[2, :], dtype="int64")
        print(imagedim.dtype)
        print(ptsx, ptsx.shape)
        print(data_mecalign.shape)
        # SWD data is actually counts per bin.
        pn1 = np.digitize(data_y[:, 0], bins, right=True)
        pn2 = np.digitize(data_y[:, 1], bins, right=True)
        pn3 = np.digitize(data_y[:, 2], bins, right=True)
        pn4 = np.digitize(data_y[:, 3], bins, right=True)
        try:
            np.add.at(dmap, (ptsx, ptsy, ptsz, pn1, 0), 1)
            np.add.at(dmap, (ptsx, ptsy, ptsz, pn2, 1), 1)
            np.add.at(dmap, (ptsx, ptsy, ptsz, pn3, 2), 1)
            np.add.at(dmap, (ptsx, ptsy, ptsz, pn4, 3), 1)
        except:
            raise ValueError("Something went wrong.")

    return (dmap, bins)


# probably this isn't jittable, but maybe.
# @jit
def mapMultipleMonths(sc, datatype, year1, year2, month1, month2, imagedim):
    # Generates maps between year1month1 and year2month2.
    # sanitize input
    year1, year2, month1, month2 = (int(year1), int(year2), int(month1), int(month2))
    # print out for debugging/reference purposes
    print("year1:", str(year1))
    print("year2:", str(year2))
    print("m1:", str(month1))
    print("m2:", str(month2))

    # further sanity checks
    if int(year1) > int(year2):
        raise ValueError("year2 can't be before year1.")

    # load first month, set up the "map,bins" struct
    year = year1
    month = month1
    try:
        map, bins = load_data_and_map(sc, datatype, str(year), str(month), imagedim)
    except:
        raise ValueError("First month didn't load; load a month with good data")
    # should only get here if it passes.

    while (month < month2) or (year < year2):
        month += 1
        # if we've crossed NYE, increment year and reset month
        if month == 13:
            year += 1
            month = 1
        # load this next month

        map_n, b = load_data_and_map(sc, datatype, str(year), str(month), imagedim)
        if map.shape == (0,):
            print(
                "--------------------------------------------------Original map is blank :(---------------------"
            )
            map = map_n
        elif map_n.shape == (0,):
            print(
                "+++++++++++++++++++++++++++++++++++++++This map is shaped like nothing at all: "
                + str(year)
                + str(month)
            )
        else:
            map += map_n

    #    except:
    #        # print a warning if it fails, continue otherwise
    #        print("Failed to load " + str(year) + str(month) + ", skipping")
    #        print("imagedim:")
    #        print(imagedim)
    #        print(datatype)
    #        print(year)
    #        print(sc)
    #        print(month)
    #        print("Shape of map")
    #        print(map.shape)
    #        print("Shape of map_n")
    #        print(map_n.shape)
    #        raise ValueError("Failed to load")

    return map, bins


def mapMultipleMonthsAllSC(datatype, year1, year2, month1, month2, imagedim):
    map1, bins = mapMultipleMonths(1, datatype, year1, year2, month1, month2, imagedim)
    map2, bins = mapMultipleMonths(2, datatype, year1, year2, month1, month2, imagedim)
    map3, bins = mapMultipleMonths(3, datatype, year1, year2, month1, month2, imagedim)
    map4, bins = mapMultipleMonths(4, datatype, year1, year2, month1, month2, imagedim)
    # verify that the maps are actually gucci
    print(map1.size)
    print(map2.size)
    print(map3.size)
    print(map4.size)

    maptot = map1 + map2 + map3 + map4
    return maptot, bins


def plotFlatMap(map):
    flatmap = np.sum(map, (2, 3))
    while flatmap.ndim > 2:
        flatmap = np.sum(flatmap, 2)
    b = plt.imshow(flatmap)
    plt.show(block=False)
    return b


def FpiNiDemo():
    map, bins = load_data_and_map("1", "fpi_ni", "2017", "5")
    map2, bins = load_data_and_map("1", "fpi_ni", "2017", "6")
    map3, bins = load_data_and_map("1", "fpi_ni", "2017", "7")
    map4, bins = load_data_and_map("1", "fpi_ni", "2017", "8")
    map5, bins = load_data_and_map("1", "fpi_ni", "2017", "9")
    map += map2 + map3 + map4 + map5
    flatmap = np.sum(map, (2, 3))
    b = plt.imshow(flatmap)
    plt.show(block=False)


def FpiNiFMDemo():
    x, bins = mapMultipleMonths("1", "fpi_ni", 2015, 2018, "1", "06")
    b = plotFlatMap(x)
    return b


def shortFpiNiDemo():
    x, bins = mapMultipleMonths("1", "fpi_ni", 2015, 2015, "2", "7")
    b = plotFlatMap(x)
    return b, x, bins


# coordsx and coordsy should be the same, so we only include one
def writeHDF5(filename, map, datatype, bins, coordsx, coordsz):
    f = h5py.File(filename, "w")
    f.create_dataset("map", data=map, compression="lzf")
    f["datatype"] = datatype.encode("utf8")
    f["bins"] = bins
    f["coordsx"] = coordsx
    f["coordsz"] = coordsz
    f.close()


def readHDF5(filename):
    f = h5py.File(filename, "r")
    map = np.array(f["map"][()])
    bins = np.array(f["bins"][()])
    coordsx = np.array(f["coordsx"][()])
    coordsz = np.array(f["coordsz"][()])
    datatype = str(f["datatype"][()])
    f.close()
    return map, bins, datatype, coordsx, coordsz


# def gen_list_dates(timerange):
#     years = ['2017']
#     months = ['1','2','3','4']
#     return (years, months)


# def directory_gen(year, month):
#     some_directory = [""]
#     return some_directory

# in 30 seconds, spacecraft moves 30*~10 km.
# If we bundle SWD, ExB, FPI data around MEC,
# we only have to do ~2800 binnings per day.


# need to assemble all MEC data together to make mapping to it
# from each CDF easier
# mecdata = {'x':np.array([]),'y':np.array([])}
# for year in range(2016,2018):
#     for month in range(11):
#         mectemp = read_cdfs_month('1','mec',str(year),str(month))
# c = read_cdfs_month(etc etc etc)
# varname = c[0]['varname']
# datax = c[0][varname]
# c[0][varname][0][0:3]

# def init():
#    print("do nothing")
#    # this works
#    data = np.array([[2,3,22],[40,-3,-14],[-33,-22,23],[-13,-1,-21],[12,-40,-2]])
#    bins = np.array([-40,-30,-20,-10,0,10,20,30,40])
#    spatial_binned = np.digitize(data, bins, right=True)
#    print(spatial_binned)
#
#    x2 = mapMultipleMonths('2','fpi_ni',2015,2018,'6','6')
#    #x = mapMultipleMonths('2','edp',2015,2018,'6','6')
#
#    b2 = plotFlatMap(x2[0])


# try:
#     t1 = threading.Thread( target=align_data_with_mec, args=('thread1','1','exb1'))
#     t2 = threading.Thread( target=align_data_with_mec, args=('thread2','2','exb2'))
##     t3 = threading.Thread( target=align_data_with_mec, args=('thread3','3','exb3'))
#     t4 = threading.Thread( target=align_data_with_mec, args=('thread4','4','exb4'))
# except:
#     print("Error: unable to create threads.")
##
# try:
#     t1.start()
#     t2.start()
#     t3.start()
#     t4.start()
# except:
#     print("Error: unable to start threads.")

# to generate identity matrix:
#
# a = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
# >>> np.add.at(a,([0,1,2],[0,1,2]),1)
# indices given: tuple of arrays; [xcoords],[ycoords],[zcoords]
# a = [[1,0,0],[0,1,0],[0,0,1]]


def read_cdfs_day(spacecraft, datatype, year, month, day):
    if int(year) < 2015:
        print("Year was set to " + str(year) + ".")
        raise ValueError("Spacecraft wasn't flyingh until 2015.")
    if int(year) > 2019:
        print("Year is > 2019.")
        raise ValueError("Year > 2019")
    if int(month) < 10:
        month = "0" + str(int(month))  # yep
    if int(month) > 12:
        print(month)
        raise ValueError("Month > 12")
    if int(day) > 31:
        raise ValueError("Day > 31")

    elif datatype is "epar":  # float32
        pathext = "/edp/brst/l2/dce/"
        cdf_var = "mms" + str(spacecraft) + "_edp_dce_par_epar_brst_l2"
        cdf_err_var = "mms" + str(spacecraft) + "_edp_dce_err_slow_l2"
        cdf_time_var = "mms" + str(spacecraft) + "_edp_epoch_brst_l2"  # confirmed int64
        # 0.03125 s between samples 32/s

    else:
        raise ValueError("Data type unknown. ~epar")

    print(pathext + cdf_var)
    try:
        cdfpath = str(
            "/media/data/data/mms/mms"
            + str(spacecraft)
            + pathext
            + str(year)
            + "/"
            + str(month)
            + "/"
            + str(day)
            + "/"
        )
        print("reading from " + cdfpath)
        cdf_files = [f for f in listdir(cdfpath) if isfile(join(cdfpath, f))]
        list_cdfs = [cdflib.CDF(cdfpath + cdffile) for cdffile in cdf_files]

        print("loaded CDFs")
        cdf_varlist = [cdf_var, cdf_time_var]  # cdf_err_var,
        desired_data = get_desired_vars(list_cdfs, cdf_varlist)
        # save our RAM
        for cdf in list_cdfs:
            cdf.close()

        return desired_data

    except:
        print("Couldn't read CDF directory.")
        return []


# does a cross product on one CDF
# @jit
# def calc_ExB_cdf(bdata, edata, spacecraft):
#     if(len(bdata) is not len(edata)):
#         ValueError("Can't calculate cross products; E&B different lengths.")
#
#     data_b = bdata[bdata['varname']]
#     data_e = edata[edata['varname']]
#
#     time_b = bdata[bdata['timevarname']]
#     time_e = edata[edata['timevarname']]
#
#     # this is the jitted/vectorized function.
#     iemin,iemax,ibmin,ibmax = trim_E_B(time_b,time_e)
#
#     e  = data_e[iemin:iemax-iemin]
#     te = time_e[iemin:iemax-iemin]
#
#     b  = data_b[ibmin:ibmax-ibmin]
#     tb = time_b[ibmin:ibmax-ibmin]
#
#     print(iemin,iemax,ibmin,ibmax)
#     print(iemin-iemax)
#     print(ibmin-ibmax)
#
#     print(e)
#     print(b)
#     print(e.shape,b.shape)
#
#     # interpolate the B onto the E
#     # right now it assumes both lengths are the same
#     # cross product won't work due to lengths
#     exb = crossp_div(e,b)
#     return exb

# # @cuda.jit#(nopython=True)
# def trim_E_B(time_b,time_e):
#     temax = np.max(time_e)
#     temin = np.min(time_e)
#
#     print("TEMAX,TEMIN:",temax, temin)
#     tbmax = np.max(time_b)
#     tbmin = np.min(time_b)
#     print("TBMAX,TBMIN:",tbmax, tbmin)
#     tmin = np.max([temin,tbmin])
#     tmax = np.min([temax,tbmax])
#
#     # tlength = tmax - tmin
#
#     iemin = find_nearest(time_e, tmin)
#     iemax = find_nearest(time_e, tmax)
#     ibmin = find_nearest(time_b, tmin)
#     ibmax = find_nearest(time_b, tmax)
#
#     return (iemin,iemax,ibmin,ibmax)
#
