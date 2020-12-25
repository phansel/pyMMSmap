# real	0m0.934s
# user	0m0.208s
# sys	0m0.440s
# cdflib wins this race.

import cdflib

mypath = "/media/data/data/mms/mms1/fpi/fast/l2/dis-moms/2017/08/"

bulkv_1 = "mms1_dis_bulkv_gse_fast"
bulkv_1_l = "mms1_dis_bulkv_gse_label_fast"
epoch = 'Epoch'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for filename in onlyfiles:
    cdffile = cdflib.CDF(mypath+filename)
