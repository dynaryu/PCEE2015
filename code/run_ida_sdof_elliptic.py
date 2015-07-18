import pandas as pd
import os
import numpy as np
from collections import OrderedDict

from model_GA import read_EQRM_bldg_params
from model_elliptical import model_elliptical
from gmotion import gmotion
from perform_ida import IDA_scale

#########################
# read ground motion

#output_file_name = 'urml_mean_ellip_result.csv'
output_file_name = 'hazus_urml_pre_result.csv'

# model definition
g_const = 9.806
(dy, fy, du, fu, damp_ratio, period) = read_EQRM_bldg_params('urmlmean', g_const)

urml_mean_ellip = model_elliptical(dy=dy, fy=fy, du=du, fu=fu, damp_ratio=damp_ratio)

im_range = np.exp(np.linspace(np.log(0.01), np.log(3.0), 30))

gmotion_path = '/Users/hyeuk/Project/sv01-30'
gmotion_list_file = 'sv_list.csv'
gmotion_psa_file = 'sv_psa.csv'

gmotion_list = pd.read_csv(os.path.join(gmotion_path, gmotion_list_file))
gmotion_psa = np.loadtxt(os.path.join(gmotion_path, gmotion_psa_file),
    delimiter=',', dtype=float)
gmotion_period = gmotion_psa[0,:]
gmotion_psa = gmotion_psa[1:,:]

dt_comp = 0.005

output_path = '/Users/hyeuk/Project/PCEE2015/data'
output_file = open(os.path.join(output_path, output_file_name), 'wb')
output_file.write('dis, vel, acc, tacc, force, scale, im, gm\n')
output_file.close()

output_file = open(os.path.join(output_path, output_file_name), 'ab')

df_result_dic = OrderedDict()
for item in gmotion_list.FileName:

    print 'running IDA with %s' %item

    val = gmotion_list.loc[gmotion_list.FileName == item]
    gfile = os.path.join(gmotion_path, item)
    dt_gmotion = val.DT.values[0]

    grecord = gmotion(gfile, dt_gmotion, gmotion_period,\
        gmotion_psa[val.index[0], :])

    #df_result = compute_IDA(model=urml_mean_bin,
    #    grecord=grecord, period=0.0, dt_comp=dt_comp,
    #    g_const=g_const, incr_sa=0.02, df_factor=20.0)

    df_result = IDA_scale(model=urml_mean_ellip, grecord=grecord, period=0.0,\
        dt_comp=dt_comp, g_const=g_const, im_range=im_range)

    df_result['gm'] = pd.Series([item.split('.')[0]]*len(df_result))
    df_result.to_csv(output_file, header=False, index=False)

    df_result_dic[item.split('.')[0]] = df_result

# write csv file
output_file.close()
print '%s is written' %os.path.join(output_path, output_file_name)
