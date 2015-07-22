import pandas as pd
import os
import numpy as np
from collections import OrderedDict
import scipy.io

from model_GA import read_EQRM_bldg_params
from model_elliptical import model_elliptical
from gmotion import gmotion
from perform_ida import IDA_scale

#########################

HAZUS_BLDG_TYPES = ['W1', 'W2', 'S1L', 'S1M', 'S1H', 'S2L', 'S2M', 'S2H', 'S3', 'S4L',\
    'S4M', 'S4H', 'S5L', 'S5M', 'S5H', 'C1L', 'C1M', 'C1H', 'C2L', 'C2M', 'C2H',\
    'C3L', 'C3M', 'C3H', 'PC1', 'PC2L', 'PC2M', 'PC2H', 'RM1L', 'RM1M', 'RM2L',\
    'RM2M', 'RM2H', 'URML', 'URMM', 'MH']

output_path = '/Users/hyeuk/Project/PCEE2015/data/ellip'
gmotion_path = '/Users/hyeuk/Project/sv01-30'

gmotion_list = pd.read_csv(os.path.join(gmotion_path, 'sv_list.csv'))
gmotion_psa = np.loadtxt(os.path.join(gmotion_path, 'sv_psa.csv'),
    delimiter=',', dtype=float)
gmotion_period = gmotion_psa[0,:]
gmotion_psa = gmotion_psa[1:,:]

im_range = np.exp(np.linspace(np.log(0.05), np.log(5.0), 30))

dt_comp = 0.005
g_const = 9.806

HAZUS_BLDG_TYPES = ['URML']

for bldg in HAZUS_BLDG_TYPES:

    print 'HAZUS bldg: %s' %bldg

    #output_file_name = 'urml_mean_ellip_result.csv'
    #output_file_name = 'hazus_urml_pre_result.csv'

    output_file_name = os.path.join(output_path, 'hazus_' + bldg + '_pre_result.csv')

    # model definition
    (dy, fy, du, fu, damp_ratio, period) = read_EQRM_bldg_params(bldg.lower(), g_const,\
        EQRM_input_file = '/Users/hyeuk/Project/eqrm/resources/data/building_parameters_hazus.csv')

    ellip = model_elliptical(dy=dy, fy=fy, du=du, fu=fu, damp_ratio=damp_ratio)

    output_file = open(os.path.join(output_path, output_file_name), 'wb')
    output_file.write('dis, vel, acc, tacc, force, scale, im, gm\n')
    output_file.close()

    output_file = open(os.path.join(output_path, output_file_name), 'ab')

    #df_result_dic = OrderedDict()
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

        df_result = IDA_scale(model=ellip, grecord=grecord, period=0.0,\
            dt_comp=dt_comp, g_const=g_const, im_range=im_range)

        df_result['gm'] = pd.Series([item.split('.')[0]]*len(df_result))
        df_result.to_csv(output_file, header=False, index=False)

        #df_result_dic[item.split('.')[0]] = df_result

    # write csv file
    output_file.close()
    print '%s is written' %os.path.join(output_path, output_file_name)
