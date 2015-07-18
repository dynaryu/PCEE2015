# batch job

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict

from run_nonlinear_sdof import nonlinear_response_sdof
from model_GA import read_EQRM_bldg_params
from model_bilinear import model_bilinear
from gmotion import gmotion

def compute_IDA(model, grecord, period, dt_comp, g_const, incr_sa=0.1,
    df_factor=20.0):

    df = df_factor*model.dy
    idx_model_period = np.where(grecord.period==model.period)[0]
    scale2dy_est = model.fy/(grecord.psa[idx_model_period]*g_const)

    idx_period = np.where(grecord.period==period)[0]
    sv_psa = grecord.psa[idx_period] # g scale
    incr_scale = incr_sa/sv_psa

    result = []

    istep = 1
    flag = 'NOT_REACHED'
    while flag == 'NOT_REACHED':
        if istep == 1:
            scale = scale2dy_est
        else:
            scale = scale2dy + (istep-1)*incr_scale

        response = nonlinear_response_sdof(model, grecord, -1.0*scale*g_const,
            dt_comp)
        spec = np.abs(response).max()
        spec = spec.append(pd.Series(scale, index=[u'scale']))
        result.append(spec)

        #print "istep: %s, scale: %s, Sdr: %s" %(istep, scale, a0)

        if istep == 1:
            if spec.dis > model.dy:
                scale2dy_est = 0.8*scale
            else:
                scale2dy = scale*model.dy/spec.dis
                istep += 1
        else:
            istep += 1
            if spec.dis > df:
                flag = 'REACHED'

    df_result = pd.DataFrame(result)

    return df_result

def compute_IDA_scale(model, grecord, period, dt_comp, g_const, im_range):

    idx_model_period = np.where(grecord.period==model.period)[0]
    scale2dy = 0.8*model.fy/(grecord.psa[idx_model_period]*g_const)

    idx_period = np.where(grecord.period==period)[0]

    scale_factor = im_range/grecord.psa[idx_period]
    result = []

    for scale in scale_factor[scale_factor > scale2dy]:

        response = nonlinear_response_sdof(model, grecord, -1.0*scale*g_const,
            dt_comp)
        spec = np.abs(response).max()
        spec = spec.append(pd.Series(np.array([scale, scale*grecord.psa[idx_period]]), index=[u'scale',u'im']))
        result.append(spec)

    df_result = pd.DataFrame(result)

    return df_result

def predict_segmented(x, regression_params):

    (interc, slope1, slope2, break_pt)=regression_params

    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x*1.0, float):
        x = np.array([x*1.0])

    y = np.zeros_like(x)

    tf = (log(x) <= break_pt)
    y[tf] =  interc + slope1*np.log(x[tf])

    tf = (log(x) > break_pt)
    y[tf] =  interc + slope1*break_pt + slope2*(np.log(x[tf])-break_pt)

    return np.exp(y)

#########################
# read groundmotion

g_const = 9.806

print "******"

# model definition
#urml_mean_bin = model_EPP_GA('urmlmean', g_const)

(dy, fy, du, fu, damp_ratio, period) = read_EQRM_bldg_params('urmlmean', g_const)
# change  period
period=0.2

urml_mean_bin = model_bilinear(mass=1.0, period=period, fy=fu, alpha=0.0,
    damp_ratio=damp_ratio)

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
output_file_name = 'urml_mean_bin_result_period0p2.csv'
output_file = open(os.path.join(output_path, output_file_name), 'wb')
output_file.write('dis, vel, acc, tacc, force, scale, im, gm\n')
output_file.close()

output_file = open(os.path.join(output_path, output_file_name), 'ab')

df_result_dic = OrderedDict()
for item in gmotion_list.FileName:

    print 'running IDA with %s' %item

    val = gmotion_list.loc[gmotion_list.FileName==item]
    gfile = os.path.join(gmotion_path, item)
    dt_gmotion = val.DT.values[0]

    grecord = gmotion(gfile, dt_gmotion, gmotion_period,
        gmotion_psa[val.index[0],:])

    #df_result = compute_IDA(model=urml_mean_bin,
    #    grecord=grecord, period=0.0, dt_comp=dt_comp,
    #    g_const=g_const, incr_sa=0.02, df_factor=20.0)

    df_result = compute_IDA_scale(model=urml_mean_bin,
       grecord=grecord, period=0.0, dt_comp=dt_comp,
       g_const=g_const, im_range=im_range)

    df_result['gm'] = pd.Series([item.split('.')[0]]*len(df_result))
    df_result.to_csv(output_file, header=False, index=False)

    df_result_dic[item.split('.')[0]] = df_result

# write csv file
output_file.close()
print '%s is written' %os.path.join(output_path, output_file_name)