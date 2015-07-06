# batch job

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict

from run_nonlinear_sdof import nonlinear_response_sdof
from model_GA import model_EPP_GA
from gmotion import gmotion

def compute_IDA(model, grecord, period, g_const, incr_sa=0.1, df_factor=10.0):

    df = df_factor*model.dy
    #incr_sa = 0.1

    idx_period = np.where(gmotion.period==period)[0]

    sv_psa = gmotion.psa[idx_period] # g scale
    scale2dy_est = model.fy/(sv_psa*g_const)
    incr_scale = incr_sa/sv_psa

    Sdr, Sar, Sat, Sforce, PGA = [], [], [], [], []

    istep = 1 
    flag = 'NOT_REACHED'
    while flag == 'NOT_REACHED':
        if istep == 1:
            scale = scale2dy_est
        else:
            scale = scale2dy + (istep-1)*incr_scale

        (spec_dis, spec_vel, spec_acc, spec_tacc, spec_force) = \
        nonlinear_response_sdof(model, gmotion, scale*g_const)
        
        Sdr.append(spec_dis)
        Sar.append(spec_acc/g_const)
        Sat.append(spec_tacc/g_const)
        Sforce.append(spec_force/g_const)
        PGA.append(scale*np.max(np.abs(gmotion.ground)))

        #print "istep: %s, scale: %s, Sdr: %s" %(istep, scale, a0)  

        if istep == 1:
            if spec_dis > model.dy:
                scale2dy_est = 0.8*scale
            else:
                scale2dy = scale*model.dy/spec_dis
                istep += 1
        else:
            istep += 1
            if spec_dis > df:
                flag = 'REACHED'                

    return (Sdr, Sar, Sat, Sforce, PGA)

#########################
# read groundmotion

g_const = 9.806

# model definition
urml_mean_bin = model_EPP_GA('urmlmean', g_const)

gmotion_path = '/Users/hyeuk/Project/sv01-30'
gmotion_list_file = 'sv_list.csv'
gmotion_psa_file = 'sv_psa.csv'

gmotion_list = pd.read_csv(os.path.join(gmotion_path, gmotion_list_file))
gmotion_psa = np.loadtxt(os.path.join(gmotion_path, gmotion_psa_file), 
    delimiter=',', dtype=float)
gmotion_period = gmotion_psa[0,:]
gmotion_psa = gmotion_psa[1:,:]

for item in gmotion_list.FileName:

    val = gmotion_list.loc[gmotion_list.FileName==item]
    gfile = os.path.join(gmotion_path, item)
    dt_gmotion = val.DT.values[0]
    dt_analysis = val.DT.values[0]

    grecord = gmotion(gfile, dt_gmotion, gmotion_period, 
        gmotion_psa[val.index[0],:])

    (Sdr, Sar, Sat, Sforce, PGA) = compute_IDA(urml_mean_bin, 
        dic_gmotion, 0.0, g_const, incr_sa=0.1, df_factor=10.0)