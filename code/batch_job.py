# batch job for URML, pre

import pandas as pd

bldg_params = pd.read_csv('/Users/hyeuk/Project/eqrm/resources/data/building_parameters.csv',
	header=0)

idx = np.where(bldg_params['structure_classification']=='URML')[0] # 33

cs = bldg_params['design_strength'].values[idx]
period = bldg_params['natural_elastic_period'].values[idx] # 0.1
alpha1 = bldg_params['fraction_in_first_mode'].values[idx] # 0.75
alpha2 = bldg_params['height_to_displacement'].values[idx] # 0.75
gamma_ = bldg_params['yield_to_design'].values[idx] # 1
lambda_ = bldg_params['ultimate_to_yield'].values[idx] # 1.5
mu_ = bldg_params['ductility'].values[idx] # 1.5
damp_ratio = bldg_params['damping_Be'].values[idx] 

g_const = 9.806 # g -> m/sec2
ay = cs*gamma_/alpha1
au = lambda_*ay
dy = period/(2.0*np.pi)**2.0*ay
du = lambda_*mu_*dy

#URML_pre = model_elliptical(0.24, 0.2*386.089, 2.4, 0.4*386.089, 0.1)

from run_nonlinear_sdof import *

URML_pre_GA = model_GA(dy, ay, du, au, damp_ratio, g_const)

#structure_class	structure_classification	design_strength	height	natural_elastic_period	fraction_in_first_mode	height_to_displacement	yield_to_design	ultimate_to_yield	ductility	damping_Be
#BUILDING	URML	0.15	15	0.1	0.75	0.75	1	1.5	1.5	0.05


dy_eq = au/ay*dy

URML_pre_bi = model_bilinear(1.0, period, dy_eq, 0.0, damp_ratio)
