import numpy as np
import pandas as pd
from model_bilinear import model_bilinear

def compare_bldg_params_HAZUS_GA(input_file_GA, bldg_types_GA,
    input_file_HAZUS, bldg_types_HAZUS):

    if input_file_GA is None:    
        input_file_GA = '/Users/hyeuk/Project/eqrm/resources/data/building_parameters.csv'
    
    if input_file_HAZUS is None:
        input_file_HAZUS ='/Users/hyeuk/Project/eqrm/resources/data/building_parameters_hazus.csv'

    bldg_params_GA = pd.read_csv(input_file_GA)
    bldg_params_HAZUS = pd.read_csv(input_file_HAZUS)

    if (bldg_types_GA is None) or (bldg_types_HAZUS is None):
        bldg_types_GA= np.intersect1d(bldg_params_HAZUS.\
            structure_classification,
        bldg_params_GA.structure_classification)
        bldg_types_HAZUS = bldg_types_GA

    col_names = list(np.intersect1d(bldg_params_HAZUS.columns,\
        bldg_params_GA.columns))
    [col_names.remove(x) for x in ['structure_class',\
        'structure_classification']] 

    for (bldg_GA, bldg_HAZUS) in zip(bldg_types_GA, bldg_types_HAZUS):

        for col in col_names:

            value_GA = bldg_params_GA.loc[bldg_params_GA.\
                structure_classification==bldg_GA][col].values
            value_HAZUS = bldg_params_HAZUS.loc[bldg_params_HAZUS.\
                structure_classification==bldg_HAZUS][col].values

            if not np.isclose(value_HAZUS, value_GA):
                print "%s of %s for GA: %s vs. %s for HAZUS: %s" %(
                    col, bldg_GA, value_GA, bldg_HAZUS, value_HAZUS)

def read_EQRM_bldg_params(bldg_type, g_const,
    EQRM_input_file = '/Users/hyeuk/Project/eqrm/resources/data/building_parameters.csv'):

    bldg_params = pd.read_csv(EQRM_input_file)

    idx = np.where(bldg_params.structure_classification == bldg_type.upper())[0]

    if len(idx) > 0:

        # idx 50 for URMLMEAN
        cs = bldg_params['design_strength'].values[idx]
        period = bldg_params['natural_elastic_period'].values[idx] # 0.1
        alpha1 = bldg_params['fraction_in_first_mode'].values[idx] # 0.75
        alpha2 = bldg_params['height_to_displacement'].values[idx] # 0.75
        gamma_ = bldg_params['yield_to_design'].values[idx] # 1
        lambda_ = bldg_params['ultimate_to_yield'].values[idx] # 1.5
        mu_ = bldg_params['ductility'].values[idx] # 1.5
        damp_ratio = bldg_params['damping_Be'].values[idx]

        ay = cs*gamma_/alpha1
        au = lambda_*ay
        dy = ay*g_const*(period/(2.0*np.pi))**2.0
        du = lambda_*mu_*dy

    else:
        print('There is no %s.' %bldg_type)

    return (dy, ay*g_const, du, au*g_const, damp_ratio, period)

def model_EPP_GA(bldg_type, g_const):

    (dy, fy, du, fu, damp_ratio, period) = read_EQRM_bldg_params(bldg_type, g_const)

    # creat equivalent EPP system
    return model_bilinear(mass=1.0, period=period, fy=fu, alpha=0.0, damp_ratio=damp_ratio)

class model_GA(object):
    '''SDOF model definition
    (dy, fy): yield point
    (alpha): k2/k1
    damp_ratio: damping ratio'''

    def __init__(self, dy, fy, du, fu, damp_ratio):
        '''dy, du (meter)
           fy, fu (g)
           conv_factor
        '''
        self.mass = 1.0
        self.dy = dy # yield point
        self.fy = fy
        self.du = du # ultimate point
        self.fu = fu
        self.damp_ratio = damp_ratio # damping ratio zi

        self.stiff = self.fy/dy

        self.CONST_B = fy/(dy*(fu-fy))
        self.CONST_A = (self.fy-self.fu)*np.exp(self.CONST_B*dy)

        self.period = 2.0*np.pi*np.sqrt(dy/fy) # elastic period
        self.omega = 2.0*np.pi/self.period
        self.damp = 2.0*damp_ratio*self.omega
        self.stiff_hat = None
        self.ref_d0 = 0.0
        self.Iunloading = 0

    def pushover(self, dis):
        ''' pushover
            input: displacement
            output: g scale
        '''
        if isinstance(dis, list):
            dis = np.array(dis)
        elif isinstance(dis*1.0, float):
            dis = np.array([dis*1.0])

        val = (self.du/self.dy)/(self.fu/self.fy-1)
        if  val < 10:
            print "GAP %s" %val

        force = np.zeros_like(dis)

        tf = np.abs(dis) <= self.dy
        force[tf] = self.stiff*dis[tf]

        tf = (np.abs(dis) > self.dy) & (np.abs(dis) < self.du)
        force[tf] = np.sign(dis[tf])*(
            self.fu + self.CONST_A*np.exp(-self.CONST_B*np.abs(dis[tf])))

        tf = np.abs(dis) >= self.du
        force[tf] = self.fu * np.sign(dis[tf])

        return force


