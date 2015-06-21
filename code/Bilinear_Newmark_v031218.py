import numpy as np
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt

class model(object):

    def __init__(self, dy, ay, du, au, damp_ratio):
        self.mass = 1.0
        self.dy = dy # yield point
        self.ay = ay
        self.du = du # ultimate point
        self.au = au
        self.period = 2.0*np.pi*np.sqrt(self.dy/self.ay) # elastic period        
        self.omega = 2.0*np.pi/self.period
        self.damp_ratio = damp_ratio # damping ratio zi
        self.damp = 2.0*damp_ratio*self.omega
        self.stiff = self.omega**2.0
        self.stiff_hat = None
        self.hysteresis = {'ref_d0': 0.0, 'Iunloading': 0}

        # compute backbone curve parameters: B and C
        bb_1 = (-dy*(au**2) - du*(ay**2) + ay*au*(dy+du))
        bb_2 = ay*(du-dy) + 2.0*dy*(ay-au)
        bb = bb_1/bb_2
        cc_1 = (bb**2)*(dy-du)**2
        cc_2 = (bb**2) - (ay-au+bb)**2

        self.ellip_B = bb
        self.ellip_C = np.sqrt(cc_1/cc_2)

        # compute backbone curve parameters: poly_a, poly_b, poly_c
        self.poly_c = au
        self.poly_b = (ay/dy)/(au-ay)
        self.poly_a = (ay-au)*np.exp(self.poly_b*dy)

    def elliptical_pushover(self, disp):
        '''elliptical_pushover
            input: disp (array or scalar)
            output: force
        '''
        if isinstance(disp, list):
            disp = np.array(disp)
        elif isinstance(disp*1.0, float):
            disp = np.array([disp*1.0])

        force = np.zeros_like(disp)

        tf = np.abs(disp) <= self.dy
        force[tf] = self.ay/self.dy*disp[tf]

        tf = (np.abs(disp) > self.dy) & (np.abs(disp) < self.du)
        force[tf] = np.sign(disp[tf])*(
            self.au-self.ellip_B+self.ellip_B*np.sqrt(
            1.0 -((self.du-np.abs(disp[tf]))/self.ellip_C)**2))

        tf = np.abs(disp) >= self.du
        force[tf] = self.au * np.sign(disp[tf])

        return force

    # def polynomial_pushvoer(self, disp):
        
    #     if isinstance(disp, list):
    #         disp = np.array(disp)
    #     elif isinstance(disp*1.0, float):
    #         disp = np.array([disp*1.0])

    #     force = np.zeros_like(disp)

    #     tf = np.abs(disp) <= self.dy
    #     force[tf] = self.ay/self.dy*disp[tf]

    #     tf = (np.abs(disp) > self.dy) & (np.abs(disp) < self.du)
    #     force[tf] = np.sign(disp[tf])*(
    #         self.poly_c + self.poly_a*np.exp(-self.poly_b*np.abs(disp[tf])))

    #     tf = np.abs(disp) >= self.du
    #     force[tf] = self.au * np.sign(disp[tf])

    #     return force

    # def elastoplastic_pushvoer(self, disp):
        
    #     if isinstance(disp, list):
    #         disp = np.array(disp)
    #     elif isinstance(disp*1.0, float):
    #         disp = np.array([disp*1.0])

    #     force = np.zeros_like(disp)

    #     tf = np.abs(disp) <= self.dy
    #     force[tf] = self.ay/self.dy*disp[tf]

    #     tf = np.abs(disp) > self.dy
    #     force[tf] = np.sign(disp[tf])*()

    #     tf = np.abs(disp) >= self.du
    #     force[tf] = self.au * np.sign(disp[tf])

    #     return force

class gmotion(object):

    def __init__(self, gfile, dt_gmotion, dt_analysis, factor=386.089):

        acc_gm = np.load(gfile)
        npts = len(acc_gm)
        ts_gm = np.arange(0, npts*dt_gmotion, dt_gmotion)
        ts_ef = np.arange(0, npts*dt_gmotion, dt_analysis) 
        flin = interp1d(ts_gm, -factor*acc_gm[:,0])
        self.eforce = flin(ts_ef)
        self.ts = ts_ef
        self.dt = dt_analysis

def nonlinear_response_sdof(model, gmotion, flag_newmark='LA'):

    '''
    flag_newmark = 'LA'
    T = 0.35
    z = 0.1
    dy = 0.24 # inch
    ay = 0.2 # g
    du = 2.4 # inch
    au = 0.4 # g
    conv_g_inPersec2 = 386.089
    ag = np.load('../data/El_Centro_Chopra.npy')*conv_g_inPersec2
    dtg = 0.02 # sec 
    dt = 0.02 # sec 
    '''

    # Newmark parameters definition
    # -----------------------------
    CONST_GAMMA = 0.5
    if flag_newmark == 'LA': 
        CONST_BETA = 1.0/6.0   # linear acceleration (stable if Dt/T<=0.551)
    elif flag_newmark == 'AA':
        CONST_BETA = 0.25   # average acceleration (unconditionally stable)

    # read earthquake force    
    eforce = gmotion.eforce
    npts = len(eforce)

    CONST_A = (1.0/(CONST_BETA*gmotion.dt)*model.mass
     + CONST_GAMMA/CONST_BETA*model.damp)
    CONST_B = 1.0/(2.0*CONST_BETA)*model.mass + gmotion.dt*(
        CONST_GAMMA/(2.0*CONST_BETA)-1.0)*model.damp

    disp = np.zeros((npts, 1))
    vel = np.zeros((npts, 1))
    acc = np.zeros((npts, 1))
    force = np.zeros((npts, 1))

    # Initial calculations
    # --------------------
    acc[0] = (eforce[0] - model.damp*vel[0] - force[0])
    model.stiff_hat = model.stiff + CONST_A/gmotion.dt

    # Time stepping (Table 5.7.2)
    # -------------
    for i in range(npts-1):
        print "Time: %s" %i
        # 2.1 
        delta_phat = eforce[i+1]-eforce[i] + CONST_A*vel[i] + CONST_B*acc[i]

        # 2.2 determine the tangent stiffness ki   
        (delta_disp, force[i+1]) = modified_Newton_Raphson_method(disp[i], 
            force[i], delta_phat, model)

        # 2.5 
        delta_vel = CONST_GAMMA/(CONST_BETA*gmotion.dt)*delta_disp -(
            CONST_GAMMA/CONST_BETA*vel[i]) + gmotion.dt*(
            1.0-CONST_GAMMA/(2.0*CONST_BETA))*acc[i]

        # 2.7
        disp[i+1] = disp[i] + delta_disp
        vel[i+1] = vel[i] + delta_vel
        acc[i+1] = (eforce[i+1] - model.damp*vel[i+1] - force[i+1])/model.mass

        #   Dai = 1/(BETA*Dt^2)*Ddi - 1/(BETA*Dt)*v(i,:) - 1/(2*BETA)*a(i,:)  # alternative
        #   a(i+1,:) = a(i,:) + Dai  # alternative
       
    # Spectral values
    # ---------------
    spec_disp = np.max(abs(disp), axis=0)
    spec_vel = np.max(abs(vel), axis=0)
    spec_acc = np.max(abs(acc), axis=0)
    spec_tacc = np.max(abs(acc + (-eforce/model.mass)[:,np.newaxis]), axis=0)  
    spec_force = np.max(abs(force), axis=0)
    
    return (disp, vel, acc, force)

def modified_Newton_Raphson_method(disp_current, force_current, dres_current, 
    model):

    # Table 5.7.1. from Chopra Book
    CONST_TOL = 1e-4
    max_iter = 100
    incr_ratio = 1.0
    j = 0

    delta_u = np.zeros((max_iter, 1))
    delta_u[0] = dres_current/model.stiff_hat

    while (incr_ratio > CONST_TOL) & (j < max_iter):
                  
        print "deltaD: %s, incr_ratio: %s" %(delta_u[j], incr_ratio)          
        disp_new = disp_current + delta_u[j]

        # determine force(j)
        force_new = ellipitical_hysteresis(force_current, disp_current, 
            disp_new, model)

        dres_new = dres_current - (force_new - force_current + (
            model.stiff_hat-model.stiff)*delta_u[j])

        delta_u[j+1] = dres_new/model.stiff_hat # 2.1

        incr_ratio = delta_u[j+1]/np.sum(delta_u)

        j += 1 # increase index

        # update disp, force
        disp_current = disp_new
        force_current = force_new
        dres_current = dres_new

    return (np.sum(delta_u), force_new)

def ellipitical_hysteresis(force_current, disp_current, disp_new, model):
    ''' ellipitical_hysteresis
    input: fs0: 
            d: 2x1
            backbone
    output: a0, hysteresis
    '''

    disp_incr = disp_new - disp_current
    force_new = force_current + disp_incr*model.stiff

    # Incipient Yielding
    if ((force_new > model.ay) & (force_current <= model.ay) & (
        model.hysteresis['Iunloading']  != 1) | 
        (force_new < -model.ay) & (force_current >= -model.ay) & (
        model.hysteresis['Iunloading'] != -1)):
      
        model.hysteresis['ref_d0'] = disp_current - force_current/model.stiff
        model.hysteresis['Iunloading'] = 0

        print 'Incipient: fs0:%s, fs1: %s, d0: %s d1: %s' %(force_current, force_new,
            disp_current, disp_new)

    # Yielding
    elif (np.abs(force_new) > model.ay) & (np.sign(disp_incr) == np.sign(
        force_new)):
        force_ =  model.elliptical_pushover(disp_new-model.hysteresis['ref_d0'])
        print 'Yielding: fs_new=%s, force_:%s' %(force_new, force_)
        force_new = np.sign(force_) * min(np.abs(force_new), np.abs(force_))

    # Unloading
    elif (np.abs(force_current) > model.ay) & (np.sign(disp_incr) != np.sign(
        force_current)):
        model.hysteresis['Iunloading'] = np.sign(force_new) 
        print ('Unloading')  

    return force_new

if __name__ == '__main__':
    conv_g_inPersec2 = 386.089
    hazus_URML_pre = model(0.24, 0.2*conv_g_inPersec2, 2.4, 0.4*conv_g_inPersec2, 
        0.1)
    el_centro = gmotion('../data/El_Centro_Chopra.npy', 0.02, 0.02,factor=386.089*2.0)
    (disp, vel, acc, force) = nonlinear_response_sdof(hazus_URML_pre, el_centro)    
