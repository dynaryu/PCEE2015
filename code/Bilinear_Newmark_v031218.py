import numpy as np
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt

def nonlinear_response_SDOF(T, z, dy, ay, du, au, ag, dtg, dt, flag_newmark='LA'):

    '''
    flag_newmark = 'LA'
    T = 0.35
    z = 0.1
    dy = 0.24 # inch
    ay = 0.2 # g
    du = 2.4 # inch
    au = 0.4 # g
    conv_g_inPersec2 = 386.089
    ag = np.load('/Users/hyeuk/Project/PCEE2015/data/El_Centro_Chopra.npy')*conv_g_inPersec2
    dtg = 0.02 # sec 
    dt = 0.02 # sec 
    '''
    #define [ S, varargout ] = Bilinear_Newmark_v031218( T, z, dy, alpha, ag, Dtg, Dt )

    #
    # function [ S, varargout ] = Bilinear_Newmark_v031218( T, z, dy, alpha, ag, Dtg, Dt )
    #
    # Author:  Nicolas Luco
    # Last Revised:  18 December 2003
    # Reference:  "Dynamics of Structures" (1995) by A.K. Chopra 
    # python coded by Hyeuk Ryu
    # 
    # INPUT:
    # ======
    # T     = periods                           ( 1 x num_oscillators, or scalar )
    # z     = damping ratios                    ( " )
    # dy    = yield displacements               ( " )
    # alpha = strain hardening ratios           ( " )
    # ag    = ground acceleration time history  ( length(ag) x 1 )
    # Dtg   = time step of ag                   ( scalar )
    # Dt    = analyis time step                 ( scalar )
    #
    # OUTPUT:
    # =======
    # S.d  = relative displacement spectrum               ( 1 x num_oscillators ) 
    #  .v  = relative velocity spectrum                   ( " )
    #  .a  = acceleration spectrum                        ( " )
    # varargout{1} = E, where ...
    # E.K  = kinetic energy (at end)                      ( 1 x num_oscillators )
    #  .D  = energy dissipated by damping (at end)        ( " )
    #  .S  = strain energy (at end)                       ( " )
    #  .Y  = energy dissipated by yielding (at end)       ( " )
    #  .I  = total input energy (at end)                  ( " )
    # varargout{2} = H, where ...
    # H.d  = relative displacement response time history  ( length(ag) x num_oscillators ) 
    #  .v  = relative velocity response time history      ( " )
    #  .a  = acceleration response time history           ( " )
    #  .fs = force response time history                  ( " )
    #

    # Newmark parameters definition
    # -----------------------------
    GAMMA = 1.0/2.0
    if flag_newmark == 'LA': 
        BETA = 1.0/6.0   # linear acceleration (stable if Dt/T<=0.551)
    elif flag_newmark == 'AA':
        BETA = 1.0/4.0   # average acceleration (unconditionally stable)

    # m*a + c*v + fs(k,fy,kalpha) = p
    # -------------------------------
    m = 1.0
    w = 2.0*np.pi/T
    c = z*(2.0*m*w)
    k = np.power(w,2.0)*m

    # backbone parameters BB and CC
    bb_1 = -dy*(au**2) - du*(ay**2) + ay*au*(dy+du)
    bb_2 = ay*(du-dy) + 2*dy*(ay-au)
    bb = bb_1/bb_2
    cc_1 = (bb**2)*(dy-du)**2
    cc_2 = (bb**2) - (ay-au+bb)**2
    cc = np.sqrt(cc_1/cc_2)

    backbone = {}
    backbone['dy'] = dy
    backbone['ay'] = ay
    backbone['du'] = du
    backbone['au'] = au
    backbone['B'] = bb
    backbone['C'] = cc

    # Interpolate p=-ag*m (linearly)
    # ------------------------------
    tg = np.arange(0,ag.shape[0])*dtg 
    t = np.arange(0,tg[-1]+dt,dt)
    flin = interp1d(tg,-ag[:,0]*m)
    p = flin(t)

    # Memory allocation & initial conditions
    # --------------------------------------
    lp = len(p)
    disp = np.zeros((lp, 1))
    vel = np.zeros((lp, 1))
    acc = np.zeros((lp, 1))
    fs = np.zeros((lp, 1))

    hysteresis = {}
    hysteresis['ref_d0'] = 0.0
    hysteresis['Iunloading'] = 0

    # Initial calculations
    # --------------------
    acc[0] = (p[0] - c*vel[0] - fs[0])/m
    A = 1.0/(BETA*dt)*m + GAMMA/BETA*c
    B = 1.0/(2.0*BETA)*m + dt*(GAMMA/(2.0*BETA)-1.0)*c

    # Time stepping
    # -------------
    for i in range(lp-1):

        # 2.1 
        DPi = p[i+1]-p[i] + A*vel[i] + B*acc[i]

        # 2.2 determine the tangent stiffness ki
        ki = k

        ki_hat = ki + A/dt
    
        (Ddi, fs[i+1], hysteresis) = ellipitcal_Newton_Raphson( disp[i], fs[i], 
            DPi, ki_hat, ki, backbone, hysteresis )

        # 2.5 
        Dvi = GAMMA/(BETA*dt)*Ddi - GAMMA/BETA*vel[i] + dt*(
            1.0-GAMMA/(2.0*BETA))*acc[i]

        # 2.7
        disp[i+1] = disp[i] + Ddi
        vel[i+1] = vel[i] + Dvi
        acc[i+1] = (p[i+1] - c*vel[i+1] - fs[i+1])/m

        #   Dai = 1/(BETA*Dt^2)*Ddi - 1/(BETA*Dt)*v(i,:) - 1/(2*BETA)*a(i,:)  # alternative
        #   a(i+1,:) = a(i,:) + Dai  # alternative
       

    # Spectral values
    # ---------------
    Sdr = np.max(abs(disp),axis=0)
    Svr = np.max(abs(vel),axis=0)
    Sar = np.max(abs(acc),axis=0)
    Sat = np.max(abs(acc + (-p/m)[:,np.newaxis]), axis=0)  # Note: ag itself was not interpolated
    Sforce = np.max(abs(fs),axis=0)
    
    return (Sdr, Sar, Sat, Sforce, disp, vel, acc, fs)

def ellipitical_pushover(d, backbone):
    '''ellipitical_pushover
    input: d (displacement) 2x1
            backbone (dict) 
    output: a (force)

    '''

    Dy = backbone['dy']
    Du = backbone['du']
    Au = backbone['au']
    B = backbone['B']
    C = backbone['C']

    if np.abs(d) <= Dy:
        a = Ay/Dy*d

    elif (np.abs(d) > Dy) & (np.abs(d) < Du):
        a = np.sign(d)* (Au - B + B * np.sqrt(1.0 -((Du-np.abs(d))/C)**2))

    else:    
        a = Au * np.sign(d)
    
    return(a)   

def ellipitical_hysteresis(fs0, d, backbone, hysteresis):
    ''' ellipitical_hysteresis
    input: fs0: 
            d: 2x1
            backbone
    output: a0, hysteresis
    '''

    Dy = backbone['dy']
    Du = backbone['du']
    Ay = backbone['ay']
    Au = backbone['au']
    B = backbone['B']
    C = backbone['C']

    ka = Ay/Dy

    fs = np.zeros((2,1));
    fs[0] = fs0;

    Dd = d[1] - d[0]
    fs[1] = fs[0] + Dd * ka

    # Incipient Yielding
    if ( (fs[1]> Ay) & (fs[0]<=Ay) & (hysteresis['Iunloading']  != 1) | 
         (fs[1]< -Ay) & (fs[0]>= -Ay) & (hysteresis['Iunloading'] != -1) ):
      
        temp = (d[1]-d[0])/(fs[1]-fs[0])
      
        hysteresis['ref_d0'] = d[0] + (
            (np.sign(fs[1])*Ay)- fs[0]) * temp - np.sign(fs[1])*Dy       
        hysteresis['Iunloading'] = 0
  
    # Yielding
    elif (np.abs(fs[1])>Ay) & (np.sign(Dd) == np.sign(fs[1])):

        tmp_a =  ellipitical_pushover( d[1]-hysteresis['ref_d0'], backbone )
        fs[1] = np.sign(tmp_a) * min( np.abs(fs[1]), np.abs(tmp_a) )
  
    # Unloading
    elif (np.abs(fs[0])>Ay)  &  (np.sign(Dd) != np.sign(fs[0])):
        hysteresis['Iunloading'] = np.sign(fs[1]) 

    return(fs, hysteresis)

def ellipitcal_Newton_Raphson(disp_i, fs_i, DPi, kt_hat, kt, backbone, 
    hysteresis):

    # Table 5.7.1. from Chopra Book
    TOL = 1e-4
    max_iter = 100
    incr_ratio = 1.0
    j = 0

    # initialize data
    Dd = [] # Delta u(j)

    disp = np.zeros((2,1))
    disp[0] = disp_i

    # fs
    fs = np.zeros((2,1))
    fs[0] = fs_i

    # dr
    dr = np.zeros((max_iter,1))
    dr[0] = DPi

    while (incr_ratio > TOL) & (j < max_iter):

        Dd.append(dr[j] / kt_hat) # 2.1
                  
        disp[1] = disp[0] + Dd[j]

        # determine fs(j+1,:)
        (fs_temp, hysteresis) = ellipitical_hysteresis( fs[0], disp, 
            backbone, hysteresis )

        fs[1] = fs_temp[1]

        df = fs[1] - fs[0] + (kt_hat-kt)*Dd[j]

        dr[j+1] = dr[j] - df

        incr_ratio = Dd[j]/sum(Dd)

        # update disp
        disp[0] = disp[1]
        fs[0] = fs[1]

        # increas index
        j += 1

    return(sum(Dd), fs[1], hysteresis)
