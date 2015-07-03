import numpy as np
import time
from scipy.interpolate import interp1d
#import timeit
#import pdb
#pdb.set_trace()

def newmark_parameters(model, dt_comp, flag_newmark):

    # Newmark parameters definition
    # -----------------------------
    CONST_GAMMA = 0.5
    if flag_newmark == 'LA': 
        CONST_BETA = 1.0/6.0   # linear acceleration (stable if Dt/T<=0.551)
        assert dt_comp/model.period <= 0.551, "Newmakr method is unstable"
    elif flag_newmark == 'AA':
        CONST_BETA = 0.25   # average acceleration (unconditionally stable)

    CONST_A = (1.0/(CONST_BETA*dt_comp)*model.mass
     + CONST_GAMMA/CONST_BETA*model.damp)
    CONST_B = 1.0/(2.0*CONST_BETA)*model.mass + dt_comp*(
        CONST_GAMMA/(2.0*CONST_BETA)-1.0)*model.damp

    return (CONST_A, CONST_B, CONST_BETA, CONST_GAMMA)

def external_force(grecord, dt_comp):

    ts = np.linspace(0.0, grecord.npts*grecord.dt, grecord.npts*int(
        grecord.dt/dt_comp))
    flin = interp1d(grecord.ts, grecord.ground)
    eforce = flin(ts)

    return eforce

def linear_response_sdof(model, grecord, grecord_factor, dt_comp, 
    flag_newmark='LA'):

    '''
    compute response of SDOF model 
    input: model (class)
    input: grecord (class)
    output: spectra values
    '''

    # read earthquake force
    # grecord factor should include -1 if it's ground acceleration
    if grecord.dt > dt_comp:
        eforce = grecord_factor * external_force(grecord, dt_comp)
    else:
        eforce = grecord_factor * grecord.ground 
    npts = len(eforce)

    (CONST_A, CONST_B, CONST_BETA, CONST_GAMMA) = newmark_parameters(
        model, dt_comp, flag_newmark)

    dis = np.zeros((npts))
    vel = np.zeros((npts))
    acc = np.zeros((npts))

    # Initial calculations
    # --------------------
    acc[0] = (eforce[0] - model.damp*vel[0] - model.stiff*dis[0])

    model.stiff_hat = model.stiff + CONST_A/dt_comp

    # Time stepping (Table 5.7.2)
    # -------------
    for i in range(npts-1):

        # 2.1 
        delta_phat = eforce[i+1]-eforce[i] + CONST_A*vel[i] + CONST_B*acc[i]

        # 2.2 determine the tangent stiffness ki   
        delta_dis = delta_phat/model.stiff_hat

        # 2.3 
        delta_vel = CONST_GAMMA/(CONST_BETA*dt_comp)*delta_dis -(
            CONST_GAMMA/CONST_BETA*vel[i]) + dt_comp*(
            1.0-CONST_GAMMA/(2.0*CONST_BETA))*acc[i]

        # 2.4
        delta_acc = 1.0/(CONST_BETA*dt_comp**2)*delta_dis - 1.0/(
            CONST_BETA*dt_comp)*vel[i] - 1.0/(2.0*CONST_BETA)*acc[i]  # alternative

        # 2.7
        dis[i+1] = dis[i] + delta_dis
        vel[i+1] = vel[i] + delta_vel
        #acc[i+1] = acc[i] + delta_acc
        acc[i+1] = (eforce[i+1] - model.damp*vel[i+1] - model.stiff*dis[i+1])/model.mass

        #   Dai = 1/(BETA*Dt^2)*Ddi - 1/(BETA*Dt)*v(i,:) - 1/(2*BETA)*a(i,:)  # alternative
        #   a(i+1,:) = a(i,:) + Dai  # alternative
       
    # Spectral values
    # ---------------
    tacc = acc -eforce/model.mass
    spec_dis = np.max(abs(dis), axis=0)
    spec_vel = np.max(abs(vel), axis=0)
    spec_acc = np.max(abs(acc), axis=0)
    spec_tacc = np.max(abs(tacc), axis=0)  
    
    return (dis, vel, acc, tacc, (spec_dis, spec_vel, spec_acc, spec_tacc))

def nonlinear_response_sdof(model, grecord, grecord_factor, dt_comp, flag_newmark='LA'):

    '''
    compute response of SDOF model with a hysteresis
    input: model (class)
    input: grecord (class)
    output: spectra values
    '''

    # read earthquake force
    # grecord factor should include -1 if it's ground acceleration
    if grecord.dt > dt_comp:
        eforce = grecord_factor * external_force(grecord, dt_comp)
    else:
        eforce = grecord_factor * grecord.ground 
    npts = len(eforce)

    (CONST_A, CONST_B, CONST_BETA, CONST_GAMMA) = newmark_parameters(
        model, dt_comp, flag_newmark)

    dis = np.zeros((npts))
    vel = np.zeros((npts))
    acc = np.zeros((npts))
    force = np.zeros((npts))

    # Initial calculations
    # --------------------
    acc[0] = (eforce[0] - model.damp*vel[0] - model.stiff*dis[0])

    # Time stepping (Table 5.7.2)
    # -------------
    for i in range(npts-1):
        #print "Time: %s" %i
        # 2.1 
        delta_phat = eforce[i+1]-eforce[i] + CONST_A*vel[i] + CONST_B*acc[i]

        # determine ki
        model.determine_stiff(delta_phat, force[i], dis[i])

        model.stiff_hat = model.stiff + CONST_A/dt_comp

        # 2.2 determine the tangent stiffness ki   
        (delta_dis, force[i+1]) = modified_Newton_Raphson_method(dis[i], 
            force[i], delta_phat, model)

        # 2.5 
        delta_vel = CONST_GAMMA/(CONST_BETA*dt_comp)*delta_dis -(
            CONST_GAMMA/CONST_BETA*vel[i]) + dt_comp*(
            1.0-CONST_GAMMA/(2.0*CONST_BETA))*acc[i]

        # 2.7
        dis[i+1] = dis[i] + delta_dis
        vel[i+1] = vel[i] + delta_vel
        acc[i+1] = (eforce[i+1] - model.damp*vel[i+1] - force[i+1])/model.mass

        #   Dai = 1/(BETA*Dt^2)*Ddi - 1/(BETA*Dt)*v(i,:) - 1/(2*BETA)*a(i,:)  # alternative
        #   a(i+1,:) = a(i,:) + Dai  # alternative
       
    # Spectral values
    # ---------------
    spec_dis = np.max(abs(dis), axis=0)
    spec_vel = np.max(abs(vel), axis=0)
    spec_acc = np.max(abs(acc), axis=0)
    spec_tacc = np.max(abs(acc -eforce/model.mass), axis=0)  
    spec_force = np.max(abs(force), axis=0)
    
    return (dis, vel, acc, force, (spec_dis, spec_vel, spec_acc, spec_tacc, spec_force))

def modified_Newton_Raphson_method(dis_current, force_current, dres_current, 
    model):

    # Table 5.7.1. from Chopra Book
    CONST_TOL = 1e-8
    max_iter = 20
    incr_ratio = 1.0
    j = 0

    delta_u = np.zeros((max_iter, 1))
    delta_u[0] = dres_current/model.stiff_hat

    while (incr_ratio > CONST_TOL) & (j < max_iter):
                  
        #print "deltaD: %s, incr_ratio: %s" %(delta_u[j], incr_ratio)          
        dis_new = dis_current + delta_u[j]

        # determine force(j)
        force_new = model.hysteresis(force_current, dis_current, 
            dis_new)

        dres_new = dres_current - (force_new - force_current + (
            model.stiff_hat-model.stiff)*delta_u[j])

        delta_u[j+1] = dres_new/model.stiff_hat # 2.1

        incr_ratio = delta_u[j+1]/np.sum(delta_u)

        j += 1 # increase index

        # update dis, force
        dis_current = dis_new
        force_current = force_new
        dres_current = dres_new

    return (np.sum(delta_u), force_new)

# if __name__ == '__main__':
#     #conv_g_inPersec2 = 386.089
#     #hazus_URML_pre = model_elliptical(0.24, 0.2*conv_g_inPersec2, 2.4, 0.4*conv_g_inPersec2, 
#     #    0.1)
#     el_centro = grecord('../data/El_Centro_Chopra.npy', 0.02, 0.02, factor=386.089*1.0)
#     #(dis, vel, acc, force) = nonlinear_response_sdof(hazus_URML_pre, el_centro)    

#     start_time = time.time()
#     #half_sine = grecord(np.array([[0.0, 5.0, 8.6602, 10.0, 8.6603, 5.0, 0.0, 0.0, 0.0, 0.0]]).T, 0.1, 0.1, factor=-1.0)
#     #bilinar0 = model_bilinear(0.2533, 1.0, 0.75, 0.00, 0.01266873347011487)
#     bilinar0 = model_bilinear(1.0, 0.35, 0.24, 0.05, 0.1)
#     #(dis, vel, acc, force) = nonlinear_response_sdof(bilinar0, half_sine, flag_newmark='AA')  
#     (dis, vel, acc, force) = nonlinear_response_sdof(bilinar0, el_centro, flag_newmark='AA')  

#     print("--- %s seconds ---" % (time.time() - start_time))
