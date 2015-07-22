import pandas as pd
import numpy as np

from run_nonlinear_sdof import nonlinear_response_sdof

def IDA_limit(model, grecord, period, dt_comp, g_const, incr_sa=0.1,
    df_factor=20.0):

    df_limit = df_factor*model.dy
    idx_model_period = np.where(grecord.period == model.period)[0]
    scale2dy_est = model.fy/(grecord.psa[idx_model_period]*g_const)

    idx_period = np.where(grecord.period == period)[0]
    sv_psa = grecord.psa[idx_period] # g scale
    incr_scale = incr_sa/sv_psa

    result = []

    istep = 1
    flag = 'NOT_REACHED'
    while flag == 'NOT_REACHED':
        # if istep == 1:
        #     scale = scale2dy_est
        # else:
        #     scale = scale2dy_est + (istep-1)*incr_scale

        scale = scale2dy_est + (istep-1)*incr_scale

        response = nonlinear_response_sdof(model, grecord, -1.0*scale*g_const,\
            dt_comp)
        spec = np.abs(response).max()
        spec = spec.append(pd.Series(scale, index=[u'scale']))
        result.append(spec)

        #print "istep: %s, scale: %s, Sdr: %s" %(istep, scale, a0)

        if istep == 1:
            if spec.dis > model.dy:
                scale2dy_est = 0.8*scale
            else:
                scale2dy_est = scale*model.dy/spec.dis
                istep += 1
        else:
            istep += 1
            if spec.dis > df_limit:
                flag = 'REACHED'

    df_result = pd.DataFrame(result)

    return df_result

def IDA_scale(model, grecord, period, dt_comp, g_const, im_range):

    idx_model_period = np.where(grecord.period == model.period)[0]
    scale2dy = 0.8*model.fy/(grecord.psa[idx_model_period]*g_const)

    idx_period = np.where(grecord.period == period)[0]

    scale_factor = im_range/grecord.psa[idx_period]
    result = []

    for scale in scale_factor[scale_factor > scale2dy]:

        response = nonlinear_response_sdof(model, grecord, -1.0*scale*g_const,\
            dt_comp)
        spec = np.abs(response).max()
        spec = spec.append(pd.Series(np.array([scale,\
            scale*grecord.psa[idx_period]]), index=[u'scale', u'im']))
        result.append(spec)

    df_result = pd.DataFrame(result)

    return df_result

def predict_segmented(x, regression_params):

    (interc, slope1, slope2, break_pt) = regression_params

    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x*1.0, float):
        x = np.array([x*1.0])

    y = np.zeros_like(x)

    tf = (np.log(x) <= break_pt)
    y[tf] = interc + slope1*np.log(x[tf])

    tf = (np.log(x) > break_pt)
    y[tf] = interc + slope1*break_pt + slope2*(np.log(x[tf])-break_pt)

    return np.exp(y)

