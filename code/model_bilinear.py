import numpy as np

class model_bilinear(object):
    '''SDOF model definition
    (dy, fy): yield point 
    (alpha): k2/k1
    damp_ratio: damping ratio'''

    def __init__(self, mass, period, fy, alpha, damp_ratio):
        self.mass = mass
        self.period = period # elastic period
        self.fy = fy # yield point
        self.alpha = alpha # k2 = alpha*k1
        self.damp_ratio = damp_ratio # damping ratio zi

        self.omega = 2.0*np.pi/self.period
        self.stiff = mass*self.omega**2.0
        self.dy = fy/self.stiff
        self.damp = 2.0*damp_ratio*self.omega*self.mass
        self.stiff_hat = None

    def determine_stiff(self, dres_current, force_current, dis_current):

        fsmax = self.fy + self.alpha*self.stiff*(dis_current - self.dy)
        fsmin = -self.fy + self.alpha*self.stiff*(dis_current + self.dy)

        if ((dres_current > 0) & (force_current >= fsmax)) | (
            (dres_current < 0) & (force_current <= fsmin)):
            self.stiff = self.alpha*self.fy/self.dy
        else:
            self.stiff = self.mass*self.omega**2.0   

    def pushover(self, dis):
        '''elliptical_pushover
            input: dis (array or scalar)
            output: force
        '''
        if isinstance(dis, list):
            dis = np.array(dis)
        elif isinstance(dis*1.0, float):
            dis = np.array([dis*1.0])

        force = np.zeros_like(dis)

        tf = np.abs(dis) <= self.dy
        force[tf] = self.fy/self.dy*dis[tf]

        tf = np.abs(dis) > self.dy 
        force[tf] = np.sign(dis[tf])*(
            self.fy + self.alpha*self.stiff*(np.abs(dis[tf])-self.dy))

        return force

    def hysteresis(self, force_current, dis_current, dis_new):
        ''' ellipitical_hysteresis
        input: fs0: 
                d: 2x1
                backbone
        output: a0, hysteresis
        '''

        fsmax = self.fy + self.alpha*self.stiff*(dis_new - self.dy)
        fsmin = -self.fy + self.alpha*self.stiff*(dis_new + self.dy)

        force_new = force_current + self.stiff*(dis_new - dis_current)

        if force_new > fsmax:
            force_new = fsmax

        if force_new < fsmin:
            force_new = fsmin

        return force_new    