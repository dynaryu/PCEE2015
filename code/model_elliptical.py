class model_elliptical(object):
    '''SDOF model definition
    (dy, fy): yield point 
    (du, fu): ultimate point
    damp_ratio: damping ratio'''

    def __init__(self, dy, fy, du, fu, damp_ratio):
        self.mass = 1.0
        self.dy = dy # yield point
        self.fy = fy
        self.du = du # ultimate point
        self.fu = fu
        self.period = 2.0*np.pi*np.sqrt(self.dy/self.fy) # elastic period        
        self.omega = 2.0*np.pi/self.period
        self.damp_ratio = damp_ratio # damping ratio zi
        self.damp = 2.0*damp_ratio*self.omega
        self.stiff = self.omega**2.0
        self.stiff_hat = None
        self.ref_d0 = 0.0
        self.Iunloading = 0

        # compute backbone curve parameters: B and C
        bb_1 = (-dy*(fu**2) - du*(fy**2) + fy*fu*(dy+du))
        bb_2 = fy*(du-dy) + 2.0*dy*(fy-fu)
        bb = bb_1/bb_2
        cc_1 = (bb**2)*(dy-du)**2
        cc_2 = (bb**2) - (fy-fu+bb)**2

        self.ellip_B = bb
        self.ellip_C = np.sqrt(cc_1/cc_2)

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

        tf = (np.abs(dis) > self.dy) & (np.abs(dis) < self.du)
        force[tf] = np.sign(dis[tf])*(
            self.fu-self.ellip_B+self.ellip_B*np.sqrt(
            1.0 -((self.du-np.abs(dis[tf]))/self.ellip_C)**2))

        tf = np.abs(dis) >= self.du
        force[tf] = self.fu * np.sign(dis[tf])

        return force

    def determine_stiff(self):
        pass

    def hysteresis(self, force_current, dis_current, dis_new):
        ''' ellipitical_hysteresis
        input: fs0: 
                d: 2x1
                backbone
        output: a0, hysteresis
        '''

        dis_incr = dis_new - dis_current
        force_new = force_current + dis_incr*self.stiff

        # Incipient Yielding
        if ((force_new > self.fy) & (force_current <= self.fy) & (
            self.Iunloading  != 1) | 
            (force_new < -self.fy) & (force_current >= -self.fy) & (
            self.Iunloading != -1)):
          
            self.ref_d0 = dis_current - force_current/self.stiff
            self.Iunloading = 0

            force_new =  self.pushover(dis_new-self.ref_d0)

            #print 'Incipient: fs0:%s, fs1: %s, d0: %s d1: %s' %(force_current, force_new,
            #    dis_current, dis_new)

        # Yielding
        elif (np.abs(force_new) > self.fy) & (np.sign(dis_incr) == np.sign(
            force_new)):
            force_ =  self.pushover(dis_new-self.ref_d0)
            #print 'Yielding: fs_new=%s, force_:%s' %(force_new, force_)
            force_new = np.sign(force_) * min(np.abs(force_new), np.abs(force_))

        # Unloading
        elif (np.abs(force_current) > self.fy) & (np.sign(dis_incr) != np.sign(
            force_current)):
            self.Iunloading = np.sign(force_new) 
            #print ('Unloading')  

        return force_new    