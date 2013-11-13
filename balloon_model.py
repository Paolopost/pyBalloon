"""Simulation of the hemodynamic response through the
ballon model. See Buxton (1998), Friston (2000), Zheng (2002).

The implementation presented here derives directly from the
set of differential equations and notation available in Zheng (2002)
(page 622).
"""

# OPEN ISSUES:
# *) add CCA fMRI ODEs solution as method, to compare results
# *) where are the intial values in the Zheng (2002) paper?

import numpy as N
import scipy.integrate

class ModelParameter(object):
    """One parameter of the Balloon model: its value(s) and
    description.
    """
    def __init__(self, name=None, default_value=0.0, interval_half_width=0.0, description=""):
        self.name = name
        if self.name == None: # parameter needs a name!
            raise Exception
        self.default_value = default_value # remember default value
        self.interval_half_width = interval_half_width
        self.min = self.default_value-self.interval_half_width
        self.max = self.default_value+self.interval_half_width
        self.description = description
        self.value = self.default_value # actual value
        pass

    def __call__(self):
        """Allow the syntax self.parameter() to get self.parameter.value .
        """
        return self.value

    def random(self):
        """Generate one (or more) random values for the parameter
        according to the uniform distribution defined by [min,max).
        """
        self.value = N.random.uniform(self.min,self.max)
        return

    def __str__(self):
        """Informative printout of the parameter contents.
        """
        return "%s, value=%s, default_value=%s, range=[%s,%s), description='%s: %s'" % (self.name,str(self.value),str(self.default_value),str(self.min),str(self.max),self.name,self.description)

    pass


class BalloonModel(object):
    """The Balloon model deascribed by Buxton (1998)
    and then extended by Friston (2000) and Zheng (2002).

    See Zheng at al., A Model of the Hemodynamic Response and Oxygen
    Delivery to Brain, Neuroimage 2002.
    """

    def __init__(self):
        self.default_values()
        pass

    def default_values(self):
        """Set default values for parameters of balloon model.

        Values are taken from Zheng (2002) and CCA fMRI software.
        """
        self.CaB = ModelParameter("C^a_B",1.0,0.0,"Blood oxygen concentration in the arterial end of capillary")
        self.epsilon = ModelParameter("epsilon",0.5,0.15,"neuronal efficacy")
        self.tau_s = ModelParameter("tau_s",1.2,0.3,"signal decay")
        self.tau_f = ModelParameter("tau_f",2.4,0.5,"autoregulation")
        self.E_0 = ModelParameter("E_0",0.4,0.1,"baseline value of oxygen extraction fraction")
        self.tau_0 = ModelParameter("tau_0",1.0,0.5,"venous time constant")
        self.T = ModelParameter("T",1.0,0.5,"mean capillary transit time")
        self.r = ModelParameter("C_P/C_B",0.01,0.005,"ratio between oxygen concentration in plasma (C_P) and total oxygen concentration in blood (C_B)")
        self.vol_ratio = ModelParameter("Vtis/Vcap",75.0,25.0,"ratio between the volume of blood in tissue (Vtis) and the volume of blood in capillary (Vcap)")
        self.K = ModelParameter("K",0.1,0.05,"constant of metabolic demand")
        self.alpha = ModelParameter("alpha",0.3,0.1,"Grubb's exponent (stiffness)")
        self.g_0 = ModelParameter("g_0 = C_T/C^a_P",0.1,0.05,"baseline value of tissue oxygen concentration")
        self.J = ModelParameter("J",5.0,0.0,"scaling of the tissue oxygen concentration")
        self.V_0 = ModelParameter("V_0",0.02,0.0,"TODO")

        pass


    def random(self):
        """Clever trick to run .random() on all ModelParameter
        instances inside this object
        """
        for i in list(self.__dict__):
            if isinstance(self.__dict__[i], ModelParameter):
                self.__dict__[i].random()
                pass
            pass
        return


    def __str__(self):
        """Printout of all parameters of the balloon model.
        """
        s = "Balloon Model\nParameters values\n"
        for i in list(self.__dict__):
            if isinstance(self.__dict__[i], ModelParameter):
                s += "\t- "+str(self.__dict__[i])+"\n"
                pass
            pass
        return s


    def ODE_ballon(self,Y,protocol_time):
        """The set of differential equations of the balloon model.
        """
        # get the current value of the stimulation protocol:
        if self.sp:
            current_event = self.stimulation_protocol.stimulus(protocol_time)
        else:
            try:
                current_event = self.event_time_course[N.floor(protocol_time/self.TR),1]
            except IndexError:
                current_event = 0.0
            pass
        
        s,f_in,E,CB,g,v,q = Y
        f_out = v**(1.0/self.alpha())

        # ODEs:
        ds_dt = self.epsilon() * current_event - s / self.tau_s() - (f_in - 1.0) / self.tau_f()
        df_in_dt = s
        dE_dt = f_in  / self.phi * (-E + (1.0 - g) * (1.0 - (1.0 - (self.E_0() / (1.0 - self.g_0()))) ** (1.0 / f_in)))
        dCB_dt = f_in / self.phi * (-CB - self.CaB()*E / N.log(1.0 - E / (1.0 - g)) + self.CaB()*g)
        dg_dt = (self.E_0() / (self.J() * self.vol_ratio() * self.r() * self.T())) * (((CB - g*self.CaB()) / (self.CB_0 - self.g_0()*self.CaB()) - 1.0) - self.K() * current_event)
        dv_dt = (1.0 / self.tau_0()) * (f_in - f_out)
        dq_dt = (1.0 / self.tau_0()) * (f_in * (E / self.E_0()) - f_out * (q / v))
        return N.array([ds_dt,df_in_dt,dE_dt,dCB_dt,dg_dt,dv_dt,dq_dt])


    def solve_ode(self, event_time_course=None, stimulation_protocol=None, timesteps=None):
        """Solve the balloon model ODEs.
        """
        self.sp = False
        if stimulation_protocol is not None:
            self.sp = True
            self.stimulation_protocol = stimulation_protocol
            self.timesteps = timesteps
        elif event_time_course is not None:
            self.event_time_course = event_time_course
            self.TR = self.event_time_course[1,0]-self.event_time_course[0,0]
            # print self.TR
            pass
        else:
            raise Exception
        

        # define some more constants (See Zheng (2002)):
        self.CB_0 = -self.E_0() / N.log(1.0-self.E_0()/(1.0-self.g_0())) + self.g_0() # baseline value of spatial mean total blood oxygen concentration
        self.phi = 0.15 * self.T()
                
        # ODEs initial values, taken from Zheng (2002) and CCA fMRI:
        s = 0.0
        f_in = 1.0
        E = self.E_0()
        CB = self.CB_0
        g = self.g_0()
        v = 1.0
        q = 1.0

        Y_0 = N.array([s,f_in,E,CB,g,v,q])

        if self.sp:
            # self.critical_timesteps = N.sort(N.hstack([stimulation_protocol.onset,stimulation_protocol.onset+stimulation_protocol.duration])) # USING ONSETS AND OFFSETS RAISES AN ERROR IN ODEINT!!!
            self.critical_timesteps = stimulation_protocol.onset # WE JUST FEED ONSENTS
            # hmax = stimulation_protocol.duration.min()/2.0 # NOT USED ANYMORE
            Y, self.infodict = scipy.integrate.odeint(self.ODE_ballon, Y_0, self.timesteps, ixpr=True, full_output=True, tcrit=self.critical_timesteps, printmessg=True)
            
        else:
            Y = scipy.integrate.odeint(self.ODE_ballon, Y_0, self.event_time_course[:,0], hmax=0.1, ixpr=True)
            pass

        # REMEMBER TO SWITCH ARGS IN ODE_balloon IF YOU USE THIS:
        # t0 = self.event_time_course[0,0]
        # result = scipy.integrate.ode(derivative,jac=None).set_integrator('vode').set_initial_value(Y0,t0)
        # Y = N.zeros((self.event_time_course.shape[0],7))
        # Y[0,:] = Y0
        # for t in event_time_course[1:,0]:
        #     if result.successful():
        #         result.integrate(t)
        #         # print result.t
        #         # print result.y
        #         Y[result.t,:] = result.y
        #     else:
        #         print "ERROR: Cannot find solution!"
        #         break
        #     pass

        # Extract results:
        s = Y[:,0]
        f_in = Y[:,1]
        E = Y[:,2]
        CB = Y[:,3]
        g = Y[:,4]
        v = Y[:,5]
        q = Y[:,6]

        # Compute bold (omit rescaling to V_0):
        bold = (7.0 * self.E_0() * (1.0 - q) + 2.0 * (1.0 - q / v) + (2.0 * self.E_0() - 0.2) * (1.0 - v)) # *self.V_0()

        return bold
    

if __name__ == "__main__":

    N.random.seed(1)

    import pylab as P

    import stimulation_protocol

    bm = BalloonModel()
    # bm.random()
    
    # Set a protocol
    TR = 0.5
    experiment_duration = 101.0
    event_time_course = N.zeros((experiment_duration/TR, 2))
    event_time_course[10/TR:11/TR,1] = 1.0
    event_time_course[12/TR:13/TR,1] = 1.0
    event_time_course[17/TR:18/TR,1] = 1.0
    event_time_course[19/TR:20/TR,1] = 1.0
    event_time_course[:,0] = N.arange(0,experiment_duration, TR)

    bold = bm.solve_ode(event_time_course)
    # print bold2
    P.plot(bold)

    # bm.E_0.value = 0.3
    # bm.tau_f.value = 2.0
    # bm.epsilon.value = 0.65

    # for i in range(10):
    #     bm.random()
    #     bold = bm.solve_ode(event_time_course)
    #     P.plot(bold)
    #     pass

    sp = stimulation_protocol.StimulationProtocol()
    sp.random(how_many=20,total_duration=200.0, duration=3.0, random_duration=True)
    event_time_course = sp.generate_event_time_course(total_duration=200.0, TR=1.0)
    event_time_course2 = sp.generate_event_time_course2(total_duration=200.0, TR=1.0)
    ts = N.arange(0,200.0,1.0)
    bold = bm.solve_ode(event_time_course)
    bold2 = bm.solve_ode(event_time_course2)
    bold3 = bm.solve_ode(stimulation_protocol=sp, timesteps=ts)

    P.figure()
    P.plot(event_time_course[:,0],event_time_course[:,1], label="stimuli")
    P.plot(event_time_course[:,0],bold,label="BOLD (stimuli TR)")
    P.plot(event_time_course[:,0],bold2,label="BOLD (stimuli TR 2)")
    P.plot(ts,bold3,label="BOLD (stimuli exact)")
    P.ylim(ymin=-0.5,ymax=max(1.5,bold.max()*1.3))
    P.legend()
    P.title("BOLD (balloon model) with different stimuli generators")
    P.xlabel("time (sec.)")
    

