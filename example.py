import balloon_model as bm
import noise as ns
import numpy as N
import pylab as P

steps = 12
sti = N.zeros((steps,2))
sti[:,0] = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
sti[:,1] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
hrf = bm.BalloonModel()
bold = hrf.solve_ode(event_time_course=sti, timesteps=steps)
noise = ns.GN_AR1_recurrent(rho=0.9, sigma1=0.1, size=steps)
voxel = bold+noise
P.figure()
P.plot(voxel)
P.show()
