import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt

plt.close('all')

env = xt.Environment()

env.new('quad', 'Quadrupole', length=0.3)
env.new('sext', 'Sextupole', length=0.2)
env.new('oct', 'Octupole', length=0.1)

l1 =env.new_line(components=[
    env.new('mb1', 'Bend', angle=np.pi/4, length=1, at=5),
    env.place(['quad', 'sext', 'oct']),
    env.new('mb2', 'Bend', angle=-np.pi/4, length=1, at=10),
    env.place(['quad', 'sext', 'oct']),
    env.new('end', 'Marker', at=15)
])

sv1 = l1.survey()
sv1.plot()
sv1.plot(projection='XZ')


env.new('mm', 'Multipole', length=0.3)
l2 = env.new_line(components=[
    env.new('m1', 'Multipole', hxl=np.pi/4, length=1, at=5),
    env.place('mm', at=7),
    env.new('m2', 'Multipole', hxl=-np.pi/4, length=1, at=10),
    env.place('mm', at=12),
    env.new('ee', 'Marker', at=15)
])

sv2 = l2.survey()
sv2.plot()

env.new('bend_v', 'Bend', rot_s_rad=np.pi/2)
l3 =env.new_line(components=[
    env.new('mbv1', 'bend_v', angle=np.pi/4, length=1, at=5),
    env.place(['quad', 'sext', 'oct']),
    env.new('mbv2', 'bend_v', angle=-np.pi/4, length=1, at=10),
    env.place(['quad', 'sext', 'oct']),
    env.new('ev', 'Marker', at=15)
])
sv3 = l3.survey()
sv3.plot(projection='ZY')

l4 = env.new_line(components=[
    env.new('mv1', 'Multipole', hxl=np.pi/4, rot_s_rad=np.pi/2,
            length=1, at=5),
    env.place('mm', at=7),
    env.new('mv2', 'Multipole', hxl=-np.pi/4, rot_s_rad=np.pi/2,
            length=1, at=10),
    env.place('mm', at=12),
    env.new('eev', 'Marker', at=15)
])
sv4 = l4.survey()
sv4.plot(projection='ZY')
