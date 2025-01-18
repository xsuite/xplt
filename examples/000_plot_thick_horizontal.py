import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt

plt.close('all')

env = xt.Environment()

l1 =env.new_line(components=[
    env.new('mb1', 'Bend', angle=np.pi/4, length=1, at=5),
    env.new('mb2', 'Bend', angle=-np.pi/4, length=1, at=10),
    env.new('end', 'Marker', at=15)
])

sv1 = l1.survey()
sv1.plot()


l2 = env.new_line(components=[
    env.new('m1', 'Multipole', knl=[np.pi/4], hxl=np.pi/4, length=1, at=5),
    env.new('m2', 'Multipole', knl=[-np.pi/4], hxl=-np.pi/4, length=1, at=10),
    env.new('ee', 'Marker', at=15)
])

sv2 = l2.survey()
sv2.plot()