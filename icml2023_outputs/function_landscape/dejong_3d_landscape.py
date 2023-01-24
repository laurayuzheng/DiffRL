from envs._dejong import DejongEnv
import torch as th
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
import numpy as np


'''
Render function itself
'''
res = 400
x = np.linspace(-1., 1., res)
y = np.linspace(-1., 1., res)
x, y = np.meshgrid(x, y)

nx = x.reshape((-1,))
ny = y.reshape((-1,))
input = th.tensor([nx, ny]).transpose(0, 1)

env = DejongEnv(dim=2, device='cpu')
input = input * env.bound
z = env.evaluate(input)

x = input[:, 0].reshape((res, res)).cpu().numpy()
y = input[:, 1].reshape((res, res)).cpu().numpy()
z = z.reshape((res, res)).cpu().numpy()

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
fig.set_figwidth(6.4 * 1.4)
fig.set_figheight(6.4 * 1.4)

ls = LightSource(270, 45)

surf = ax.plot_surface(x, y, z, rstride=1, cmap='rainbow', cstride=1, linewidth=0, antialiased=True, shade=True)

plt.show()