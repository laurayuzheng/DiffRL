from envs._ackley import AckleyEnv
import torch as th
import matplotlib.pyplot as plt


hfont = {'fontname':'Roman'}

params = {'legend.fontsize': 40,
          'figure.figsize': (12 * 1.5, 9 * 1.5),
         'axes.labelsize': 35,
         'axes.titlesize': 20,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
#fig.set_figwidth(12)
#fig.set_figheight(9)

plt.rcParams.update(params)

'''
Render function itself
'''
env = AckleyEnv(dim=1, device='cpu')
x = th.arange(-500, 500, 1) / 500.0
y = env.evaluate(x.unsqueeze(-1))

# plt.rcParams['font.size'] = 14
fig, ax1 = plt.subplots()
# fig.set_figwidth(6.4 * 1.4)
# fig.set_figheight(4.8 * 1.4)

ax1.set_xlabel('x')
ax1.set_ylabel(r'$f(x)$')
ax1.plot(x.cpu().numpy(), y.cpu().numpy(), color='black', linewidth=3., label=r'$f(x)$')

ax1.legend(loc='upper left')

'''
Render probability distribution
'''

ax2 = ax1.twinx()

rand_mu = th.tensor([0.24,])
rand_sigma = th.tensor([0.2])
dist = th.distributions.Normal(rand_mu, rand_sigma)

dx = x
dy = dist.log_prob(dx)
dy = th.exp(dy)

px_str = r'$p(x)$'
ax2.set_ylabel(px_str)
ax2.plot(dx.cpu().numpy(), dy.cpu().numpy(), color='blue', label=r'$\alpha=0$', linewidth=3.)

'''
Render alpha policy
'''
kx = x.clone()
kx.requires_grad = True
ky = env.evaluate(kx.unsqueeze(-1))
ky_sum = ky.sum()
ky_sum.backward()

kx_grad = kx.grad * 0.001
tx = x.clone() + kx_grad


cx = x.cpu().numpy()
fx = []
fy = []
for ccx in cx:
    min = ccx - 5e-4
    max = ccx + 5e-4

    idces = th.where((tx - min) * (tx - max) < 0.)[0]

    if len(idces) > 0:
        fx.append(ccx)
        fy.append(dy[idces].sum().cpu().item())

ax2.plot(fx, fy, color='orange', label=r'$\alpha=1e-3$', linewidth=3.)

kx_grad = kx.grad * 0.002
tx = x.clone() + kx_grad

cx = x.cpu().numpy()
fx = []
fy = []
for ccx in cx:
    min = ccx - 5e-4
    max = ccx + 5e-4

    idces = th.where((tx - min) * (tx - max) < 0.)[0]

    if len(idces) > 0:
        fx.append(ccx)
        fy.append(dy[idces].sum().cpu().item())

ax2.plot(fx, fy, color='red', label=r'$\alpha=2e-3$', linewidth=3.)

ax2.legend(loc='upper right')

# plt.title("Alpha Policy (Ackley Function)", fontweight='bold')
plt.show()