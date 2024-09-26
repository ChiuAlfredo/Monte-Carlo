#%%

import numpy as np
import matplotlib.pyplot as plt

S0 = 100
r = 0.05
q = 0.02
T = 0.5

kappa = 2
theta = 0.09
sigma = 0.4
v0 = 0.1
rho = 0.6

#%%

rng = np.random.default_rng()

nTimeStep = 180
nSim = 10
dt = T / nTimeStep

ZS = rng.standard_normal((nTimeStep, nSim))
ZV = rho * ZS + np.sqrt(1 - rho ** 2) * rng.standard_normal((nTimeStep, nSim))

V = np.zeros((nTimeStep, nSim))
V[0, :] = v0
for i in range(nTimeStep - 1):
    V[i + 1, :] = (np.sqrt(V[i, :]) + 0.5 * sigma * np.sqrt(dt) * ZV[i, :]) ** 2 + kappa * (theta - V[i, :]) * dt - 0.25 * sigma ** 2 * dt

logReturn = np.zeros((nTimeStep + 1, nSim))
logReturn[0, :] = np.log(S0)
logReturn[1 :, :] = (r - q - 0.5 * V) * dt + np.sqrt(V * dt) * ZS
logSt = np.cumsum(logReturn, axis=0)
St = np.exp(logSt)

#%%

tt = np.arange(nTimeStep + 1) * dt
plt.plot(tt, St)
plt.show()


# %%