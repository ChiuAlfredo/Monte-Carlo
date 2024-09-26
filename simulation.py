#%%

import numpy as np
import scipy.stats as sci

S = 100
X = 3
T = 0.5
r = 0.08
q = 0.04
sigma = 0.25

K = 90
H = 95

nSim = 10000
nTimeStep = 10000
dt = T / nTimeStep

rng = np.random.default_rng()
Z = rng.standard_normal((nSim, nTimeStep))
logReturn = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.log(S) + np.cumsum(logReturn, axis=1)
St = np.exp(logSt)
ST = St[:, -1]

#%% down and in

inFlag = np.any(St <= H, axis=1)
cdiPayoff = (ST - K) * (ST > K) * inFlag + X * ~inFlag
cdiPrice = np.exp(-r * T) * cdiPayoff
EcdiPrice = np.mean(cdiPrice)
SE = sci.sem(cdiPrice)
CI = sci.norm.interval(0.95, loc=EcdiPrice, scale=SE)

print('Price:', EcdiPrice)
print('Confidence Interval:', CI)

# %% down and out

flag = (St <= H) * np.arange(1, nTimeStep + 1)
outFlag = np.any(flag, axis=1)
flag[flag == 0] = nTimeStep
outT = np.min(flag, axis=1) * dt
cdoPrice = (ST - K) * (ST > K) * ~outFlag * np.exp(-r * T) + X * outFlag * np.exp(-r * outT)
EcdoPrice = np.mean(cdoPrice)
SE = sci.sem(cdoPrice)
CI = sci.norm.interval(0.95, loc=EcdoPrice, scale=SE)

print('Price:', EcdoPrice)
print('Confidence Interval:', CI)


# %%
