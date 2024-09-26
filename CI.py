# %% 

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci

rng = np.random.default_rng()

S0 = 100
r = 0.05
sigma = 0.3
T = 0.5

nSim = 1000000

Z = rng.standard_normal(nSim)
logSt = np.log(S0) + (r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z
St = np.exp(logSt)

plt.hist(St, density=True, bins=100)

mvsk = sci.describe(St)

print('Simulation Mean:', mvsk[2])
print('Simulation Variance:', mvsk[3])
print('Simulation Skewness:', mvsk[4])
print('Simulation Kurtosis:', mvsk[5])

mu = np.log(S0) + (r - 0.5 * sigma ** 2) * T
v = sigma * np.sqrt(T)

x = np.linspace(0, 300, 1001)
f = sci.lognorm.pdf(x, v, scale=np.exp(mu))
plt.plot(x, f)

mean, var, skew, kurt = sci.lognorm.stats(v, scale=np.exp(mu), moments='mvsk')
print('Mean:', mean)
print('Variance:', var)
print('Skewness:', skew)
print('Kurtosis:', kurt)

# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci

rng = np.random.default_rng()

S0 = 100
r = 0.05
sigma = 0.3
T = 0.5

nSim = 1000000

Z = rng.standard_normal(nSim)
logSt = np.log(S0) + (r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z
St = np.exp(logSt)

ESt = np.mean(St)
SE = sci.sem(St)
CI = sci.norm.interval(0.95, loc=ESt, scale=SE)

mu = np.log(S0) + (r - 0.5 * sigma ** 2) * T
v = sigma * np.sqrt(T)
ExESt = sci.lognorm.moment(1, sigma * np.sqrt(T), scale=np.exp(mu))

print('Mean:', ESt)
print('Confidence Interval:', CI)
print('Exact Mean:', ExESt)

# %%

import numpy as np
import scipy.stats as sci

rng = np.random.default_rng()

nSim = 100000

xy = rng.uniform(size=(nSim, 2))
Z = np.sum(xy ** 2, axis=1) < 1

EPi = np.mean(4 * Z)
SE = sci.sem(4 * Z)
CI = sci.norm.interval(0.95, loc=EPi, scale=SE)

print('Mean:', EPi)
print('Confidence Interval:', CI)


# %%
