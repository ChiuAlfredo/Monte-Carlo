# %% V1 使用 for loop (初階)

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

S0 = 100
r = 0.05
sigma = 0.3
T = 0.5

nTimeStep = 100
dt = T / nTimeStep

St = np.zeros(nTimeStep + 1)
St[0] = S0

for i in range(nTimeStep):
    Z = rng.standard_normal()
    St[i + 1] = St[i] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

tt = np.arange(nTimeStep + 1) * dt
plt.plot(tt, St)


# %% V2 向量化 (高階)

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

S0 = 100
r = 0.05
sigma = 0.3
T = 0.5

nTimeStep = 100
dt = T / nTimeStep

Z = rng.standard_normal(nTimeStep)
logReturn = np.zeros(nTimeStep + 1)
logReturn[0] = np.log(S0)
logReturn[1 :] = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.cumsum(logReturn)
St = np.exp(logSt)

tt = np.arange(nTimeStep + 1) * dt
plt.plot(tt, St)

# %% V3 同時模擬多條路徑

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

S0 = 100
r = 0.05
sigma = 0.3
T = 0.5

nTimeStep = 10
nSim = 50

dt = T / nTimeStep
Z = rng.standard_normal((nTimeStep, nSim))
logReturn = np.zeros((nTimeStep + 1, nSim))
logReturn[0, :] = np.log(S0)
logReturn[1 :, :] = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.cumsum(logReturn, axis=0)
St = np.exp(logSt)

tt = np.arange(nTimeStep + 1) * dt
plt.plot(tt, St)

