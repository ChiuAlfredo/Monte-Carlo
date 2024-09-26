# %%

import numpy as np
from numpy.polynomial import polynomial as poly
import scipy.stats as sci

def bsFormula(S, K, T, r, q, sigma, isCall):
    v = (sigma * np.sqrt(T))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / v
    d2 = d1 - v
    sign = 2 * isCall - 1
    price = sign * S * np.exp(-q * T) * sci.norm.cdf(sign * d1) - sign * K * np.exp(-r * T) * sci.norm.cdf(sign * d2)
    return price

rng = np.random.default_rng()

# %% 1. American

S = 42
K = 40
r = 0.04
q = 0.08
sigma = 0.35
T = 9 / 12

nSim = 10000
nTimeStep = 10000
dt = T / nTimeStep
discount = np.exp(-r * dt)

Z = rng.standard_normal((nSim, nTimeStep))
logReturn = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.log(S) + np.cumsum(logReturn, axis=1)
St = np.exp(logSt)

ST = St[:, -1]
price = np.maximum(ST - K, 0)
for i in range(2, nTimeStep + 1):
    price = discount * price
    ST = St[:, -i]
    STIn = ST[ST > K]
    if STIn.size == 0:
        continue
    X = STIn
    Y = price[ST > K]
    coef = poly.polyfit(X, Y, 2)
    execV = np.maximum(STIn - K, 0)
    holdV = poly.polyval(STIn, coef)
    price[ST > K][execV > holdV] = execV[execV > holdV]

price = discount * price
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 2. Holder-Extendible

S = 100
K1 = 100
K2 = 105
r = 0.08
q = 0
sigma = 0.25
T1 = 6 / 12
T2 = 9 / 12
X = 1

nSim = 10000000

Z = rng.standard_normal(nSim)
logST1 = np.log(S) + (r - q - 0.5 * sigma ** 2) * T1 + sigma * np.sqrt(T1) * Z
ST1 = np.exp(logST1)

execV = np.maximum(ST1 - K1, 0)
holdV = bsFormula(ST1, K2, T2 - T1, r, q, sigma, True) - X
payoff = np.maximum(execV, holdV)

price = payoff * np.exp(-r * T1)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 3. Writer-Extendible

S = 80
K1 = 90
K2 = 82
r = 0.1
q = 0
sigma = 0.3
T1 = 6 / 12
T2 = 9 / 12

nSim = 10000000

Z = rng.standard_normal(nSim)
logST1 = np.log(S) + (r - q - 0.5 * sigma ** 2) * T1 + sigma * np.sqrt(T1) * Z
ST1 = np.exp(logST1)

execV = np.maximum(ST1 - K1, 0)
holdV = bsFormula(ST1, K2, T2 - T1, r, q, sigma, True)
payoff = execV * (ST1 >= K1) + holdV * (ST1 < K1)

price = payoff * np.exp(-r * T1)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 4. Forward Start

S = 60
r = 0.08
q = 0.04
sigma = 0.3
T1 = 3 / 12
T2 = 1
alpha = 1.1

nSim = 10000000

Z = rng.standard_normal(nSim)
logST1 = np.log(S) + (r - q - 0.5 * sigma ** 2) * T1 + sigma * np.sqrt(T1) * Z
ST1 = np.exp(logST1)

K = alpha * ST1
payoff = bsFormula(ST1, K, T2 - T1, r, q, sigma, True)

price = payoff * np.exp(-r * T1)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 5. Asian

S = 6.8
K = 6.9
r = 0.07
q = 0.09
sigma = 0.14
T = 6 / 12

nSim = 10000
nTimeStep = 10000
dt = T / nTimeStep

Z = rng.standard_normal((nSim, nTimeStep))
logReturn = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.log(S) + np.cumsum(logReturn, axis=1)
St = np.exp(logSt)

SAV = np.mean(St, axis=1)
payoff = np.maximum(K - SAV, 0)

price = payoff * np.exp(-r * T)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 6. Chooser

S = 50
Kc = 55
Kp = 48
r = 0.1
q = 0.05
sigma = 0.35
T1 = 3 / 12
Tc = 6 / 12
Tp = 7 / 12

nSim = 10000000

Z = rng.standard_normal(nSim)
logST1 = np.log(S) + (r - q - 0.5 * sigma ** 2) * T1 + sigma * np.sqrt(T1) * Z
ST1 = np.exp(logST1)

callV = bsFormula(ST1, Kc, Tc - T1, r, q, sigma, True)
putV = bsFormula(ST1, Kp, Tp - T1, r, q, sigma, False)
payoff = np.maximum(callV, putV)

price = payoff * np.exp(-r * T1)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 7. Loocback

S = 120
K = 100
r = 0.1
q = 0.06
sigma = 0.3
T = 6 / 12

nSim = 10000
nTimeStep = 10000
dt = T / nTimeStep

Z = rng.standard_normal((nSim, nTimeStep))
logReturn = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.log(S) + np.cumsum(logReturn, axis=1)
St = np.exp(logSt)
ST = St[:, -1]

Smin = np.min(St, axis=1)
K = np.minimum(Smin, K)
payoff = np.maximum(ST - K, 0)

price = payoff * np.exp(-r * T)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 8. Compound

Su = 500
Ku = 520
K = 50
r = 0.08
q = 0.03
sigma = 0.35
T = 3 / 12
Tu = 6 / 12

nSim = 10000000

Z = rng.standard_normal(nSim)
logSt = np.log(Su) + (r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z
St = np.exp(logSt)

C = bsFormula(St, Ku, Tu - T, r, q, sigma, True)
payoff = np.maximum(K - C, 0)

price = payoff * np.exp(-r * T)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 9. Look-Barrier

S = 100
K = 100
H = 120
r = 0.1
q = 0
sigma = 0.3
T1 = 6 / 12
T2 = 1

nSim = 10000
nTimeStep = 10000

dt = T1 / nTimeStep
Z = rng.standard_normal((nSim, nTimeStep))
logReturn = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.log(S) + np.cumsum(logReturn, axis=1)
St = np.exp(logSt)

outFlag = np.any(St >= H, axis=1)
ST1 = St[:, -1][~outFlag]

dt = (T2 - T1) / nTimeStep
Z = rng.standard_normal((ST1.size, nTimeStep))
logReturn = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.reshape(np.log(ST1), (-1, 1)) + np.cumsum(logReturn, axis=1)
St = np.exp(logSt)

Smax = np.max(St, axis=1)
payoff = np.maximum(Smax - K, 0)
payoff = np.pad(payoff, (0, nSim - payoff.size), 'constant', constant_values=0)

price = payoff * np.exp(-r * T2)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)

# %% 10. Double-Barrier Binary

S = 100
L = 80
H = 120
r = 0.05
q = 0.02
sigma = 0.3
T = 3 / 12
X = 10

nSim = 10000
nTimeStep = 10000
dt = T / nTimeStep

Z = rng.standard_normal((nSim, nTimeStep))
logReturn = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
logSt = np.log(S) + np.cumsum(logReturn, axis=1)
St = np.exp(logSt)

outFlag = np.any(St >= H, axis=1) | np.any(St <= L, axis=1)
payoff = np.zeros(nSim)
payoff[~outFlag] = X

price = payoff * np.exp(-r * T)
EPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPrice, scale=SE)

print('Price:', EPrice)
print('Confidence Interval:', CI)


# %%
