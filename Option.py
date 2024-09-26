#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci

S0 = 100
r = 0.05
sigma = 0.3
T = 1

nSim = 1000000

rng = np.random.default_rng()
Z = rng.standard_normal(nSim)
logSt = np.log(S0) + (r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z
St = np.exp(logSt)

# Forward Price
ESt = np.mean(St)
SE = sci.sem(St)
CI = sci.norm.interval(0.95, loc=ESt, scale=SE)

print('Forward Price:', ESt)
print('Confidence Interval:', CI)



# %%

# Call Option
K = 120
payoff = (St - K) * (St > K)
price = np.exp(-r * T) * payoff
ECPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=ECPrice, scale=SE)

print('Call Price:', ECPrice)
print('Confidence Interval:', CI)

# %%

# Put Option
K = 80
payoff = (K - St) * (K > St)
price = np.exp(-r * T) * payoff
EPPrice = np.mean(price)
SE = sci.sem(price)
CI = sci.norm.interval(0.95, loc=EPPrice, scale=SE)

print('Put Price:', EPPrice)
print('Confidence Interval:', CI)

# %%

# Portfolio Profit
payoff = St - ESt - (St - 120) * (St > 120) + (80 - St) * (80 > St)
profit = payoff - (EPPrice - ECPrice) * np.exp(r * T)
Eprofit = np.mean(profit)
SE = sci.sem(profit)
CI = sci.norm.interval(0.95, loc=Eprofit, scale=SE)

print('Profit:', Eprofit)
print('Confidence Interval:', CI)


# %%

# Profit Distribution
plt.hist(profit, density=True, bins=100)

# %%
