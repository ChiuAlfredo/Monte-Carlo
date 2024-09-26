#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
from scipy.stats import norm
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

#買進履約價 780 元買權
call_price_780 = black_scholes_call(800, 780, 1/12, 0.05, 0.2)
#賣出履約價 790 元買權
call_price_790 = black_scholes_call(800, 790, 1/12, 0.05, 0.2)
#賣出履約價 810 元買權
call_price_810 = black_scholes_call(800, 810, 1/12, 0.05, 0.2)
#買進履約價 820 元買權
call_price_820 = black_scholes_call(800, 820, 1/12, 0.05, 0.2)

print(f'現在四個合約價格分別為{call_price_780}, {call_price_790}, {call_price_810}, {call_price_820}')


value = -call_price_780 + call_price_790 + call_price_810 - call_price_820

print(f'組合的價值為{value}')


def payoff(S, K1, K2, K3, K4):
  return max(0, S - K1) - max(0, S - K2) - max(0, S - K3) + max(0, S - K4)

Stock_range = np.linspace(750, 850, 100)

profolio_payoff = [payoff(S, 780, 790, 810, 820) for S in Stock_range]

# 繪製報酬函數圖形
plt.plot(Stock_range, profolio_payoff)
plt.xlabel('S')
plt.ylabel('Payoff')
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci

S0 = 800
r = 0.05
sigma = 0.2
T = 1/12

nSim = 100000

rng = np.random.default_rng()
Z = rng.standard_normal(nSim)
logSt = np.log(S0) + (r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z
St = np.exp(logSt)

# Forward Price
ESt = np.mean(St)
SE = sci.sem(St)
CI = sci.norm.interval(0.95, loc=ESt, scale=SE)

# print('Forward Price:', ESt)
# print('Confidence Interval:', CI)



# %%
def call_option(K):
    """_summary_

    Args:
        K (s): _description_

    Returns:
        _type_: _description_
    """    
    # Call Option
    # K = 780
    payoff = (St - K) * (St > K)
    price = np.exp(-r * T) * payoff
    ECPrice = np.mean(price)
    SE = sci.sem(price)
    CI = sci.norm.interval(0.95, loc=ECPrice, scale=SE)
    
    return {'price':ECPrice, 'CI':CI}

# call_780 = call_option(780)
# print('Call Price:', call_780['price'])
# print('Confidence Interval:', call_780['CI'])
    

# %%

# Put Option
def put_option(K):
    """_summary_

    Args:
        K (s): _description_

    Returns:
        _type_: _description_
    """    
    # K = 780
    payoff = (K - St) * (K > St)
    price = np.exp(-r * T) * payoff
    EPPrice = np.mean(price)
    SE = sci.sem(price)
    CI = sci.norm.interval(0.95, loc=EPPrice, scale=SE)
    
    return {'price':EPPrice, 'CI':CI}

# put_890 = put_option(890)
# print('Put Price:', put_890['price'])
# print('Confidence Interval:', put_890['CI'])
call_780 = call_option(780)
call_790 = call_option(790)
call_810 = call_option(810)
call_820 = call_option(820)
#%%
print(f'蒙地卡羅法的四個合約價格分別為{call_780["price"]},信心去區間為{call_780["CI"]},\n {call_790["price"]},信心去區間為{call_790["CI"]},\n {call_810["price"]},信心去區間為{call_810["CI"]},\n {call_820["price"]},信心去區間為{call_820["CI"]}')


#%%

# %%
payoff = (St-780) * (St > 780) + (790 - St) * (St>790) + (810 - St) * (St>810) +(St-820) * (St > 820)

# %%

# Portfolio Profit
# payoff = St - ESt - (St - 120) * (St > 120) + (80 - St) * (80 > St)
profit = payoff - (call_option(790)['price'] -call_option(810)['price']-call_option(780)['price']+call_option(820)['price']) * np.exp(r * T)
Eprofit = np.mean(profit)
SE = sci.sem(profit)
CI = sci.norm.interval(0.95, loc=Eprofit, scale=SE)

print('Profit:', Eprofit)
print('Confidence Interval:', CI)


# %%

# Profit Distribution
plt.hist(profit, density=True, bins=100)

# %%
