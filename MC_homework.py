#%%
import numpy as np

# Parameters
S0 = 42          # Current stock price
K = 40           # Strike price
r = 0.04         # Risk-free interest rate
q = 0.08         # Dividend yield
sigma = 0.35     # Volatility
T = 9 / 12       # Time to expiry in years
n = 1000         # Number of time steps

# Calculate parameters
dt = T / n
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u
p = (np.exp((r - q) * dt) - d) / (u - d)

# Initialize asset prices at maturity
asset_prices = np.zeros((n + 1, n + 1))
for i in range(n + 1):
    for j in range(i + 1):
        asset_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)

# Initialize option values at maturity
option_values = np.zeros((n + 1, n + 1))
option_values[:, n] = np.maximum(0, asset_prices[:, n] - K)

# Step back through the tree
for i in range(n - 1, -1, -1):
    for j in range(i + 1):
        continuation_value = np.exp(-r * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
        option_values[j, i] = np.maximum(asset_prices[j, i] - K, continuation_value)

# Option price is the value at the root of the tree
option_price = option_values[0, 0]
print(f"第一題：American Call Option Price: {option_price:.2f}")
#%%
import numpy as np

# Parameters
S0 = 100              # Initial stock price
K1 = 100              # Initial strike price
K2 = 105              # Extended strike price
r = 0.08              # Risk-free interest rate
sigma = 0.25          # Volatility
T1 = 0.5              # Initial time to maturity (6 months)
T2 = 0.75             # Extended time to maturity (9 months)
extension_fee = 1     # Extension fee
n_simulations = 1000000  # Number of Monte Carlo simulations

# Generate random numbers for stock prices
np.random.seed(42)
Z1 = np.random.normal(size=n_simulations)

Z2 = np.random.normal(size=n_simulations)

# Simulate stock prices at T1
ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T1 + sigma * np.sqrt(T1) * Z1)

# Payoff at T1 without extension
payoff_T1 = np.maximum(ST1 - K1, 0)

# Find which paths are out-of-the-money at T1
out_of_the_money = payoff_T1 == 0

# Simulate stock prices at T2 for those out-of-the-money at T1
ST2 = ST1[out_of_the_money] * np.exp((r - 0.5 * sigma**2) * (T2 - T1) + sigma * np.sqrt(T2 - T1) * Z2[out_of_the_money])

# Payoff at T2 with extension
payoff_T2 = np.zeros(n_simulations)
payoff_T2[out_of_the_money] = np.maximum(ST2 - K2, 0)

# Combine payoffs and consider extension fee
final_payoff = np.where(out_of_the_money, payoff_T2 * np.exp(-r * (T2 - T1)) - extension_fee, payoff_T1)

# Discount final payoffs to present value
discounted_payoff = final_payoff * np.exp(-r * T1)

# Calculate option price
option_price = np.mean(discounted_payoff)
print(f"第二題：Extendible Call Option Price (Monte Carlo): {option_price:.2f}")

#%%
import numpy as np

# Parameters
S0 = 80                # Initial stock price
K1 = 90                # Initial strike price
K2 = 82                # Extended strike price
r = 0.10               # Risk-free interest rate
sigma = 0.30           # Volatility
T1 = 6 / 12            # Initial time to maturity (6 months)
T2 = 9 / 12            # Extended time to maturity (9 months)
n_simulations = 1000000  # Number of Monte Carlo simulations

# Time increments
np.random.seed(42)
Z1 = np.random.normal(size=n_simulations)
Z2 = np.random.normal(size=n_simulations)

# Simulate stock prices at T1
ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T1 + sigma * np.sqrt(T1) * Z1)

# Payoff at T1 without extension
payoff_T1 = np.maximum(ST1 - K1, 0)

# Find which paths are out-of-the-money at T1
out_of_the_money = ST1 < K1

# Simulate stock prices at T2 for those out-of-the-money at T1
ST2 = ST1[out_of_the_money] * np.exp((r - 0.5 * sigma**2) * (T2 - T1) + sigma * np.sqrt(T2 - T1) * Z2[out_of_the_money])

# Payoff at T2 with extension
payoff_T2 = np.zeros(n_simulations)
payoff_T2[out_of_the_money] = np.maximum(ST2 - K2, 0)

# Combine payoffs
final_payoff = np.where(out_of_the_money, payoff_T2 * np.exp(-r * (T2 - T1)), payoff_T1)

# Discount final payoffs to present value
discounted_payoff = final_payoff * np.exp(-r * T1)

# Calculate option price
option_price = np.mean(discounted_payoff)
print("第三題：Writer-Extendible Call Option Price (Monte Carlo):", option_price)
#%%
import numpy as np

# Parameters
S0 = 60          # Current stock price
r = 0.08         # Risk-free rate
q = 0.04         # Continuous dividend yield
sigma = 0.30     # Volatility
T_start = 3 / 12 # Time to start in years
T_maturity = 1   # Total time to maturity in years from today
T = T_maturity - T_start  # Time from start to maturity
n_simulations = 1000000  # Number of Monte Carlo simulations

# Generate random numbers for stock prices at T_start
np.random.seed(12)
Z_start = np.random.normal(size=n_simulations)

# Simulate stock prices at T_start
ST_start = S0 * np.exp((r - q - 0.5 * sigma**2) * T_start + sigma * np.sqrt(T_start) * Z_start)

# Calculate the strike price at T_start (10% out-of-the-money)
strike_price = ST_start * 1.10

# Generate random numbers for stock prices at T_maturity
Z_maturity = np.random.normal(size=n_simulations)

# Simulate stock prices at T_maturity from T_start
ST_maturity = ST_start * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_maturity)

# Calculate option payoffs at maturity
payoffs = np.maximum(ST_maturity - strike_price, 0)

# Discount payoffs to present value
discounted_payoffs = payoffs * np.exp(-r * T_maturity)

# Calculate option price
option_price = np.mean(discounted_payoffs)
print("第四題：Forward Start Employee Call Option Price (Monte Carlo):", option_price)

#%%
import numpy as np

# Parameters
S0 = 6.80            # Initial spot price
K = 6.90             # Strike price
r_domestic = 0.07    # Domestic risk-free interest rate
r_foreign = 0.09     # Foreign risk-free interest rate
sigma = 0.14         # Volatility of the spot rate
T = 0.5              # Time to expiration (6 months)
N = 10000              # Number of time steps
M = 10000            # Number of simulations

# Time increment
dt = T / N
discount_factor = np.exp(-r_domestic * T)

# Generate stock price paths
np.random.seed(42)
Z = np.random.normal(size=(M, N))
ST = np.zeros((M, N + 1))
ST[:, 0] = S0

for t in range(1, N + 1):
    ST[:, t] = ST[:, t - 1] * np.exp((r_domestic - r_foreign - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# Calculate arithmetic average of the path
average_price = np.mean(ST[:, 1:], axis=1)

# Calculate payoff at maturity
payoff = np.maximum(K - average_price, 0)

# Discount the payoff back to present value
option_price = np.mean(payoff) * discount_factor

print(f"第五題：The price of the Asian currency put option is: {option_price:.4f}")
#%%
import numpy as np

# 參數設置
S0 = 6.80            # 初始現貨價格
K = 6.90             # 行權價
r_domestic = 0.07    # 本國無風險利率
r_foreign = 0.09     # 外國無風險利率
sigma = 0.14         # 現貨價格的波動率
T = 0.5              # 到期時間（6個月）
N = 1              # 時間步數
M = 10000000            # 模擬次數

# 時間增量
dt = T / N
discount_factor = np.exp(-r_domestic * T)

# 生成股票價格路徑
np.random.seed(3)
Z = np.random.normal(size=(M, N))
ST = np.zeros((M, N + 1))
ST[:, 0] = S0

for t in range(1, N + 1):
    ST[:, t] = ST[:, t - 1] * np.exp((r_domestic - r_foreign - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# 計算路徑的算術平均價格
average_price = np.mean(ST[:, 1:], axis=1)

# 計算到期時的收益
payoff = np.maximum(K - average_price, 0)

# 收益折現回現值
discount_payoff = payoff* discount_factor

# 將
option_price = np.mean(discount_payoff) 

print(f"第五題：The price of the Asian currency put option is: {option_price:.4f}")
#%%
import numpy as np
from scipy.stats import norm

# Parameters
S0 = 50               # Initial stock price
K_call = 55           # Call option strike price
K_put = 48            # Put option strike price
T_call = 0.5          # Time to expiration for call (6 months)
T_put = 7/12          # Time to expiration for put (7 months)
T_choose = 3/12       # Time to choose (3 months)
Tc = 0.25
r = 0.10              # Risk-free interest rate
q = 0.05              # Dividend yield
sigma = 0.35          # Volatility
n_simulations = 100000

# Generate random numbers for stock price in 3 months
np.random.seed(8)
Z = np.random.normal(size=n_simulations)

# Simulate stock price in 3 months
stc = S0 * np.exp((r - q - 0.5 * sigma ** 2) * Tc + sigma * np.sqrt(Tc) * Z)



# Calculate the value of the call and put options at the time of choosing
def black_scholes_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

call_values = black_scholes_call(stc, K_call, T_call - T_choose, r, q, sigma)
put_values = black_scholes_put(stc, K_put, T_put - T_choose, r, q, sigma)

# The value of the chooser option is the maximum of the call and put values at the decision point
chooser_values = np.maximum(call_values, put_values)

# Discount the chooser values back to present
chooser_values_discounted = chooser_values * np.exp(-r * Tc)
# Discount the chooser values back to present
chooser_option_price = np.mean(chooser_values_discounted) 

print(f"第六題：The price of the chooser option is: {chooser_option_price:.2f}")

#%%
import numpy as np

# Parameters
S0 = 120            # Current stock price
S_min = 100         # Minimum stock price observed so far
T = 0.5             # Time to expiration (6 months)
r = 0.10            # Risk-free interest rate
q = 0.06            # Dividend yield
sigma = 0.30        # Volatility
N = 100000          # Number of simulations
M = 1000            # Number of time steps

# Time increment
dt = T / M
discount_factor = np.exp(-r * T)

# Generate stock price paths
np.random.seed(42)
Z = np.random.normal(size=(N, M))
ST = np.zeros((N, M + 1))
ST[:, 0] = S0
S_min_paths = np.full((N, M + 1), S_min)

for t in range(1, M + 1):
    ST[:, t] = ST[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    S_min_paths[:, t] = np.minimum(S_min_paths[:, t - 1], ST[:, t])

# Calculate payoff at maturity
payoff = np.maximum(ST[:, -1] - S_min_paths[:, -1], 0)

# discount 
discount_payoff  = payoff * discount_factor
# Discount the payoff back to present
option_price = np.mean(discount_payoff)

print(f"第七題：The price of the lookback call option is: {option_price:.2f}")

#%%
import numpy as np
from scipy.stats import norm

# Parameters
S0 = 500            # Current stock price
K_call = 520        # Strike price of the underlying call option
T_call = 6/12       # Time to maturity of the call option (6 months)
K_put = 50          # Strike price of the put-on-call option
T_put = 3/12        # Time to exercise the put-on-call option (3 months)
r = 0.08            # Risk-free interest rate
q = 0.03            # Dividend yield
sigma = 0.35        # Volatility
N = 100000           # Number of simulations
M = 100             # Number of time steps

# Time increment
dt = T_put / M
discount_factor_put = np.exp(-r * T_put)

# Generate stock price paths
np.random.seed(42)
Z = np.random.normal(size=(N, M))
ST = np.zeros((N, M + 1))
ST[:, 0] = S0

for t in range(1, M + 1):
    ST[:, t] = ST[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# Stock price at the time of choosing
S_choose = ST[:, -1]

# Black-Scholes formula for European call option
def black_scholes_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Value of the call option at T_put
call_values = black_scholes_call(S_choose, K_call, T_call - T_put, r, q, sigma)

# Calculate the payoff of the put-on-call option at T_put
payoff = np.maximum(K_put - call_values, 0)

discount_payoff = payoff * discount_factor_put

# Discount the payoff back to present value
put_on_call_price = np.mean(discount_payoff) 

print(f"第八題：The price of the put-on-call option is: {put_on_call_price:.2f}")
#%%
import numpy as np

# Given parameters
S0 = 100  # initial stock price
K = 100  # strike price
r = 0.1  # risk-free interest rate
sigma = 0.3  # volatility
T = 1  # time to expiry
L = 120  # barrier price
M = 10000  # number of time steps
I = 50  # number of paths
dt = T / M  # time step
discount_factor = np.exp(-r * T)  # discount factor

# Initialize stock price matrix
ST = np.zeros((I, M + 1))
ST[:, 0] = S0

# Simulate the paths and check barriers
np.random.seed(0)
Z = np.random.standard_normal((I, M + 1))
for t in range(1, M + 1):
    ST[:, t] = ST[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# Check if barriers are hit during the monitoring period
hit_barrier = np.any((ST[:, :M//2] >= L), axis=1)

# If the barrier is not hit, calculate the payoff of the lookback option
payoff = np.where(hit_barrier, 0, np.maximum(ST[:, -1] - K, 0))

# Discount the payoff to present value
option_price = np.mean(payoff) * discount_factor


print(f"第九題：The price of the look-barrier call option is: {option_price:.2f}")
#%%

#%%
import numpy as np

# Parameters
S0 = 100            # Current stock price
L = 80              # Lower barrier
U = 120             # Upper barrier
T = 3/12            # Time to expiration (3 months)
r = 0.05            # Risk-free interest rate
q = 0.02            # Dividend yield
sigma = 0.30        # Volatility
P = 10              # Payout if barriers are not hit
N = 10000           # Number of simulations
M = 100             # Number of time steps

# Time increment
dt = T / M
discount_factor = np.exp(-r * T)

# Generate stock price paths
np.random.seed(42)
Z = np.random.normal(size=(N, M))
ST = np.zeros((N, M + 1))
ST[:, 0] = S0

# Simulate the paths and check barriers
for t in range(1, M + 1):
    ST[:, t] = ST[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# Check if barriers are hit
hit_barrier = np.any((ST <= L) | (ST >= U), axis=1)

# Calculate the payoff
payoff = np.where(hit_barrier, 0, P)

# Discount the payoff to present value
option_price = np.mean(payoff) * discount_factor

print(f"第十題：The price of the double-barrier knock-out binary option is: {option_price:.2f}")


#%%

