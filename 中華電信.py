# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import scipy.stats as stats

# Parameters
n_simulations = 100000  # Number of Monte Carlo simulations
n_years = 10  # Number of years
n_weeks = n_years * 52  # Total number of weeks
n_days_per_week = 5  # Number of trading days per week
n_days = n_weeks * n_days_per_week+1  # Total number of trading days

initial_exchange_rate = 33.06
annual_interest_rate_ntd = 0.0254
annual_interest_rate_usd = 0.0401
volatility = 0.045665
unit = 2000000
margin_deposit = 1.5*unit
strike_price = 31.50
delivery_price = 30.00
termination_price = 32.70
n_periods = int(n_weeks/2)


# 每日漂移和波动率
# dt = 1 / 260  # 每个时间步的时间长度，以年为单位（每日）
dt = 1/260
# daily_volatility = volatility / np.sqrt(260)
# mu = r - q - 0.5 * daily_volatility ** 2
# mu = annual_interest_rate_ntd-annual_interest_rate_usd-0.5 * volatility ** 2

mu = annual_interest_rate_ntd - annual_interest_rate_usd -0.5 * volatility ** 2

# 进行蒙特卡罗模拟

rng = np.random.default_rng()  # 使用新的随机数生成器
Z = rng.standard_normal((n_simulations, n_days))
log_return = ((mu) * dt + volatility * np.sqrt(dt) * Z)
log_st = np.log(initial_exchange_rate) + np.cumsum(log_return, axis=1)
st = np.exp(log_st)

# 添加初始价格到模拟路径中
exchange_rate_simulations = np.hstack((np.full((n_simulations, 1), initial_exchange_rate), st))

# 绘制模拟结果
plt.figure(figsize=(10, 6))
for i in range(1000):
    plt.plot(st[i, :], lw=0.5, alpha=0.3)
plt.title("exchange rate GBM")
plt.xlabel("day")
plt.ylabel("rate")
plt.grid(True)
plt.show()


simulation_exchange_rate_every_deal = exchange_rate_simulations[:, ::10][:,1::]

npv_tw =  np.zeros_like(simulation_exchange_rate_every_deal)

termination_count = 0
profit = 0
loss = 0

for i in range(n_simulations):
    for j in range(n_periods):
        # i=0
        # j=248
        # Check if contract terminates\
        in_period = j+1

        # 還沒到期
        if j != n_periods-1:

            # 提前結束
            if j >= 6 and simulation_exchange_rate_every_deal[i,j] >= 32.7:

                termination_count+=1
                # 先計算美元保證金複利並折現，最後換算回台幣
                npv_tw[i, j] += (
                    (
                        margin_deposit * (1 + 0.08) ** ((in_period) * 2 * 5 / 260)
                        - margin_deposit* (1 + annual_interest_rate_usd) ** ((in_period) * 2 * 5 / 260)
                    )
                    * (np.exp((-annual_interest_rate_usd) * (in_period * 2 / 52)))
                    * initial_exchange_rate
                )
                break

            elif simulation_exchange_rate_every_deal[i,j] >= 31.50:
                profit +=1
                npv_tw[i,j] += (simulation_exchange_rate_every_deal[i,j]-30) *unit *(np.exp((-annual_interest_rate_ntd)* (in_period*2/52)))

            elif simulation_exchange_rate_every_deal[i,j] < 31.50:
                loss+=1
                npv_tw[i,j] = (simulation_exchange_rate_every_deal[i,j]-31.5) *2*unit *(np.exp((-annual_interest_rate_ntd)* (in_period*2/52)))
                # print(f'第{i}筆交易在第{j}期')

        # 已經到期
        elif j == n_periods-1:
            if j >= 6 and simulation_exchange_rate_every_deal[i,j] >= 32.7:

                termination_count+=1
                #

            elif simulation_exchange_rate_every_deal[i,j] >= 31.50:
                profit +=1
                npv_tw[i,j] += (simulation_exchange_rate_every_deal[i,j]-30) *unit *(np.exp((-annual_interest_rate_ntd)* (in_period*2/52)))

            elif simulation_exchange_rate_every_deal[i,j] < 31.50:
                loss+=1
                npv_tw[i,j] = (simulation_exchange_rate_every_deal[i,j]-31.5) *2*unit *(np.exp((-annual_interest_rate_ntd)* (in_period*2/52)))
                # print(f'第{i}筆交易在第{j}期')

            # 先計算美元保證金複利並折現，最後換算回台幣
            npv_tw[i, j] += (
                    (
                        margin_deposit * (1 + 0.08) ** ((in_period) * 2 * 5 / 260)
                        - margin_deposit* (1 + annual_interest_rate_usd) ** ((in_period) * 2 * 5 / 260)
                    )
                    * (np.exp((-annual_interest_rate_usd) * (in_period * 2 / 52)))
                    * initial_exchange_rate
                )
npv_simu = np.sum(npv_tw, axis=1)
npv_mean = np.mean(npv_simu)
npv_std = np.std(npv_tw)
confidence_interval = stats.norm.interval(0.95, loc=npv_mean, scale=npv_std / np.sqrt(npv_simu))

SE = sci.sem(npv_simu)
CI = sci.norm.interval(0.95, loc=npv_mean, scale=SE)

# Calculate Value at Risk (VaR) and Expected Shortfall (ES)
var_90 = np.percentile(npv_simu, 10)
var_95 = np.percentile(npv_simu, 5)
var_99 = np.percentile(npv_simu, 1)
es_99 = np.mean(npv_simu[npv_simu < var_99])
es_95 = np.mean(npv_simu[npv_simu < var_95])
es_90 = np.mean(npv_simu[npv_simu < var_90])

# Calculate probabilities
# loss_probability = np.sum(npv_tw<0) / (np.sum(npv_tw<0)+np.sum(npv_tw>0))
loss_probability = np.sum(npv_simu <0)/n_simulations
early_termination_probability = termination_count / n_simulations

# Calculate average contract lifespan
first_zero_positions = np.argmax(npv_tw == 0, axis=1)

# 如果行中没有 0，则 `np.argmax` 会返回 0，因此我们需要处理这种情况
no_zero_mask = np.all(npv_tw != 0, axis=1)
first_zero_positions[no_zero_mask] = 260

# 将索引值全部减 1，对于 260 的值不做减法
first_zero_positions = np.where(first_zero_positions == 260, 260, first_zero_positions - 1)

average_lifespan = np.mean(first_zero_positions)

# Print results
print("Contract Value:")
print(f"Mean: {npv_mean}")
print(f"95% Confidence Interval: {CI}")
print("\nValue at Risk (VaR) and Expected Shortfall (ES):")
print(f"90% VaR: {var_90}")
print(f"95% VaR: {var_95}")
print(f"99% VaR: {var_99}")
print(f"90% ES: {es_90}")
print(f"95% ES: {es_95}")
print(f"99% ES: {es_99}")
print("\nProbabilities:")
print(f"Loss Probability: {loss_probability}")
print(f"Early Termination Probability: {early_termination_probability}")
print("\nAverage Contract Lifespan:")
print(f"{average_lifespan} 期")
# %%
