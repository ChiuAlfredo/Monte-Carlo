#%%

import numpy as np
from numpy.polynomial import polynomial as poly
import matplotlib.pyplot as plt

K = 1.1
r = 0.06

#%% Step 1: 模擬股價
St = np.array([[1., 1.09, 1.08, 1.34],
               [1., 1.16, 1.26, 1.54],
               [1., 1.22, 1.07, 1.03],
               [1., 0.93, 0.97, 0.92],
               [1., 1.11, 1.56, 1.52],
               [1., 0.76, 0.77, 0.90],
               [1., 0.92, 0.84, 1.01],
               [1., 0.88, 1.22, 1.34]])

#%% Step 2: 計算最後履約點 (t=3) 之 payoff
CF = np.zeros([8, 3])
execTable = np.zeros([8, 3])
St3 = St[:, 3]
CF[:, 2] = (K - St3) * (K > St3)
execTable[:, 2] = K > St3

#%% Step 3: 找出 t=2 in-the-money 之股價
St2 = St[:, 2]
St2In = St2[St2 < K]

CF3 = CF[:, 2]
X = St2In
Y = CF3[St2 < K] * np.exp(-r)

coef = poly.polyfit(X, Y, 2)

#%% 

plt.scatter(X, Y)

x = np.linspace(0.7, 1.1, 201)
xx = np.reshape(x, (-1, 1))
y = np.sum(coef * xx ** np.array([0, 1, 2]), axis=1)

plt.plot(x, y)

#%% Step 4: 計算 t=2 下 execrise 及 holding value 
exec2 = K - St2In
hold2 = np.sum(coef * np.reshape(St2In, (-1, 1)) ** np.array([0, 1, 2]), axis=1)

#%% Step 5: 將 t=2 下 execrise value 大於 holding value者記入 cash flow 表
CF[St2 < K, 1] = exec2 * (exec2 > hold2)
execTable[St2 < K, 1] = exec2 > hold2

# %% Step 6: 重複 step 3 至 step 5
St1 = St[:, 1]
St1In = St1[St1 < K]

CF2 = CF[:, 1]
X = St1In
Y = CF2[St1 < K] * np.exp(-r)

coef = poly.polyfit(X, Y, 2)

# %%
exec1 = K - St1In
hold1 = np.sum(coef * np.reshape(St1In, (-1, 1)) ** np.array([0, 1, 2]), axis=1)
CF[St1 < K, 0] = exec1 * (exec1 > hold1)
execTable[St1 < K, 0] = exec1 > hold1

# %% Step 7: 利用 execrise table 得知每條 path 之最佳履約點 (最早可被 exercise 時間點) 將 payoff 折現計算價格
execTable = np.cumsum(execTable, axis=1)
execTable = np.cumsum(execTable, axis=1)
execTable[execTable > 1] = 0

CF = CF * execTable

tau = np.array([1, 2, 3])
price = np.mean(np.sum(CF * np.exp(-r * tau), axis=1))
