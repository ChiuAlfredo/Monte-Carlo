#%%

import numpy as np
import scipy.stats as sci

S = 100
X = 3
T = 0.5
r = 0.08
q = 0.04

K = np.tile(np.reshape(np.array([90, 100, 110]), (-1, 1)), (6, 1))
H = np.tile(np.reshape(np.repeat(np.array([95, 100, 105]), 3), (-1, 1)), (2, 1))                                                           
sigma = np.tile(np.array([0.25, 0.3]), (1, 2))

isCall = np.repeat(np.array([True, False]), 2)
isUp = np.tile(np.reshape(np.repeat(np.array([False, True]), (6, 3)), (-1, 1)), (2, 1))
isIn = np.reshape(np.repeat(np.array([False, True]), (9, 9)), (-1, 1))

# K = np.tile(np.reshape(np.array([90, 100, 110]), (-1, 1)), (6, 4))
# H = np.tile(np.reshape(np.repeat(np.array([95, 100, 105]), 3), (-1, 1)), (2, 4))
# sigma = np.tile(np.array([0.25, 0.3]), (18, 2))

# isCall = np.tile(np.repeat(np.array([True, False]), 2), (18, 1))
# isUp = np.tile(np.reshape(np.repeat(np.array([False, True]), (6, 3)), (-1, 1)), (2, 4))
#isIn = np.tile(np.reshape(np.repeat(np.array([False, True]), (9, 9)), (-1, 1)), (1, 4))

eta = 2 * (~isUp) - 1
zeta = 2 * isCall - 1

mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
lam = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)
v = sigma * np.sqrt(T)
mv = (1 + mu) * v

x1 = np.log(S / K) / v + mv
x2 = np.log(S / H) / v + mv
y1 = np.log(H ** 2 / S / K) / v + mv
y2 = np.log(H / S) / v + mv
z = np.log(H / S) / v + lam * v

zS = zeta * S * np.exp(-q * T)
zK = zeta * K * np.exp(-r * T)
HS2 = (H / S) ** (2 * mu)
HS21 = (H / S) ** (2 * (mu + 1))

A = zS * sci.norm.cdf(zeta * x1) - zK * sci.norm.cdf(zeta * (x1 - v))
B = zS * sci.norm.cdf(zeta * x2) - zK * sci.norm.cdf(zeta * (x2 - v))
C = zS * HS21 * sci.norm.cdf(eta * y1) - zK * HS2 * sci.norm.cdf(eta * (y1 - v))
D = zS * HS21 * sci.norm.cdf(eta * y2) - zK * HS2 * sci.norm.cdf(eta * (y2 - v))
E = X * np.exp(-r * T) * (sci.norm.cdf(eta * (x2 - v)) - HS2 * sci.norm.cdf(eta * (y2 - v)))
F = X * ((H / S) ** (mu + lam) * sci.norm.cdf(eta * z) + (H / S) ** (mu - lam) * sci.norm.cdf(eta * (z - 2 * lam * v)))

CE = C + E
ABDE = A - B + D + E
AE = A + E + np.zeros(np.shape(C))
BCDE = B - C + D + E
ACF = A - C + F
BDF = B - D + F
ABCDF = A - B + C - D + F
FF = F + np.zeros(np.shape(C))

price = np.zeros(np.shape(C))
flag = (isCall & ~isUp & isIn & (K >= H)) | (~isCall & isUp & isIn & (K < H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = CE[flag]
flag = (isCall & ~isUp & isIn & (K < H)) | (~isCall & isUp & isIn & (K >= H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = ABDE[flag]
flag = (isCall & isUp & isIn & (K >= H)) | (~isCall & ~isUp & isIn & (K < H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = AE[flag]
flag = (isCall & isUp & isIn & (K < H)) | (~isCall & ~isUp & isIn & (K >= H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = BCDE[flag]
flag = (isCall & ~isUp & ~isIn & (K >= H)) | (~isCall & isUp & ~isIn & (K < H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = ACF[flag]
flag = (isCall & ~isUp & ~isIn & (K < H)) | (~isCall & isUp & ~isIn & (K >= H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = BDF[flag]
flag = (isCall & isUp & ~isIn & (K >= H)) | (~isCall & ~isUp & ~isIn & (K < H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = FF[flag]
flag = (isCall & isUp & ~isIn & (K < H)) | (~isCall & ~isUp & ~isIn & (K >= H)) & np.tile(np.array([True]), np.shape(C))
price[flag] = ABCDF[flag]

#%%






