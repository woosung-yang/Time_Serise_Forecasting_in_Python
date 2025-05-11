# 3.1 확률보행 프로세스
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

steps = np.random.standard_normal(1000)
steps[0] = 0

random_walk = np.cumsum(steps)

fig, ax = plt.subplots()

ax.plot(random_walk)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')

plt.tight_layout()

## 정상적 시계열(stationary time series): 
# 평균과 분산이 상수이고 자기상관관계가 있으며, 이러한 특성들이 시간에 따라 변하지 않는 시계열

## 정상성 검증
# ADF(augmented Dickey-FUller) test: 시계열에 단위근(unit root)가 존재한다는 귀무가설을 검정
# 단위근이 있는 시계열은 다음의 특징을 가진다:
# - 시계열의 평균과 분산이 시간에 따라 변함
# - 즉, **시계열이 정상성(stationarity)**을 만족하지 않음
# - 충격이 가해졌을 때 그 효과가 영구적으로 지속됨
# - 보통 랜덤 워크(Random Walk) 형태를 보임

## 자기상관함수
# 시계열과 시계열 그 자체 사이의 상관관계를 측정

from statsmodels.tsa.stattools import adfuller

ADF_result = adfuller(random_walk)
print(f'ADF Statstic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(random_walk, lags = 20)

diff_random_walk = np.diff(random_walk, n = 1)

fig, ax = plt.subplots()
ax.plot(diff_random_walk)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()

ADF_result = adfuller(diff_random_walk)
print(f'ADF Statstic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

plot_acf(diff_random_walk, lags = 20)

# 3.2.5 GOOGL은 확률보행인가?
df = pd.read_csv('../data/GOOGL.csv')

fig, ax = plt.subplots()
ax.plot(df['Date'], df['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Closing price (USD)')
plt.xticks(
    [4, 24, 46, 68, 89, 110, 132, 152, 174, 193, 212, 235],
    ['May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2021', 'Feb', 'Mar', 'April']
)
fig.autofmt_xdate()
fig.tight_layout()

ADF_result = adfuller(df['Close'])
print(f'ADF Statstic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

diff_close = np.diff(df['Close'], n = 1)
ADF_result = adfuller(diff_close)
print(f'ADF Statstic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

plot_acf(diff_close, lags = 20)

# 3.3 확률보행 예측하기

df = pd.DataFrame({'value': random_walk})
train = df[:800]
test = df[800:]

fig, ax = plt.subplots()
ax.plot(random_walk)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
ax.axvspan(800, 1000, color = '#808080', alpha = 0.2)
plt.tight_layout()

mean = np.mean(train.values)
test.loc[:, 'pred_mean'] = mean
test.head()

last_value = train.iloc[-1].value
test.loc[:, 'pred_last'] = last_value
test.head()

deltaX = 800 - 1
deltaY = last_value - 0
drift = deltaY / deltaX
print(drift)

x_vals = np.arange(801, 1001, 1)
pred_drift = drift * x_vals
test.loc[:, 'pred_drift'] = pred_drift

fig, ax = plt.subplots()
ax.plot(train.values, 'b-')
ax.plot(test['value'], 'b-')
ax.plot(test['pred_mean'], 'r-', label = 'Mean')
ax.plot(test['pred_last'], 'g--', label = 'Last Value')
ax.plot(test['pred_drift'], 'k:', label = 'Drift')
ax.axvspan(800, 1000, color = '#808080', alpha = 0.2)
ax.legend(loc = 2)

ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')

plt.tight_layout()

from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(test['value'], test['pred_mean'])
mse_last = mean_squared_error(test['value'], test['pred_last'])
mse_drift = mean_squared_error(test['value'], test['pred_drift'])

print(mse_mean, mse_last, mse_drift)

# 그림 3.16
fig, ax = plt.subplots()

x = ['mean', 'last_value', 'drift']
y = [mse_mean, mse_last, mse_drift]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Methods')
ax.set_ylabel('MSE')
ax.set_ylim(0, 500)

for index, value in enumerate(y):
    plt.text(x=index, y=value+5, s=str(round(value, 2)), ha='center')

plt.tight_layout()

# 3.3.2 다음 시간 단계 예측하기
df_shift = df.shift(periods = 1)
df_shift.head()

# 그림 3.18
fig, ax = plt.subplots()
ax.plot(df, 'b-', label = 'actual')
ax.plot(df_shift, 'r-.', label = 'forecast')
ax.legend(loc = 2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()

mse_one_step = mean_squared_error(test['value'], df_shift[800:])
print(mse_one_step)

# 그림 3.19
fig, ax = plt.subplots()

ax.plot(df, 'b-', label='actual')
ax.plot(df_shift, 'r-.', label='forecast')

ax.legend(loc=2)

ax.set_xlim(900, 1000)
ax.set_ylim(15, 28)

ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')

plt.tight_layout()

