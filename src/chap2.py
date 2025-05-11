import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/jj.csv")
df.head()
df.tail()

train = df[:-4]
test = df[-4:]

historical_mean = np.mean(train['data'])
print(historical_mean)

test.loc[:, 'pred_mean'] = historical_mean

# MAPE 함수 정의
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/ y_true)) * 100

mape_hist_mean = mape(test['data'], test['pred_mean'])
print(mape_hist_mean)

# 훈련 데이터, 예측 기간, 테스트 집합상의 관측값, 1980년 각 분기에 대한 예측값을 표시
fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label = 'Train')
ax.plot(test['date'], test['data'], 'b-', label = 'Test')
ax.plot(test['date'], test['pred_mean'], 'r--', label = 'Predicted')
ax.set_xlabel('Data')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color = '#808080', alpha = 0.2)
ax.legend(loc = 2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])
fig.autofmt_xdate()
fig.tight_layout()

# 작년의 평균으로 예측하기
last_year_mean = np.mean(train.data[-4:])
print(last_year_mean)

test.loc[:, 'pred__last_yr_mean'] = last_year_mean
mape_last_yr_mean = mape(test['data'], test['pred__last_yr_mean'])
print(mape_last_yr_mean)

fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label = 'Train')
ax.plot(test['date'], test['data'], 'b-', label = 'Test')
ax.plot(test['date'], test['pred__last_yr_mean'], 'r--', label = 'Predicted')
ax.set_xlabel('Data')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color = '#808080', alpha = 0.2)
ax.legend(loc = 2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])
fig.autofmt_xdate()
fig.tight_layout()

# 마지막으로 측정된 값으로 예측하기
last = train.data.iloc[-1]
print(last)

test.loc[:, 'pred_last'] = last

mape_last = mape(test['data'], test['pred_last'])
print(mape_last)

fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label = 'Train')
ax.plot(test['date'], test['data'], 'b-', label = 'Test')
ax.plot(test['date'], test['pred_last'], 'r--', label = 'Predicted')
ax.set_xlabel('Data')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color = '#808080', alpha = 0.2)
ax.legend(loc = 2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])
fig.autofmt_xdate()
fig.tight_layout()

# 2.5 단순한 계절적 예측 구현하기
test.loc[:, 'pred_last_season'] = train['data'][-4:].values
mape_naive_seasonal = mape(test['data'], test['pred_last_season'])
print(mape_naive_seasonal)

fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label = 'Train')
ax.plot(test['date'], test['data'], 'b-', label = 'Test')
ax.plot(test['date'], test['pred_last_season'], 'r--', label = 'Predicted')
ax.set_xlabel('Data')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color = '#808080', alpha = 0.2)
ax.legend(loc = 2)

plt.xticks(np.arange(0, 85, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])
fig.autofmt_xdate()
fig.tight_layout()

 