import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv('../data/widget_sales.csv')
df.head()

fig, ax = plt.subplots()
ax.plot(df['widget_sales'])
ax.set_xlabel('Time')
ax.set_ylabel('Wdiget sales (k$)')

plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
)
fig.autofmt_xdate()
plt.tight_layout()

# 4.1 시계열 데이터 분석 기본
# 4.1.1 시계열 데이터 시각화

ADF_result = adfuller(df['widget_sales'])
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

widget_sales_diff = np.diff(df['widget_sales'], n = 1)

fig, ax = plt.subplots()
ax.plot(widget_sales_diff)
ax.set_xlabel('Time')
ax.set_ylabel('Wdiget sales (k$)')

plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
)
fig.autofmt_xdate()
plt.tight_layout()

# 차분된 시계열에 대해 ADF 테스트 -> 시계열이 정상화됨
ADF_result = adfuller(widget_sales_diff)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

plot_acf(widget_sales_diff, lags = 30)
plt.tight_layout()

# 4.2 이동평균과정 예측하기
df_diff = pd.DataFrame({'widget_sales_diff': widget_sales_diff})
train = df_diff[:int(0.9 * len(df_diff))]
test = df_diff[int(0.9 * len(df_diff)):]
print(len(train))
print(len(test))

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
ax1.plot(df['widget_sales'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Widget Sales (k$)')
ax1.axvspan(450, 500, color = '#808080', alpha = 0.2)

ax2.plot(df_diff['widget_sales_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Widget Sales - diff (k$)')
ax2.axvspan(449, 498, color = "#808080", alpha = 0.2)
plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
)
fig.autofmt_xdate()
plt.tight_layout()

# MA(q) 모델을 사용할 때 모델은 q 단계 이후의 미래에 대한 예측으로 단순히 평균을 반환하는데, 이는 모델을 표현한 수식에 q단계 이후에 대한 오차항이 없기 때문이다. 단순히 수열을 평균을 사용하지 않기 위해서는 롤링 예측(rolling prediction)을 사용하여 한 번에 q 단계까지 예측할 수 있다

from statsmodels.tsa.statespace.sarimax import SARIMAX

def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str) -> list:
    
    total_len = train_len + horizon
    
    if method == 'mean':
        pred_mean = []
        
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
            
        return pred_mean
    
    elif method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value
    
    elif method == 'MA':
        pred_MA = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order = (0,0,2))
            res = model.fit(disp = False)
            prediction = res.get_prediction(0, i + window -1)
            oos_pred = prediction.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
            
        return pred_MA

pred_df = test.copy()

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_MA = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'MA')

pred_df['pred_mean'] = pred_mean
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_MA'] = pred_MA

from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_mean'])
mse_last = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_last_value'])
mse_MA = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_MA'])

print(mse_mean, mse_last, mse_MA)

df['pred_widget_sales'] = pd.Series()
df['pred_widget_sales'][450:] = df['widget_sales'].iloc[450] + pred_df['pred_MA'].cumsum()

fig, ax = plt.subplots()

ax.plot(df['widget_sales'], 'b-', label = 'actual')
ax.plot(df['pred_widget_sales'], 'k--', label = 'MA(2)')

ax.legend(loc = 2)

ax.set_xlabel('Time')

ax.axvspan(450, 500, color = "#808080", alpha = 0.2)
ax.set_xlim(400, 500)
plt.xticks(
    [409, 439, 468, 498],
    ['Mar', 'Apr', 'May', 'Jun']
)
fig.autofmt_xdate()
plt.tight_layout()

from sklearn.metrics import mean_absolute_error

mae_MA_undiff = mean_absolute_error(df['widget_sales'].iloc[450:], df['pred_widget_sales'].iloc[450:])
print(mae_MA_undiff)

## 4.4 연습문제
# 4.4.1

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

ma2 = np.array([1, 0.9, 0.3])
ar2 = np.array([1, 0, 0])

MA2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# 2.시뮬레이션한 도식을 시각화하자
fig, ax = plt.subplots()
ax.plot(MA2_process)
ax.set_xlabel('Time')
ax.set_ylabel('Value')

# 3. ADF 테스트를 실행하고 프로세스가 정상적인지 확인하자
ADF_result = adfuller(MA2_process)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# 4. ADF를 도식화하고 지연 2 이후에 유의한 계수가 있는지 확인하자
plot_acf(MA2_process, lags = 30)
plt.tight_layout()

# 5. 시뮬레이션된 수열을 훈련 집합과 테스트 집합으로 분할 
MA2_df = pd.DataFrame({'MA2_process': MA2_process})

train = MA2_df[:800]
test = MA2_df[800:]


from statsmodels.tsa.statespace.sarimax import SARIMAX


def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str) -> list:
    
    total_len = train_len + horizon
    
    if method == 'mean':
        pred_mean = []
        
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
            
        return pred_mean
    
    elif method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value
    
    elif method == 'MA':
        pred_MA = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order = (0,0,2))
            res = model.fit(disp = False)
            prediction = res.get_prediction(0, i + window -1)
            oos_pred = prediction.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
            
        return pred_MA

pred_df = test.copy()

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_mean = rolling_forecast(MA2_df, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value = rolling_forecast(MA2_df, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_MA = rolling_forecast(MA2_df, TRAIN_LEN, HORIZON, WINDOW, 'MA')

pred_df['pred_mean'] = pred_mean
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_MA'] = pred_MA

from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(pred_df['MA2_process'], pred_df['pred_mean'])
mse_last = mean_squared_error(pred_df['MA2_process'], pred_df['pred_last_value'])
mse_MA = mean_squared_error(pred_df['MA2_process'], pred_df['pred_MA'])

print(mse_mean, mse_last, mse_MA)

fig, ax = plt.subplots()
ax.plot(pred_df['MA2_process'], 'b-', label = 'MA2 Process')
ax.plot(pred_df['pred_MA'], 'r--', label = 'pred_MA')
ax.legend(loc = 2)
plt.tight_layout()




