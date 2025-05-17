import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# 자귀회귀좌정은 변수가 자기 자신에게 회귀하는 프로세스
# 시계열에서 이는 현잿값이 과거값에 선형적으로 의존한다는 것

df = pd.read_csv("../data/foot_traffic.csv")
df.head()

fig, ax = plt.subplots()
ax.plot(df["foot_traffic"])
ax.set_xlabel("Time")
ax.set_ylabel("Average weekly foot traffic")

plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
fig.autofmt_xdate()
plt.tight_layout()

ADF_result = adfuller(df["foot_traffic"])

print(f"ADF Statistic: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")

# 변환을 적용하여 시계열을 정상화. 장기 추세 제거
foot_traffic_diff = np.diff(df["foot_traffic"], n=1)

# 변환된 시계열의 도식화
fig, ax = plt.subplots()
ax.plot(foot_traffic_diff)
ax.set_xlabel("Time")
ax.set_ylabel("Average weekly foot traffic")

plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
fig.autofmt_xdate()
plt.tight_layout()

# 변환된 수열에 대한 ADF 테스트
ADF_result = adfuller(foot_traffic_diff)
print(f"ADF Statistic: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")

# ACF의 도식화
plot_acf(foot_traffic_diff, lags=20)
plt.tight_layout()

# 편자기상관함수 (partial autocorrlation function, PACF) 확인
plot_pacf(foot_traffic_diff, lags=20)
plt.tight_layout()

# AR(3) 모형을 이용한 예측
df_diff = pd.DataFrame({"foot_traffic_diff": foot_traffic_diff})

train = df_diff[:-52]
test = df_diff[-52:]

print(len(train))
print(len(test))

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))

ax1.plot(df["foot_traffic"])
ax1.set_xlabel("Time")
ax2.set_ylabel("Avg. weekly foot traffic")
ax1.axvspan(948, 1000, color="#808080", alpha=0.2)

ax2.plot(df_diff["foot_traffic_diff"])
ax2.set_xlabel("Time")
ax2.set_ylabel("Avg. weekly foot traffic")
ax2.axvspan(947, 999, color="#808080", alpha=0.2)

plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))

fig.autofmt_xdate()
fig.tight_layout()


def rolling_forecast(
    df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str
) -> list:
    total_len = train_len + horizon
    end_idx = train_len

    if method == "mean":
        pred_mean = []

        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))

        return pred_mean

    elif method == "last":
        pred_last_value = []

        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))

        return pred_last_value

    elif method == "AR":
        pred_AR = []

        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(3, 0, 0))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window]
            pred_AR.extend(oos_pred for _ in range(window))

        return pred_AR


TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, "mean")
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_AR = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, "AR")

test["pred_mean"] = pred_mean
test["pred_last_value"] = pred_last_value
test["pred_AR"] = pred_AR

fig, ax = plt.subplots()

ax.plot(df_diff["foot_traffic_diff"])
ax.plot(test["foot_traffic_diff"], "b-", label="actual")
ax.plot(test["pred_mean"], "g:", label="mean")
ax.plot(test["pred_last_value"], "r-", label="last")
ax.plot(test["pred_AR"], "k--", label="AR(3)")
ax.legend(loc=2)

ax.set_xlabel("Time")
ax.set_ylabel("Diff. avg. weekly foot traffic")

ax.axvspan(947, 998, color="#808080", alpha=0.2)
ax.set_xlim(920, 999)

plt.xticks([936, 988], [2018, 2019])
fig.autofmt_xdate()
fig.tight_layout()

mse_mean = mean_squared_error(test["foot_traffic_diff"], test["pred_mean"])
mse_laat = mean_squared_error(test["foot_traffic_diff"], test["pred_last_value"])
mse_AR = mean_squared_error(test["foot_traffic_diff"], test["pred_AR"])

print(mse_mean, mse_laat, mse_AR)

# 예측값을 원래의 데이터 규모로 되돌리기
df["pred_foot_traffic"] = pd.Series()
df["pred_foot_traffic"][948:] = df["foot_traffic"].iloc[948] + test["pred_AR"].cumsum()

fig, ax = plt.subplots()

ax.plot(df["foot_traffic"], "b-", label="actual")
ax.plot(df["pred_foot_traffic"], "k--", label="AR(3)")
ax.legend(loc=2)

ax.set_xlabel("Time")
ax.set_ylabel("Diff. avg. weekly foot traffic")

ax.axvspan(948, 1000, color="#808080", alpha=0.2)
ax.set_xlim(920, 1000)
ax.set_ylim(650, 770)

plt.xticks([936, 988], [2018, 2019])
fig.autofmt_xdate()
fig.tight_layout()

# 원본 데이터셍의 평균절대오차 계산
mean_AR_undiff = mean_absolute_error(
    df["foot_traffic"][948:], df["pred_foot_traffic"][948:]
)
print(mean_AR_undiff)
