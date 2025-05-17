import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
