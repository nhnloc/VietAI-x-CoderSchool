import pandas as pd
import matplotlib.pyplot as plt 


data = pd.read_csv("web-traffic.csv")

data["date"] = pd.to_datetime(data["date"], yearfirst=True)
fig, ax = plt.subplots()

ax.plot(data["date"], data["users"])
ax.set_xlabel("Time")
ax.set_ylabel("Users")
plt.show()