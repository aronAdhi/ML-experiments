import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")
age = df['Age']
age = age.dropna().sort_values()
print(age.head())

plt.plot(age.values)
plt.show()