import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 24, 1) # 24 hours

temp = 15 + 10 * np.sin(time / 4)
humidity = 80 - 20 * np.sin(time / 4)
wind_speed = 5 + np.random.rand(24) * 15
rain_chance = np.linspace(0, 100, 24)

fig, ((tmp, hum), (win, rain)) = plt.subplots(2,2)

tmp.plot(time, temp)
tmp.set_title("Temperature")

hum.plot(time, humidity)
hum.set_title("Humidity")

win.plot(time, wind_speed)
win.set_title("Wind speed")

rain.plot(time, rain_chance)
rain.set_title("Rain chance")

plt.tight_layout()
plt.show()
