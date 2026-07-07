import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(0,10,20)
speed = 5*time
fuel = 7 - 0.1*time

fig, ax = plt.subplots(1,2)

ax[0].plot(time, speed)
ax[0].set_title("Speed vs time")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Speed")

ax[1].plot(time, fuel)
ax[1].set_title("Fuel vs Time")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Fuel")

plt.show()
