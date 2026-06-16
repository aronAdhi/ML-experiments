Add below patch to matplotlib library to avoid boiler plate code for 3d plotting



def plot\_3d(X, Y, Z, \*args, \*\*kwargs):

&#x20;   fig = figure()

&#x20;   ax = fig.add\_subplot(111, projection='3d')



&#x20;   ax.plot\_surface(X, Y, Z, \*args, \*\*kwargs)



&#x20;   return ax

in C:\\Users\\adhir\\AppData\\Local\\Python\\pythoncore-3.14-64\\Lib\\site-packages\\matplotlib\\pyplot.py



usage



import numpy as np



x = np.arange(-5,6,1)

y = np.arange(-3,4,1)



X,Y = np. meshgrid(x,y)



Z = X + Y



import matplotlib.pyplot as plt



x\_line = np.linspace(-5, 5, 100)

y\_line = np.zeros\_like(x\_line)

z\_line = x\_line\*\*2 + y\_line\*\*2



ax = plt.plot\_3d(X, Y, Z)

ax.plot(x\_line, y\_line, z\_line)

plt.show()

