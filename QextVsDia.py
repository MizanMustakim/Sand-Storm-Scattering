### particle refractive index

import numpy as np
from mie import pyMie
import matplotlib.pyplot as plt

a = pyMie()

f = 1000  # GHz
m = np.sqrt(3+(18.256/f)*1j)    ### f >= 80 GHz
npar = m.real   ### particle refractive index
nmed = 1  ### medium refractive index (air)
fv = 0.001   ### volume fraction of spheres in medium, 0.1%
c = 3*10**8   ### speed of wind, m/s
lambda_func = (c * 10**(-3))/f   ### [um]
dia = np.arange(10,310,10)
x = (np.pi * dia)/lambda_func

Qe = []
for j in range(len(dia)):
    Qe.append(a.mie(m,x[j]))

### Plotting the graph of Qext vs dia
plt.plot(dia, [[Qe[i][j] for j in range(3)] for i in range(len(Qe))])

f1 = f/1000
m1 = m.real
m2 = m.imag
plt.title("Mie Efficiencies, f = {}THz, m = {:.4f}+{:.4f}i".format(int(f1),m1,m2))
plt.xlabel('D/\u03bcm')
plt.axis([0,300,-0.5,5])
plt.show()