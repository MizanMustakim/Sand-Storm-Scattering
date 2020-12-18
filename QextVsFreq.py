### particle refractive index

import numpy as np
from mie import pyMie
import matplotlib.pyplot as plt

a = pyMie()

f = np.array([500,1000,1500,3000])  ### GHz
m = np.sqrt(3+(18.256/f)*1j)
c = 3*10**8   ### speed of wind, m/s
lambda_func = (c * 10**(-3))/f   ### [um]
dia = np.arange(10,310,10)

Qe = []
for i in range(len(m)):
    x = (np.pi*dia)/lambda_func[i]
    Qe.append([])
    for j in range(len(dia)):
        Qe[i].append(a.mie(m[i], x[j]))

### Plotting the graph of Qext Vs Frequency

plt.figure()
plt.plot(dia, [[Qe[j][i][0] for j in range(len(m))] for i in range(len(dia))])
plt.legend(['0.5THz','1.0THz','1.5THz','3.0THz'])
plt.xlabel("D/\u03bcm")
plt.ylabel("Q_ext")
plt.title("Qext Vs Frequency")
plt.axis([0,300,-0.5,5])
plt.show()