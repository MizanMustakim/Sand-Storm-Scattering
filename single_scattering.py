import numpy as np
from mie import pyMie
import matplotlib.pyplot as plt

a = pyMie()

V = np.arange(0.05,1.05,0.05) # visibility/km
step = 0.0001
r = np.arange(.0001,0.5001, 0.0001) # mm
m0 = -2.31
sigma = 0.296

# 调用mie.m计算Qext(r)
f = 5000 #GHz
m = np.sqrt(3+(18.256/f)*1j) # f>=80 GHz
c = 3*10**8   # speed of light
lambda_func = (c*.001)/f   # um
dia = r*2000  # um
x = (np.pi*dia)/lambda_func
Qe = []
for j in range(len(dia)):
    Qe.append(a.mie(m,x[j]))

numerator = 0
denominator = 0
for k in range(len(r)):
    p = (np.exp(-(np.log(2*(r[k]))-m0)**2 / 2*sigma**2))/(2*(r[k])*sigma*np.sqrt(2*np.pi))
    numerator += np.pi*np.square(r[k])* (Qe[k][0])* p*step
    denominator += r[k]**2*p*step 

A = []      ### Attenuation Rate
for i in range(len(V)):
    A.append(15 * numerator / (2*np.pi* V[i]*denominator))
print(A)

### plotting the graph A vs V
for i in range(len(A)):
    plt.scatter(V[i],A[i])
    plt.axis([0,1,0,400])
    plt.xlabel("Visibility/km")
    plt.ylabel("A (Attenuation rate)")
    plt.title("Single Scattering of Sand Storm")
    plt.pause(0.5)
plt.show()