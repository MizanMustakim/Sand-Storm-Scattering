import numpy as np
from mie import pyMie
import random
import matplotlib.pyplot as plt

a = pyMie()

rand = random.random()
N = 1*10**6
V = np.arange(0.05,1.05,0.05) # visibility/km
step = 1*10**(-4)
r = np.arange(step, 0.5+step, step)  # mm
m0 = -2.31 
sigma = 0.296
f = 500  ### GHz
m = np.sqrt(3+(18.256/f)*1j) ### f>=80 GHz
c = 3*10**8   # speed of wind [m/s]
lambda_func = (c*.001)/f   # um
dia = r*2000  # um
x = (np.pi*dia)/lambda_func
r_eq = 11.25*10**(-6) ### equivalent particle radius

Q=[a.mie(m, x[i]) for i in range(len(dia))]    
Qe = [Q[i][0] for i in range(len(r))]   ### Qext
Qa = [Q[i][2] for i in range(len(r))]   ### Qabs
Qs = [Q[i][1] for i in range(len(r))]   ### Qsca
g = [Q[i][4] for i in range(len(r))]    ### asymmetric factor g(r)

num_scat = 4    # number of scattering

theta_trans = 5*np.pi/180   # Beam divergence angle
theta_rthv = 10*np.pi/180

d = 1              # 接收镜头口径

mu_e = []
mu_a = []
for i in range(len(V)):
    N0 = (2.2512*10**(-9))/((V[i]**1.07)*r_eq**3)
    mu_e.append([])
    mu_a.append([])
    for j in range(len(r)):
        p = (np.exp(-(np.log(2*(r[j]))-m0)**2 / 2*sigma**2))/(2*(r[j])*sigma*np.sqrt(2*np.pi))
        mu_e[i].append(np.pi*r[j]**2 * Qe[j]*N0*p)
        mu_a[i].append(np.pi*r[j]**2 * Qa[j]*N0*p)

mu_e = np.array(mu_e)   # make a form of mu_e as numpy array
mu_a = np.array(mu_a)   # make a form of mu_a as numpy array

print('\n\n\n'"------------ Start Printing-------------",'\n')
print(f"\nAttenuation coefficient, \u03BC_e is \n\n{mu_e }")
print(f"\nAbsorption coefficient, \u03BC_a is \n\n{mu_a }")

L = [(-1 / sum(mu_e[i])) for i in range(len(V))]   # mean free path except the logarithm of random value
H = [(L[i] * 10**3 *np.log10(rand)) for i in range(len(L))]    # thickness along to z direction



"""
    Now Creating class for the implementation of Monte Carlo method. 
    According to this method, we have to make simulation by using new random number per iteration.
"""

class MonteCarlo:
    def __init__(self, L, d, V, r, r_eq, m0, sigma, Qs, g, theta_trans, num_scat, theta_rthv, N):
        self.step = 1*10**(-4)
        self.V = V
        self.r = r
        self.r_eq = r_eq
        self.m0 = m0
        self.sigma = sigma
        self.Qs = Qs
        self.g = g
        self.theta_trans = theta_trans
        self.num_scat = num_scat
        self.L = L
        self.d = d
        self.theta_rthv = theta_rthv
        self.N = N
        
    def g_avg(self):

        nume = 0
        deno = 0
        for i in range(len(self.V)):
            N0 = (2.2512*10**(-9))/((self.V[i]**1.07)*self.r_eq**3)
            for j in range(len(self.r)):
                p = (np.exp(-(np.log(2*(self.r[j]))-self.m0)**2 / 2*self.sigma**2))/(2*(self.r[j])*self.sigma*np.sqrt(2*np.pi))
                nume += np.pi*np.square(self.r[j])*self.Qs[j]*N0*p*self.g[j]
                deno += np.pi*np.square(self.r[j])*self.Qs[j]*N0*p
            
        g_avg = nume / deno
            
        return g_avg

    ### Getting the random numbers, according to Monte Carlo theorem.    
    def randomNumber(self):
        rand_num = []
        i = 0
        while i<self.N:
            rand_num.append(random.random())
            i+=1
        return rand_num


'''
    Now Call the Monte Carlo class. 
    Keep all the needed argument and call the methods.
'''
N = int(input("\n\n\nHow many Photons you want to track?\nPlease enter the N (the number of tracking photons) value: "))
b = MonteCarlo(L, d, V, r, r_eq, m0, sigma, Qs, g, theta_trans, num_scat, theta_rthv, N)
randomNum = b.randomNumber()
g_avg = b.g_avg()


'''
    Now make functions for estimating theta_0, theta_m and phi_m for N numbers of random value.
'''
def theta_0(rand_num, theta_trans):
    theta_0 =  []
    n = 0
    while n<len(rand_num):
        n3 = np.arccos(1 - rand_num[n]*(1-np.cos(theta_trans)))
        theta_0.append(n3)
        n += 1 
    return theta_0

def theta_m(rand_num, g_avg):
    theta_m = []
    n = 0
    while n < len(rand_num):
        n1 = np.arccos(((1+g_avg)**2 - np.square((1-g_avg**2)/(1-g_avg+2*g_avg*rand_num[n]))) / 2*g_avg)
        theta_m.append(n1)
        n += 1
    return theta_m

def phi_m(rand_num):
    phi_m = []
    n = 0
    while n < len(rand_num):
        n2 = 2 * rand_num[n] * np.pi
        phi_m.append(n2)
        # print(n2)

        n += 1
    return phi_m

theta_0 = theta_0(randomNum, theta_trans)
theta_m = theta_m(randomNum, g_avg)
phi_m = phi_m(randomNum)



"""
    Now Calculating the length L by using N number of random value.
"""
L1 = []     # the value of L will be stored into this empty list.
i = 0
for i in range(len(V)):
  l = []
  for j in range(len(randomNum)):
    l.append((L[i] * np.log10(randomNum[j])))
  L1.append(sum(l))


### Now Calculating our main thing.
### How many photons of N will be reached or crossed the path
### will be determined now.

def simulation(theta_0, theta_m, phi_m, num_scat, d, L, theta_rthv, randomNum):  
    numOfPhoton = []
    # num_of_Photon = 0
    for v in range(len(L)):
        num_of_Photon = 0
        for i in range(len(randomNum)): 
            P = np.zeros([1,(num_scat+2),3])   # position
            D = np.zeros([1,(num_scat+1),3])   # direction
            P[:, 0, :] = [0,0,0]                 # initial position
            D[:, 0, :] = [np.sin(theta_0[i]) * np.cos(phi_m[i]), np.sin(theta_0[i])*np.sin(phi_m[i]), np.cos(theta_0[i])]     # initial direction
            P[:, 1, :] = P[:, 0, :] + D[:, 0, :] * L[v]

            for k in range(num_scat):
                D[:, k+1, :] = [(np.sin(theta_m[i]) / np.sqrt(1 - (D[0, k, 2])**2 )) * (D[0, k, 0] * D[0, k, 2] * np.cos(phi_m[i]) - D[0, k, 1] * np.sin(phi_m[i])) + D[0, k, 0] * np.cos(theta_m[i]),
                                (np.sin(theta_m[i]) / np.sqrt(1 - (D[0, k, 2])**2 )) * (D[0, k, 0] * D[0, k, 2] * np.cos(phi_m[i]) + D[0, k, 0] * np.sin(phi_m[i])) + D[0, k, 1] * np.cos(theta_m[i]),
                                -(np.sin(theta_m[i])*np.cos(phi_m[i])*np.sqrt(1 - np.square(D[0, k, 2]))) + D[0, k, 0] * np.cos(theta_m[i])]
                P[:, k+2, :] = P[:, k+1, :] + D[:, k+1, :] * L[v]   # Photon position at i-th scattering

                weight_factor = np.arccos((L[v]-P[0, k+2, 2])/ np.sqrt(np.square(L[v]-P[0, k+2, 2]) + np.square(P[0, k+2, 1]) + np.square(P[0, k+2, 0]-(d/2))))
                if P[0, k+2, 2] > L[v] or D[0, k+1, 2] < 0:
                    break
                elif weight_factor < 10**(-4):
                    break


            for j in range(num_scat):                     
                if P[0, j+1, 2] == L[v] and (P[0, j+1, 0]**2 + P[0, j+1, 1]**2) <= (d/2)**2:
                    num_of_Photon += 1
                    break          
                elif P[0, j+1, 2] < L[v] and P[0, j+2, 2] > L[v]:
                    x = (L[v]-P[0, j, 2]) / (P[0, j+1, 2] - P[0, j, 2]) * (P[0, j+1, 0] - P[0, j, 0]) + P[0, j, 0]
                    y = (L[v]-P[0, j, 2]) / (P[0, j+1, 2] - P[0, j, 2]) * (P[0, j+1, 1] - P[0, j, 1]) + P[0, j, 1]
                    if (x**2 + y**2) <= (d / 2)**2:
                        num_of_Photon += 1
                    break
        numOfPhoton.append(num_of_Photon)

    return numOfPhoton


# Now call the simulation function and get the desired value.
x = simulation(theta_0, theta_m, phi_m, num_scat, d, L1, theta_rthv, randomNum)

W = []
mu_a_pr = [sum(mu_a[i]) for i in range(len(V))]
mu_e_pr = [sum(mu_e[i]) for i in range(len(V))]
for i in range(len(V)):
    w = 1
    W.append([])
    for j in range(x[i]):
        w *= 1 - (mu_a_pr[i] / mu_e_pr[i])
        W[i].append(w)
t = [(1/N) * sum(W[i]) for i in range(len(W))]
c = [np.log10(t[i]) * (-10/H[i])  for i in range(len(H))]       # Attenuation rate

for i in range(len(V)):
    plt.scatter(V[i],c[i])
    plt.xlabel("Visibility/km")
    plt.ylabel("A (Attenuation rate) in Db/Km")
    plt.title("Multiple Scattering of Sand Storm in MC method")
    plt.pause(0.2)
plt.plot(V, c)
plt.show()

