### Computation of Mie Efficiencies for given 
### complex refractive-index ratio m=m'+im" 
### and size parameter x=k0*a, where k0= wave number in ambient 
### medium, a=sphere radius, using complex Mie Coefficients
### an and bn for n=1 to nmax,
### s. Bohren and Huffman (1983) BEWI:TDD122, p. 103,119-122,477.
### Result: m', m", x, efficiencies for extinction (qext), 
### scattering (qsca), absorption (qabs), backscattering (qback), 
### asymmetry parameter (asy=<costeta>) and (qratio=qb/qsca).
### Uses the function "Mie_ab" for an and bn, for n=1 to nmax.
### C. Mätzler, May 2002, revised July 2002.

import numpy as np
from scipy.special import jv, yv

class pyMie:
    def mie_ab(self, m, x):
        z = m * x
        nmax = np.round(2+x+4*(x**(1/3)))
        nmx = np.round(max(nmax,np.abs(z))+16)
        n = np.arange(1,nmax+1)
        nu = n + 0.5
        
        sx = np.sqrt(0.5*np.pi*x)
        px = sx*jv(nu,x)
        p1x = np.append(np.sin(x), px[0:int(nmax)-1])
        chx = -sx*yv(nu,x)
        ch1x = np.append(np.cos(x), chx[0:int(nmax)-1])
        gsx = px-(0+1j)*chx
        gs1x = p1x-(0+1j)*ch1x
        
        Dnx = np.zeros(int(nmx), dtype=complex)
        for i in range(int(nmx)-1, 1, -1):
            Dnx[i-1] = (i/z)-(1/(Dnx[i]+i/z))
        
        D = Dnx[1:int(nmax)+1]
        da = D/m+n/x
        db = m*D+n/x
        an = (da*px-p1x)/(da*gsx-gs1x)
        bn = (db*px-p1x)/(db*gsx-gs1x)

        return an, bn
        
    
    def mie(self, m, x):
        if x == 0:
            result = [0, 0, 0, 0, 0, 1.5]
            print(result)
        elif x > 0:
            nmax = np.round(2+x+4*(x**(1/3)))
            n = np.arange(1,nmax+1)
            cn = 2*n+1 # n1
            c1n = n*(n+2)/(n+1) # n2
            c2n = cn/(n*(n+1))  # n3
            x2 = x ** 2
            
            an, bn = self.mie_ab(m, x)
            g1 = [an.real[1:int(nmax)],
                  an.imag[1:int(nmax)],
                  bn.real[1:int(nmax)],
                  bn.imag[1:int(nmax)]]
            g1 = [np.append(x, 0.0) for x in g1]
            
            qext = (2/x2)*np.sum(cn*(an.real+bn.real))
            qsca = (2/x2)*np.sum(cn*(an.real**2+an.imag**2+bn.real**2+bn.imag**2))
            qabs = qext-qsca
            
            asy = (4/(qsca*x2))*np.sum((c1n*(an.real*g1[0]+an.imag*g1[1]+bn.real*g1[2]+bn.imag*g1[3]))+(c2n*(an.real*bn.real+an.imag*bn.imag)))
            
            qback = (1/x2)*(np.abs(np.sum(cn*((-1)**n)*(an-bn)))**2)
            qratio = qback/qsca
            
            return qext, qsca, qabs, qback, asy, qratio