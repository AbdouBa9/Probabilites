#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 07:20:07 2020

@author: mac
"""

#Chargement des librairies
import numpy as np 
import scipy.stats as sps 
import matplotlib.pyplot as plt 
from scipy.stats import poisson
from scipy.stats import expon

##Loi normale de probabilités
E = np.random.randn(int(1e5))
x = np.linspace(-4,4, 1000)
f_x = sps.norm.pdf(x)
plt.plot(x,f_x,"r",label="theory")

plt.hist(E,bins=50,normed=1,label="Data")
plt.legend(loc='best')

#Loi Binomiale
n, p, N = 20, 0.3, int(1e4)
B = np.random.binomial(n, p, N)
f = sps.binom.pmf(np.arange(n+1), n, p)
plt.hist(B,bins=n+1,normed=1,range=(0.5,n+.5),color = "white",label="loi empirique")
plt.stem(np.arange(n+1),f,"r",label="loi theorique")
plt.legend()
plt.grid()

#Loi exponentielle
abs=np.linspace(0,8,200) 
plt.plot(abs,expon.pdf(abs,scale=1)) 
plt.plot(abs,expon.pdf(abs,scale=1/3)) 
plt.grid() 
plt.xlabel("Axe des X")
plt.ylabel("Données")
plt.show()

#Loi de poisson
fig, ax = plt.subplots(1, 1)
 
mu = 0.6
mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')

x = np.arange(poisson.ppf(0.01, mu),
              poisson.ppf(0.99, mu))
ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
 
rv = poisson(mu)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
         label='frozen pmf')
ax.legend(loc='best', frameon=False)
plt.show()