"""\
This script implements a back-to-back diode model
to fit current-voltage data.
The fit parameters are the two Schottky barriers
and two ideality factors for both junctions.

Islandlab 2024
"""

import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt

#load dataset
data = np.loadtxt("test_dataset.csv", delimiter=",", unpack=False)
I = data[:,3]
vg_steps = 50
v_steps = 100
I = I.reshape(vg_steps, v_steps)
V = np.linspace(-1, 1, v_steps)
Vg = np.linspace(10, -10, vg_steps)
I = -I[49,:] #take one IV curve out of dataset

#set constants
s1 = 100e-8 #cm^-2
s2 = 15e-8 #cm^-2
T = 300 #K
A = 80.3 #Richardson constant for MoS2, A cm^-2 K^-2
k = .00008617 #eV K^-1
q = 1

#Set model fit parameters and initial guesses
pm=lm.Parameters()
pm.add('phib01',value=0.5,vary=True,min=0,max=2);
pm.add('phib02',value=0.5,vary=True,min=0,max=2);
pm.add('n1',value=1,vary=True,min=0.1,max=5);
pm.add('n2',value=1,vary=True,min=0.1,max=5);

#Define back-to-back model
def func(paramsin, V):
    phib01 = paramsin['phib01'].value
    phib02 = paramsin['phib02'].value
    n1 = paramsin['n1'].value
    n2 = paramsin['n2'].value
    phib1 = phib01+(q*V/2*(1-(1/n1)))
    phib2 = phib02-(q*V/2*(1-(1/n2)))
    Is1 = s1*A*T**2*np.exp(-phib1/(k*T))
    Is2 = s2*A*T**2*np.exp(-phib2/(k*T))
    Itot = ((2*Is1*Is2*np.sinh((q*V)/(2*k*T)))/((Is1*np.exp((q*V)/(2*k*T)))+(Is2*np.exp((-q*V)/(2*k*T)))))
    return Itot

#Define error function and perform fit
funcerr = lambda p,x,y: func(p,x)-y
fitout=lm.minimize(funcerr,pm,args=(V,I))
fitted = fitout.params

pars=[fitout.params['phib01'].value,
         fitout.params['phib02'].value,
         fitout.params['n1'].value,
         fitout.params['n2'].value]

#Print fit resutls and calculate residuals
print(lm.fit_report(fitout.params))
fit=func(fitted,V)
resid=fit-I

#Create plot including residuals
fig1=plt.figure(3,figsize=(10,5))
f1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(V,I,'k.')
plt.plot(V,fit,'b')
f1.set_xticklabels([]) #We will plot the residuals below, so no x-ticks on this plot
plt.title('Diode model fit')
plt.ylabel('Current (A)')
from matplotlib.ticker import MaxNLocator
plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower')) #Removes lowest ytick label
f1.annotate('Fitting parameters:\nphib01 = %.2f\nphib02 = %.2f\nn1 = %.2f\nn2 = %.2f' \
    %(pars[0],pars[1],pars[2],pars[3]),xy=(.05,.95), \
    xycoords='axes fraction',ha="left", va="top", \
bbox=dict(boxstyle="round", fc='1'),fontsize=10)
f2=fig1.add_axes((.1,.1,.8,.2))
res=plt.plot(V,resid,'k--')
plt.ylabel('Residuals (A)')
plt.xlabel('Voltage (V)')

#Save data
save_a = np.hstack([V.reshape(100,1),fit.reshape(100,1)])
np.savetxt('100.txt', save_a)

plt.show()
