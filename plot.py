import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy

#plot1
mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

data = np.genfromtxt('content/messwerte.txt', unpack=True)

x_0 = data[0]
x = data[0]

y = data[1]/60
y_err = np.sqrt(data[1] )/60


plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'Aktivität$/\si{\becquerel}$')

plt.grid(True, which='both')


# Fitvorschrift
def f(x, A, B):
    return A*x+B      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(f, x, y)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

lin = np.linspace(x[0], x[-1], 1000)
plt.plot(lin, f(lin, *params), "xkcd:orange", label=r'Regression' )
plt.plot(x_0, y, ".", color="xkcd:blue", label="Messwerte")
plt.errorbar(x, y, yerr=y_err, elinewidth=0.7, capthick=0.7, capsize=3, fmt=".", color="xkcd:blue", label="Ungenauigkeit")


plt.tight_layout()
plt.legend()
plt.savefig('build/messung1.pdf')
plt.clf()
