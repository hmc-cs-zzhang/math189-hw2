import numpy as np
import math
import matplotlib.pyplot as plt

def normalDensity(x, mu, var):		
	exp = -((x - mu) ** 2) / (2.0 * var)
	return np.exp(exp) / (math.sqrt(2 * math.pi * var))

def laplaceDensity(x, mu, b):
	exp = -np.absolute(x - mu) / b
	return 1.0 / (2 * b) * np.exp(exp)

x = np.linspace(-3.0, 3.0, num=101)
normD = [normalDensity(xx, 0, 1) for xx in x]
laplaceD = [laplaceDensity(xx, 0, 1) for xx in x]

figure = plt.figure()
plt.title("Normal Density vs. Laplace Density")
plt.xlim([-3.0, 3.0])
plt.ylim([0, 0.55])
plt.xlabel("x")
plt.ylabel("P(x)")
normDensityPlot, = plt.plot(x, normD, 'r')
laplaceDensityPlot, = plt.plot(x, laplaceD, 'b')
plt.legend((normDensityPlot, laplaceDensityPlot), ('Normal', 'Laplace'), loc=1)

plt.show()