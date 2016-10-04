import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def normalDensity(X, mu, cov):
	n = X.shape[0]	
	diff = np.matrix(X - mu)
	exp = -diff.transpose() * np.linalg.inv(cov) * diff / 2.0
	return np.exp(exp) / (math.sqrt(2 * math.pi) ** n * math.sqrt(np.linalg.det(cov)))

def partA():	
	mu1 = np.matrix([[0], [0]])
	sigma1 = np.matrix([[6, 8],[8, 13]])

	X1 = np.linspace(-10.0, 10.0, num=1001)
	X2 = np.linspace(-10.0, 10.0, num=1001)
	normDensity = []
	for x1 in X1:
		xNorms = []
		for x2 in X2:			
			xNorms.append((normalDensity(np.matrix([[x1], [x2]]), mu1, sigma1)).item(0))
		normDensity.append(xNorms)
	
	figure = plt.figure()
	plot = figure.add_subplot(111, projection='3d')
	plot.plot_surface(X1, X2, normDensity, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	plt.show()

def partB():	
	mu2 = np.matrix([5])	
	sigma2 = np.matrix([14])

	X = np.linspace(-10.0, 20.0, num=101)
	normDensity = []
	for x in X:		
		normDensity.append((normalDensity(np.matrix([x]), mu2, sigma2)).item(0))	

	figure = plt.figure()
	plt.style.use('ggplot')
	normDensityPlot, = plt.plot(X, normDensity, 'r')

	plt.show()

def partC():
	mu1 = np.matrix([[0], [0]])
	mu2 = np.matrix([5])

	sigma11 = np.matrix([[6, 8],[8, 13]])
	sigma12 = np.matrix([[5], [11]])
	sigma22 = np.matrix([14])
	
	X1 = np.linspace(-40.0, 10.0, num=1001)
	X2 = np.linspace(-40.0, 10.0, num=1001)

	cond_sigma = sigma11 - sigma12 * np.linalg.inv(sigma22) * sigma12.transpose()
	normDensity = []
	for x1 in X1:
		xNorms = []
		for x2 in X2:
			cond_mu = mu1 + sigma12 * np.linalg.inv(sigma22) * (x2 - mu2)
			xNorms.append((normalDensity(np.matrix([[x1], [x2]]), cond_mu, cond_sigma)).item(0))	
		normDensity.append(xNorms)

	figure = plt.figure()
	plot = figure.add_subplot(111, projection='3d')
	plot.plot_surface(X1, X2, normDensity, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	plt.show()

def partD():
	mu1 = np.matrix([[0], [0]])
	mu2 = np.matrix([5])

	sigma11 = np.matrix([[6, 8],[8, 13]])
	sigma12 = np.matrix([[5], [11]])
	sigma22 = np.matrix([14])
	
	X1 = np.linspace(-5.0, 20.0, num=1001)

	cond_sigma = sigma22 - sigma12.transpose() * np.linalg.inv(sigma11) * sigma12
	normDensity = []
	for x1 in X1:
		cond_mu = mu2 + sigma12.transpose() * np.linalg.inv(sigma11) * (x1 - mu1)
		normDensity.append((normalDensity(np.matrix([x1]), cond_mu, cond_sigma)).item(0))			
			
	figure = plt.figure()	
	normDensityPlot, = plt.plot(X1, normDensity, 'r')

	plt.show()

# partA()
# partB()
# partC()
partD()
