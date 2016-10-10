import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("online_news_popularity.csv", sep=", ", engine="python")

# Treat as one data set
X = df[[col for col in df.columns if col not in ['url', 'shares', 'cohort']]]
y = np.log(df.shares).reshape(-1,1)

X = np.hstack((np.ones_like(y), X))

def objective(X, y, w, reg=1e-6):
	X = np.matrix(X)
	err = X * w - y
	err = float(err.transpose() * err)
	return (err + reg * np.abs(w).sum()) / len(y)

def grad_objective(X, y, w):
	X = np.matrix(X)
	return X.transpose() * (X * w - y) / len(y)

def prox(x, gamma):	
	# Modify data point by point
	for i in range(len(x)):
		xVal = x.item(i, 0)			
		if xVal > gamma:
			x.itemset(i, xVal - gamma)
		elif xVal < -gamma:
			x.itemset(i, xVal + gamma)
		else:
			x.itemset(i, 0)	
	return x

def lasso_grad(
	X, y, reg=1e-6, lr=1e-3, tol=1e-6,
	max_iters=300, batch_size=256, eps=1e-5,
	verbose=False, print_freq=5,
):
	X = np.matrix(X)
	y = y.reshape(-1,1)
	w = np.linalg.solve(X.transpose() * X, X.transpose() * y)
	
	index = np.random.randint(0, X.shape[0], size=batch_size)
	runningObj = [objective(X[index], y[index], w, reg=reg)]
	grad = grad_objective(X[index], y[index], w)
	
	while len(runningObj)-1 <= max_iters and np.linalg.norm(grad) > tol:
		if (len(runningObj)-1) % print_freq == 0:
			print("[i={}] objective: {}. sparsity = {:0.2f}".format(
				len(runningObj)-1, runningObj[-1], (np.abs(w) < reg*lr).mean()
			))
		
		index = np.random.randint(0, X.shape[0], size=batch_size)
		grad = grad_objective(X[index], y[index], w)

		# Apply threshold function
		w = prox(w - lr * grad, reg * lr)
		
		runningObj.append(objective(X[index], y[index], w, reg=reg))

	print("[i={}] done. sparsity = {:0.2f}".format(
		len(runningObj)-1, (np.abs(w) < reg*lr).mean()
	))
	
	return w, runningObj

def lasso_path(X, y, reg_min=1e-8, reg_max=10, regs=10, **grad_args):
	W = np.zeros((X.shape[1], regs))
	tau = np.linspace(reg_min, reg_max, regs)

	for i in range(regs):
		W[:,i] = lasso_grad(X, y, reg=1/tau[i], max_iters=1000,batch_size=1024, **grad_args)[0].flatten()
	
	return tau, W
	
tau, W = lasso_path(X, y, reg_min=1e-15, reg_max=0.02, regs=10, lr=1e-12)
# plt.title("Lasso Path")
# plt.plot(tau, W.transpose())

# find most important features
np.array(df.columns)[np.argsort(-W[:,9])[:5]+1]
w, obj = lasso_grad(X, y, reg=1e5, lr=1e-12, eps=1e-2, max_iters=2500, batch_size=1024, verbose=True, print_freq=250)
plt.title("Lasso Objective Convergence")
plt.ylabel("Stochastic Objective")
plt.xlabel("Iteration")
plt.plot(obj)

plt.show()