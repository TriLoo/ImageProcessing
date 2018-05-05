import numpy as np
import scipy.optimize
from sklearn import gaussian_process

'''
The L-BFGS-B optimization can be called in SciPy: scipy.optimize.minimize()

scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', ...)
'''


def lbfgsOptimize(f, g, eps, x0, datas, labels, m=5, maxK = 100):
    '''
    Non-linear optimization algorithm based on L-BFGS-B

    :param f: input function
    :param g: the gradient of input function
    :param eps: the minimum precision
    :param x0: the initial data
    :param datas: input datas
    :param labels: output labels
    :param m: the size of list
    :param maxK: the maximum iteration of optimization
    :return: the minimum value of f
    '''

    return 0.0


class GaussProcessRegress:
    def __init__(self):
        self.theta = 0.
        self.meanVal = [0.]
        self.varVal = [0.]
        self.retVal = [0.]

    def fit(self, func, x0, X, y):
        # x0: the initial value of theta
        self.theta = scipy.optimize.fmin_l_bfgs_b(func, x0=x0, method='L-BFGS-B')
        return self.theta

    def predict(self, x):
        return self.retVal


'''
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()

x = np.atleast_2d(np.linspace(0, 10, 1000)).T

gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X, y)

y_pred, sigma_pred = gp.predict(x, eval_MSE=True)

fig = plt.figure(1)
plotf, = plt.plot(x, f(x), label='xsin(x)')
plotfi, = plt.plot(x, y_pred, label='Predicted')
plotfi, = plt.plot(x, sigma_pred, label='predict_var')

plt.legend(handler_map={plotf: HandlerLine2D(numpoints=4)})
plt.scatter(X.reshape((-1)), y, marker='x', color='g')

plt.show()
'''
