import numpy as np
import scipy.optimize
from sklearn import gaussian_process

'''
The L-BFGS-B optimization can be called in SciPy: scipy.optimize.minimize()

scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', ...)
'''

# 需要提供梯度
# 嗨森矩阵可选，因为BFGS可以迭代更新得到H的近似
def lbfgsOptimize(f, g, x0, kws=()):
    minVal = scipy.optimize.minimize(fun=f, x0=x0, args=kws, method='L-BFGS-B', jac=g)

    return minVal           # optimize result


# 使用不需要梯度的算法: Powell, or Nelder-Mead
def otherOptimize(f, x0, kws=()):
    minVal = scipy.optimize.minimize(fun=f, x0=x0, args=kws, method='Powell')

    return minVal


def calculate_mu(x, datas, label):
    N = datas.shape[0]
    One = np.ones(shape=(N, 1))
    R = np.array([[np.exp(-x * np.abs(datas[i] - datas[j])) for i in range(N)] for j in range(N)])
    # Change list to 2d array
    print('R shape = ', R.shape)
    R_inv = np.linalg.inv(R)

    # Calculate the mean \mu^
    Mu = np.linalg.inv(np.dot(np.dot(np.matrix.transpose(One), R_inv), One))
    Mu = np.dot(Mu, np.dot(np.dot(np.matrix.transpose(One), R_inv), label))

    return Mu


def likelihood(x, datas, label):
    N = datas.shape[0]
    One = np.ones(shape=(N, 1))
    R = np.array([[np.exp(-x * np.abs(datas[i] - datas[j])) for i in range(N)] for j in range(N)])
    # Change list to 2d array
    print('R shape = ', R.shape)
    R_inv = np.linalg.inv(R)
    theta = calculate_mu(x, datas, label)

    #OneMu = np.dot(One, theta)
    OneMu = One * theta         # Work well
    label = np.reshape(label, newshape=(N, 1))
    tempA = label - OneMu
    tempB = np.dot(np.matrix.transpose(tempA), R_inv)
    tempC = np.dot(tempB, tempA)
    tempD = np.linalg.det(R)
    Lx = np.log(tempD) - N * np.log(tempC)           # Note that the result is inverted to use minimize()

    return Lx


def likelihood_Grad(x, datas, label):
    N = datas.shape[0]
    One = np.ones(shape=(N, 1))
    One_T = np.matrix.transpose(One)
    R = np.array([[np.exp(-x * np.abs(datas[i] - datas[j])) for i in range(N)] for j in range(N)])
    # Change list to 2d array
    #print('R shape = ', R.shape)
    R_inv = np.linalg.inv(R)
    dR = np.array([[-np.abs(datas[i] - datas[j]) for i in range(N)] for j in range(N)])
    dR_inv = -1 * np.dot(np.dot(R_inv, dR), R_inv)
    theta = calculate_mu(x, datas, label)
    OneMu = One * theta         # Work well
    OneMu_T = One_T * theta

    # PART I
    tempA = np.sum(dR)
    tempSumA = np.sum(R)       # return 1 * 1
    tempSumB= tempSumA**2
    tempB = tempA / tempSumB
    dMu = tempB * np.dot(np.dot(One_T, R_inv), label) + 1 / tempSumA * (np.dot(np.dot(One_T, dR_inv), label))
    dMu_One_T = np.dot(One_T, dMu)
    dMu_One = np.dot(One, dMu)

    tempC = label - OneMu
    tempC_T = np.matrix.transpose(tempC)
    tempD = np.dot(np.dot(dMu_One_T, R_inv), tempC)
    tempE = np.dot(np.dot(tempC_T, dR_inv),tempC)
    tempF = np.dot(np.dot(tempC_T, R_inv), dMu_One)
    dA = tempD + tempE + tempF
    A = np.dot(np.dot(tempC_T, R_inv), tempC)

    d_logA = N * dA / A

    # PART II
    d_detR = -1 * np.trace(np.dot(R_inv, dR))

    return d_detR + d_logA


class GaussProcessRegress:
    def __init__(self, X, y, One, Mu):
        self.theta = 0.
        self.meanVal = [0.]
        self.varVal = [0.]
        self.retVal = [0.]
        self.X = X
        self.y = y
        self.R_inv = 0.
        self.R = 0.
        self.One = One
        self.Mu = Mu

    def initGP(self, datas, labels):
        self.X = datas
        self.y = labels

    def likelihood(self, x, datas, labels):
        self.X = datas
        self.y = labels
        N = datas.shape[0]
        self.R = [[np.exp(-x * np.abs(datas[i] - datas[j])) for i in range(N)] for j in range(N)]
        self.R_inv = np.linalg.inv(self.R)
        self.One = np.ones(shape=(N, 1))

    def fit(self, x0, X, y):
        # x0: the initial value of theta
        optimizeResult = lbfgsOptimize(likelihood, likelihood_Grad, x0, X, y)
        self.theta = optimizeResult.x         # the solution of the optimization (Ndarray)

        return self.theta

    def predict(self, x):
        self.retVal = self.theta + np.dot(np.dot(np.mat(self.__calculate_r(x, self.X)), self.R_inv), np.mat(self.y - np.dot(self.One, self.Mu)))
        return self.retVal

    def __calculate_r(self, x, X):
        r = np.zeros(shape=X.shape)
        N = X.shape[0]

        for i in range(N):
            r[i] = np.exp(-self.theta * np.abs(x - X[i]))

        return r


# use default parameters
def gpSklearn(X, y, x):
    '''
    Gaussian Process Regression based on Sciket-Learn
    nugget: 用于设定高斯回归的噪声

    :param X:
    :param y:
    :param x:
    :return:
    '''

    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget=1e-5)
    #gp = gaussian_process.GaussianProcess(theta0=1e-2, nugget=1e-4)
    gp.fit(X, y)

    y_pred, sigma_pred = gp.predict(x, eval_MSE=True)

    return y_pred, sigma_pred

