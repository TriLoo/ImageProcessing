import numpy as np
import utils
import re
from sklearn import gaussian_process
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import Model

'''
a = '12 ,  34   '
re_splitA = re.compile(r'[\s\,]+')
print(re_splitA.split(a))
print(re.split(r'[\s\,]+', a))
print(re_splitA.split(a)[2] == '')      # True

b = [a for a in re_splitA.split(a) if a.isdigit()]
print(b)                   # work ! ! !
'''

def f(x):
    return x * np.sin(x)


#X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
#y = f(X).ravel()


print('Test...')
X, y = utils.readData('datas.txt')

X = np.atleast_2d(X).T
y = np.atleast_2d(y).T

#print(X)
#print(y)

print('Data info\n')
print('X = ',X.dtype, X.shape)
print('y = ',y.dtype, y.shape)

# Test datas
#x = np.atleast_2d(np.linspace(0, 50, 2000)).T
x = np.atleast_2d(np.linspace(X.min(), X.max(), 2000)).T

gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, optimizer='Welch')
gp.fit(X, y)

y_pred, sigma_pred = gp.predict(x, eval_MSE=True)

fig = plt.figure(1)
plotf, = plt.plot(X, y, label='Origin')
plotfi, = plt.plot(x, y_pred, label='Predicted')
plotfi, = plt.plot(x, sigma_pred, label='predict_var')

plt.legend(handler_map={plotf: HandlerLine2D(numpoints=4)})
plt.scatter(X.reshape((-1)), y, marker='x', color='g')

plt.show()

