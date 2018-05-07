import numpy as np
import utils
import scipy.optimize

#b = [[np.exp(np.abs(i + j)) for i in range(5)] for j in range(5)]
'''
b = np.eye(5, 5)
c = np.linalg.inv(b)
print(c)
print(np.dot(b, c))
print(np.log(b))
print(np.matrix.transpose(b))
'''


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

def calculate_likelihood(x, datas, label):
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
    #Lx = -np.log(R_inv) + N * np.log(np.dot(np.dot(np.matrix.transpose(label - OneMu), R_inv), label - OneMu))
    tempD = np.linalg.det(R)
    Lx = -np.log(tempD) + N * np.log(tempC)

    return Lx


datas, label = utils.readData('datas.xlsx')

datas = np.asarray(datas)
label = np.asarray(label)
N = datas.shape[0]
#datas = np.reshape(datas, newshape=(N, 1))
#label = np.reshape(label, newshape=(N, 1))
print('datas shape = ', datas.shape)
print('label shape = ', label.shape)
theta = calculate_mu(1, datas, label)
print('mu = ', theta)

Lx = calculate_likelihood(1, datas, label)
print('likelihood = ', Lx)


'''G
c = np.zeros(shape=(5, 5))
for i in range(5):
    for j in range(5):
        c[i][j] = np.exp(np.abs(i + j))

print(b == c)            # return a 5 * 5 True matrix

d = np.eye(3, 3)
print(d)
d = np.mat(d)
print(d.I)
print(np.linalg.det(d))


def SumVal(*kws):
    sum = kws[0] + kws[1]
    return sum


print(SumVal(1, 2, 3))


# Test Scipy.optimize.fmin_l_bfgs_b

x = np.linspace(0, 10, 10)

y = 4 * x

test = y + np.random.normal(loc=0., scale=3.)
print(test)


#def func(x, *kws):
def func(x, X, Y):
    #X = kws[0]
    #Y = kws[1]
    #X, Y = kws
    sum = 0.0
    N = X.shape[0]
    for i in range(N):
        sum = sum + np.abs(x**2 * X[i] - Y[i])

    return sum


#def g(x, *kws):               # works !
def g(x, X, Y):              # Also works !
    #X = kws[0]
    #Y = kws[1]
    #X, Y = kws
# np.sum() returns a scalar if axis is None
    return np.array(2 * x * np.sum(X))      # must return a numpy.array explicitly
    #return np.sum(X)      # must return a numpy.array explicitly


args = [x, y]
print('g = ', g(1, *args))
x0 = np.array([3.0])

# check the gradient
#checkGrad = scipy.optimize.check_grad(func=func, grad=g, x0=x0, *args)
checkGrad = scipy.optimize.check_grad(func, g, x0, x, y)
print('check out = ', checkGrad)

predictX = scipy.optimize.minimize(fun=func, x0=x0, args=(x, y), method='L-BFGS-B', jac=g)              # Work !

# Seem don't work ?
a, b, c = scipy.optimize.fmin_l_bfgs_b(func=func, x0=x0, fprime=g, args=(x, y), bounds=[(1.0, 5.0)])    # bounds must be a list of turples
#a, b, c = scipy.optimize.fmin_l_bfgs_b(func=func, x0=x0, fprime=g, args=(x, y), bounds=[(1.0, 5.0)])    # bounds must be a list of turples
print('x =', predictX.x)
print('Predict error = ', func(predictX.x, x, y))

#print(func(predictX.x, x, y))
print(func(a, x, y))
print('estimated position of the minimum = ', a)
print('value of func at the minimum = ', b)
print('Message ', c['warnflag'])
print('Error ', c['task'])
'''

'''
def f(p):
    x, y = p
    z = (1 - x)**2 + 100 * (y - x**2) * 2
    return z


def fprime(p):
    x, y = p
    dx = -2 + 2 * x + 100 * (y - x**2)**2
    dy = 200 * y - 200 * x**2
    return np.array([dx, dy])

init_point = (-2, 2)

result = scipy.optimize.minimize(fun=f, x0=init_point, jac=fprime, method='L-BFGS-B')
print(result)
'''
