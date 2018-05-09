import numpy as np
from scipy import stats
import utils
from matplotlib import pyplot as plt

X, y = utils.readData('datas.xlsx')

X = np.array(X).T
y = np.array(y).T

logY = np.log(y)
Mu = np.mean(logY)
Sigma = np.sqrt(np.var(logY))
Scale = np.exp(Mu)
Shape = Sigma

mode = np.exp(Mu - Sigma**2)
mean = np.exp(Mu + (Sigma**2)/2)

pdf = stats.lognorm.pdf(y, Shape, loc=0, scale=Scale)

def distribution(x):
    Div = 1.0 / (x * np.sqrt(Sigma) * np.sqrt(2 * np.pi))
    return Div * np.exp(-(x - Mu)**2 / (2 * Sigma))

'''
label = [distribution(a) for a in y]
'''
plt.figure()
plt.plot(X, pdf)
plt.fill_between(X, pdf, where=(X < Mu/Sigma), alpha=0.15)
plt.fill_between(X, pdf, where=(X > Mu/Sigma), alpha=0.15)
plt.fill_between(X, pdf, where=(X < Mu/Sigma**2), alpha=0.15)
plt.fill_between(X, pdf, where=(X < Mu/Sigma**2), alpha=0.15)
plt.show()


