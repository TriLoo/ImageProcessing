import numpy as np
import utils
import re
import Model


def f(x):
    return x * np.sin(x)


#X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
#y = f(X).ravel()


print('Test...')
X, y = utils.readData('datas')

print(type(X))     # the type is 'list'
print('-----------')
print(y)
# Change the list to array
X = np.array(X)
y = np.array(y)

a = '12 ,  34'
print(re.split(r'[\s+\,]+', a))
