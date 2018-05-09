import utils
import GPy
import numpy as np
#GPy.plotting.change_plotting_library('plotly')
GPy.plotting.change_plotting_library('plotly_offline')
import matplotlib.pyplot
from IPython.display import display
#import plotly
#plotly.tools.set_credentials_file(username='songmh', api_key='ORbkxXXakGsYvnVKo5Di')

X, y = utils.readData('datas.xlsx')
X = np.atleast_2d(X).T
y = np.atleast_2d(y).T

print('Data Info.')
print('X = ', X.dtype, X.shape)

#N = X.shape[0]
#X = np.reshape(X, newshape=(N, 1))
#y = np.reshape(y, newshape=(N, 1))

Kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1.)

model = GPy.models.GPRegression(X, y, Kernel)
model.optimize(messages=True)

#model.save_model(output_filename='GP.model')
print(model.optimizer_array) # lengthscale = 34.8179, i.e. theta in RBF = 0.1694

print(model)
display(model)
fig = model.plot(plot_density = True)
GPy.plotting.show(fig, filename='GPonGPy.html')




