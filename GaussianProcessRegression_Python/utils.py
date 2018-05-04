'''
Used to read from a file to generate source data.
'''
import numpy as np

import os
import sys

def readData(filename):
    if not os.path.exists(filename):
        print("File {} doesn't exist.".format(filename))
    else:
        print("Reading {}".format(filename))
        with open(filename, 'r') as f:
            X, y = [], []
            for line in f.readlines():
                #print(type(line), len(line.strip()))
                a = [float(x) for x in line.split() if x.isdigit()]
                #print(type(a), np.shape(a))
                # OR
                #b = np.array(a)
                #if b.shape[0] == 2:
                if np.shape(a)[0] == 2:        # Only numpy.array have the 'shape' attribute
                    X.append(a[0])
                    y.append(a[1])
                else:
                    print('Data format error, the dimension is not 2!')

            return X, y
