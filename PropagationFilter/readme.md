# Propagated image filtering

## Usage

* In Matlab

* mex pfilter-matrix.cpp pfilter.cpp -output pfilter

* call the pfilter function in a `.m` file:

  OutputImage = pfilter(guided image, image, w, sigma)

  shape of two input images can be m * n * 1 (gray-scale image) or m * n * c (color image)

  w is the radius of filter window

  sigma: shape (1, 2), including sigma_d, sigma_r
