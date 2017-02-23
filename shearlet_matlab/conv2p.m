function [ z ] = conv2p( x, y )
% -----------------------------------
% Author : smh
% Data   : 2017. 02. 23
% Description :
%       This function performs periodic convolution.
% -----------------------------------
% Input: x and y are the input arrays. X is assumed to be fft2 of x
%
% Output: z is the resultant convolution

z = real(ifft2(x.*fft2(y)));

end

