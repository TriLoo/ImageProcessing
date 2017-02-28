function [ z ] = conv2p( x, y )
% -------------------------------------
% Author : smh
% Data   : 2017, 02, 28
% Description :
%       This function calculate periodic convolution.
% -------------------------------------

z = real(ifft2(x.* fft2(y)));

end

