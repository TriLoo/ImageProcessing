function [ imgOut ] = usample( imgIn, samp )
% ---------------------
% Author : smher
% Data   : 2017. 08. 01
% Description :
%       This file implement the up sample of input image under Fourier
%       Transform domain.
% ---------------------

[m, n] = size(imgIn);

imgIn = double(imgIn);

um = m * samp;
umnum = um - m;
un = n * samp;
unnum = un - n;

imgFFT = fft2(imgIn);

nR = round(n / 2);
mR = round(m / 2);

imgRow = [imgFFT(:, 1:nR), zeros(m, unnum), imgFFT(:, nR + 1 : n)];

imgCol = [imgRow(1:mR, :); zeros(umnum, un); imgRow(mR + 1 : m, :)];

imgOut = ifft2(imgCol);

end

