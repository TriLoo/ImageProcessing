function [ res ] = inverseRGF( layers, weights )
%U -----------------------
% Author : smh
% Date   : 2018.01.05
% Description:
%   This file implementing the inverse transform of RGF, in ZJF2015
%U -----------------------

len = size(layers, 3);
[M, N, z] = size(layers);

res = zeros(M, N);
for i = 1 : len
    res = res + layers(:, :, i) * weights(i);
end

end

