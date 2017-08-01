function [ imgOut ] = subsample( imgIn, samp )
% ----------------------
% Author : smher
% Data   : 2017. 08. 1
% Description :
%       This file implement the subsample of input image
% ----------------------

[m, n] = size(imgIn);

sm = round(m / samp);
sn = round(n / samp);

sImg = zeros(sm, sn);

p = 1;
q = 1;

for i = 1 : samp : m
    for j = 1 : samp : n
        sImg(p, q) = imgIn(i, j);
        q = q + 1;
    end
    q = 1;
    p = p + 1;
end

imgOut = sImg;

end

