function [ u, s, v, p ] = choosing( Z )
%---------------------------
% Author : smh
% Data   : 2017.02.16
% Description : 
%       choose the value of factorization rank p.
%       see paper : "New SVD based initializaion strategy for Non-Negative Matrix Factorization"
%---------------------------

Z = double(Z);
[u, s, v] = svd(Z);

sum1 = sum(s);
sum2 = sum(sum1);

extract = 0;

p = 0;

dsum = 0;

while(extract/sum2 < 0.90)
    p = p + 1;
    dsum = dsum + s(p, p);
    extract = dsum;
end

end

