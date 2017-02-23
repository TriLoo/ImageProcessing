function [ y ] = meyer( x )
%---------------------------------
% Author : smh
% Data   : 2017.02.23
% Description :
%       This function computes the meyer window function of signal x.
%       see wikipedia : meyer wavelet for details
%---------------------------------

if -1/3+1/2 < x & x < 1/3+1/2
    y = 1;
elseif (1/3+1/2 <= x & 2/3+1/2) | (-2/3+1/2 <= x <= -1/3+1/2)
    w = 3*abs(x-1/2) - 1;
    z = w^4*(35-84*w +70*w^2 -20*w^3);
    y = cos(pi/2*(z))^2;
else
    y = 0;
end

end

