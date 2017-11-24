function [ y ] = my_convert( x )
% -----------------------------
% Author : smh
% Data   : 2017. 02. 28
% Description :
%       This function change the scope of x to 225

a = max(max(x));
b = min(min(x));

y = (x - b)./(a - b) * 225;

end

