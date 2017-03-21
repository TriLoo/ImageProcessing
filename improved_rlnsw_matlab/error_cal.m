function [ err ] = error_cal( V, R)
%---------------------------
% Author : smh
% Data   : 2017.03.15
% Description :
%        Calculate the difference of V and restored image .
%        ||V - WH||_F / ||V||
%---------------------------
sub = double(V) - R;

div = norm(sub,'fro');

err = div / norm(double(V),'fro');

% err = norm(sub, 1);

end

