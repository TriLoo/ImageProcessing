function [ err ] = error_cal( V, W, H)
%---------------------------
% Author : smh
% Data   : 2017.02.16
% Description :
%        Calculate the difference of V and W*H.
%        ||V - WH||_F / ||V||_F
%---------------------------
sub = double(V) - W*H;

div = norm(sub,'fro');

err = div / norm(double(V),'fro');

% err = norm(sub, 1);

end

