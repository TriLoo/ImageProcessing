function [ y ] = shearlet_rec( x )
% ---------------------------------
% Author : smh
% Data   : 2017.02.23
% Description :
%       This function implement the invert shearlet transform.
% ---------------------------------

% Input  : x : the shearlet coefficients obtained by shearlet transform
% Output : y : the origin picture.

% level = length(dst) - 1;

% for i=1:2^3 
%     y = real(sum(x{1}(:, :, 3)));
% end

y = real(sum(x{1}, 3));

imshow(y, []);

end
