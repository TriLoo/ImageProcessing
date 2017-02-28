function [ y ] = nsst_rec( dst )
% ----------------------------------
% Author : smh
% Data   : 2017. 02. 28
% Description :
%       This function implement the invert decompose of improved NSST.
% ----------------------------------

% Input : dst : the coefficient obtained in nsst_dec.
% Output : y  : the origin image.

level = length(dst) - 1;

z = cell(1, level + 1);

z{1} = dst{1};

for i = 1:level
    z{i+1} =real(sum(dst{i+1}, 3));
end

y = real(inv_rlnsw(z, level));

end

