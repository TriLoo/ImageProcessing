function [ dst, shear_f ] = nsst_dec( x, shear_parameters )
% ----------------------------------
% Author : smh
% Data   : 2017. 02. 28
% Description :
%       This function implement the nsst decompose.
% ----------------------------------

[m, N] = size(x);
level = length(shear_parameters.dcomp);

z = rlnsw(x, level);

dst = cell(1, level+1);
dst{1} = z{1};

shear_f = cell(1, level);

for i = 1:level
    shear_f{i} = shearing_filters_meyer(shear_parameters.dsize(i), shear_parameters.dcomp(i)) .* sqrt(shear_parameters.dsize(i));
    for k = 1:2^shear_parameters.dcomp(i)
        dst{i+1}(:, :, k) = conv2(z{i+1}, shear_f{i}(:, :, k), 'same');
    end
end

end

