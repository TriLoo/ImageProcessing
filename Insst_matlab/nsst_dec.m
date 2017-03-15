function [ dst, shear_f ] = nsst_dec( x, shear_parameters )
% ----------------------------------
% Author : smh
% Data   : 2017. 02. 28
% Description :
%       This function implement the nsst decompose.
% ----------------------------------

[M, N] = size(x);
level = length(shear_parameters.dcomp);

z = rlnsw(x, level);

dst = cell(1, level+1);
dst{1,1} = z{1, 1};

shear_f = cell(1, level);


% calculate the directional filters
for i=1:level
    w_s = shearing_filters_meyer(shear_parameters.dsize(i), shear_parameters.dcomp(i));
    for k = 1:2^shear_parameters.dcomp(i)
        shear_f{i}(:, :, k) = (fft2(w_s(:, :, k), M, N) ./ max(M, N));
    end
end

for i=1:level
    d = sum(shear_f{i}, 3);
    for k = 1:2^shear_parameters.dcomp(i)
        shear_f{i}(:, :, k) = shear_f{i}(:, :, k) ./d;
        dst{i+1}(:, :, k) = conv2p(shear_f{i}(:, :, k), z{i+1});
    end
end
% disp('1234');
% for i = 1:level
%     shear_f{i} = shearing_filters_meyer(shear_parameters.dsize(i), shear_parameters.dcomp(i)) .* sqrt(shear_parameters.dsize(i));
%     for k = 1:2^shear_parameters.dcomp(i)
%         dst{1, i+1}(:, :, k) = conv2(z{i+1}, shear_f{1, i}(:, :, k), 'same');
%     end
% end

end

