function [ w_s ] = shearing_filters_meyer( n1, level )
% -------------------------------------------
% Author : smh
% Data   : 2017. 02.38
% Description :
%       This function computers the directional/sharing filters using the
%       meyer window.
% -------------------------------------------

[x11, y11, x12, y12, F1] = gen_x_y_cordinates(n1);

wf = windowing(ones(2*n1, 1), 2^level);
w_s = zeros(n1, n1, 2^level);

for k=1:2^level
    temp = wf(:, k) * ones(n1, 1)';
    w_s(:, :, k) = rec_from_pol(temp, n1, x11, y11, x12, y12, F1);
    w_s(:, :, k) = real(fftshift(ifft2(fftshift(w_s(:, :, k)))))./sqrt(n1);
end

end
