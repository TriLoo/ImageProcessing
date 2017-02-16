function [ W, H, err ] = I_nmf_svd( V, maxiter )
%---------------------------
% Author : smh
% Data   : 2017.02.16
% Description : 
%       implement the nmf based on imporved svd ways.
%---------------------------

m = size(V, 1);

[u, s, v, p] = choosing(V);

W = abs(u(:, 1:p));
H = abs(s(1:p, :)*v');

[W, H] = multiplicative(V, H, W, maxiter);

err = error_cal(V, W, H);

% show the figure
subplot(1, 2, 1);
imshow(V,[]);
title('origin image');
subplot(1,2,2);
imshow(W*H, []);
title('NMFed image');

end

