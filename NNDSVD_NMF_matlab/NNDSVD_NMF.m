function [W, H, err] = NNDSVD_NMF(V, rank, flag, maxiter)
% ----------------------------
% Author : smher
% Data   : 2017. 06. 06
% Description :
%       This function implement the NMF based on NNDSVD
% ----------------------------
V = double(V);

[W, H] = NNDSVD(V, rank, flag);

[W, H] = multiplicative(V, H, W, maxiter);

err = error_cal(V, W, H);

result = W * H;

result = uint8(result);

imwrite(result, 'result.png');

imshow(result, []);

end