function [ res ] = guidedfilter( imgI, imgP, rad, sig, eps)
% -------------------
% Author : smh
% Date   : 2017.12.23
% Description:
%       This file implement the 'Guided Image Filter' based on 'Gaussian
%       filter'.
% -------------------

% eps_stable = 10^-6;

se = fspecial('gaussian', rad, sig);
gauI = imfilter(imgI, se);
gauP = imfilter(imgP, se);

corrI = imfilter(imgI .* imgI, se);
corrIp = imfilter(imgI .* imgP, se);

varI = corrI - gauI .* gauI;
covIp = corrIp - gauI .* gauP;

a = covIp ./ (varI + eps);
b = gauP - a .* gauI;

gauA = imfilter(a, se);
gauB = imfilter(b, se);

res = gauA .* imgI +  gauB;

end

