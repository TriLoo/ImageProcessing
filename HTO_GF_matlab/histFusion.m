function [ imgRes ] = histFusion( imgA, imgB)
% -------------------------
% Author : smher
% Date   : 2017.04.26
% Description :
%       this function implement the fusion of two images based on the image
%       B pixel histogram.
% -------------------------

% Input : A is the visiable image
%         B is the infrared image

[m,n] = size(imgA);

imgSize = m * n;

imgRes = zeros(m,n);


histB = calHist(imgB);
histB = histB/ imgSize;

for i = 1:m
    for j = 1:n
        valA = imgA(i,j);
        valB = imgB(i,j);
        imgRes(i,j) =  valA * (1 - histB(1,valB + 1)) + valB * histB(1, valB + 1);  
    end
end


end

