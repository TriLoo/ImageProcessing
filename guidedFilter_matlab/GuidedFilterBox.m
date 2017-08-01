function [ Res ] = GuidedFilterBox( imgA, imgG, rad, eps)
% ------------------------
% Author : smher
% Data   : 2017. 07. 31
% Description:
%   This file implement the guided filter based on "Guided Image Filtering"
%       Input imgA: filtering image
%             imgG: guided image
%             rad : the radius
%             eps : regularization 
%       Output Res: Result image
% ------------------------


imgA = double(imgA);
imgG = double(imgG);

tic;

[m, n, z] = size(imgA);
if z == 3
    imgA = rgb2gray(imgA);
end

N = boxFilter(ones(m, n), rad);

meanI = boxFilter(imgG, rad) ./ N;
meanP = boxFilter(imgA, rad) ./ N;

corrI = boxFilter(imgG .* imgG, rad) ./ N;
corrIp = boxFilter(imgG .* imgA, rad) ./ N;

varI = corrI - meanI .* meanI;
covIp = corrIp - meanI .* meanP;

a = covIp ./ (varI + eps);
b = meanP - a .* meanI;

meanA = boxFilter(a, rad) ./ N;
meanB = boxFilter(b, rad) ./ N;

Res = meanA .* imgG + meanB;

toc;

imshow(uint8(imgA), []);
figure;
imshow(uint8(Res), []);

end

