function [ Res ] = GuidedFilter( imgA, imgG, rad, eps)
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

rad = 2 * rad + 1;

tic;

[m, n, z] = size(imgA);
if z == 3
    imgA = rgb2gray(imgA);
end

% Get the mean filtered result of guidance image I 
meanI = nlfilter(imgG, [rad, rad], @average);

% Get the mean filtered result of input image P
meanP = nlfilter(imgA, [rad, rad], @average);

% Get the corr of guidance image I
corrI = nlfilter(imgG .* imgG, [rad, rad], @average);

% Get the corr of input image P
corrIP = nlfilter(imgG .* imgA, [rad, rad], @average);

% Get the Variable of Guidance image I
varI = corrI - meanI .* meanI;

% Get the Cov of I and P
covIP = corrIP - meanI .* meanP;

a = covIP ./ (varI + eps);
b = meanP - a .* meanI;

meanA = nlfilter(a, [rad, rad], @average);
meanB = nlfilter(b, [rad, rad], @average);

Res = meanA .* imgG + meanB;

toc;

imshow(uint8(imgA), []);
figure;
imshow(uint8(Res), []);

end

