function [ imgRes ] = fusion( imgA, imgB )
% -------------------------------
% Author : smher
% Date   : 2017.04.26
% Description :
%       this function implement the fusion of two images based on
%       hot-target-oriented and guided filter . 
% -------------------------------

% Input : imgA : the visible image
%         imgB : the infrared image
%         All input images are doubles !

%   define some parameters
eps = 0.01;
r = 5;
w = 0.8;

[m,n] = size(imgA);

tic;

histImg = histFusion(imgA, imgB);

gfImg = guided_filter(imgA, imgB, r, eps);

imgRes = zeros(m,n);

for i = 1 : m
    for j = 1 : n
        valA = imgA(i,j);
        valH = histImg(i,j);
        valGF = gfImg(i,j);
        imgRes(i,j) = (valA - valGF) * w + valH; 
    end
end

toc;

end

