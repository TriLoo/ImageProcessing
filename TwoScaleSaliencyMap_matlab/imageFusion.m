function [ res ] = imageFusion( imgA, imgB )
% ---------------------
% Author : smh
% Date   : 2017.12.11
% Description:
%   This is the top-level function of 'two-scale image fusion of visible
%   and infrared images using saliency detection'.
% ---------------------

% [M, N] = size(imgA);

imgA = im2double(imgA);
imgB = im2double(imgB);

se = fspecial('average', [35, 35]);
baseA = imfilter(imgA, se, 'replicate');
baseB = imfilter(imgB, se, 'replicate');

detailA = imgA - baseA;
detailB = imgB - baseB;

[wmA, wmB] = WeightedMap(imgA, imgB);

fuseDetail = wmA .* detailA + wmB .* detailB;
fuseBase = (baseA + baseB) / 2;

res = fuseBase + fuseDetail;

end

