function [ sal ] = SaliencyMap( img )
% --------------------
% Author : smh
% Date   : 2017.12.11
% Description:
%   Saliency detection using 'Two-scale image fusion of visible and
%   infrared images using saliency detection'.
% --------------------

se = fspecial('average', [35, 35]);
eleA = imfilter(img, se, 'replicate');

eleB = medfilt2(img, [3, 3]);

sal = abs(eleA - eleB);

end

