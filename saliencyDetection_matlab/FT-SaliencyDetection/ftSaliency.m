function [ salFT ] = ftSaliency( img )
% ----------------------
% Author : smh
% Date   : 2017.12.04
% Description:
%   This file including frequency-tuned saliency detection based on
%   'Frequency-tuned Saliency Region Detection' & 'Saliency_CVPR2009.m'
% ----------------------

[m, n, z] = size(img);

gfrgb = imfilter(img, fspecial('gaussian', 3, 3), 'symmetric', 'conv');
%---------------------------------------------------------
% Perform sRGB to CIE Lab color space conversion (using D65)
%------------------------------------------x---------------
% cform = makecform('srgb2lab', 'whitepoint', whitepoint('d65'));
if (z == 3)
    cform = makecform('srgb2lab', 'AdaptedWhitePoint', whitepoint('d65'));   % changed by smh
    lab = applycform(gfrgb,cform);
    
    l = double(lab(:,:,1)); lm = mean(mean(l));
    a = double(lab(:,:,2)); am = mean(mean(a));
    b = double(lab(:,:,3)); bm = mean(mean(b));
    
    salFT = (l-lm).^2 + (a-am).^2 + (b-bm).^2;
elseif z == 1
    l = double(gfrgb); lm = mean(mean(l));
    
    salFT = (l-lm).^2;
end

end

