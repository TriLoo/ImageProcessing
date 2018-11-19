clc;
% clear;

close all;

% imgA = imread('../datas/source20_1.tif');
% imgB = imread('../datas/source20_2.tif');

% imgA = imread('../datas/1827i.bmp');
% imgB = imread('../datas/1827v.bmp');

imgA = imread('../datas/source22_1.tif');
imgB = imread('../datas/source22_2.tif');
                     
% preprocess inputs
if(size(imgA, 3) == 3)
    imgA = rgb2gray(imgA);
end;
if(size(imgB, 3) == 3)
    imgB = rgb2gray(imgB);
end

if(~isfloat(imgA))    %  OR: isa(imgA, 'double')
    imgA = im2double(imgA);
end
if(~isfloat(imgB))
    imgB = im2double(imgB);
end


% parameters of gaussian filter
% params.gaussian_sigma = 1;
% sigmas of gaussian filter are same as propagation filter.

% parameters of propagation filter
pfilter_sigma_d = [0.5, 1.0, 2.0, 4.0];
pfilter_sigma_r = [1.5, 1.5, 1.5, 1.5];

params.pfilter_w = 3;    
params.pfilter_sigma_d = pfilter_sigma_d;
params.pfilter_sigma_r = pfilter_sigma_r;
% params.pfilter_sigma = [pfilter_sigma_d, pfilter_sigma_r];


% parameters of fusion rules



% do fusion
imgRes = PfilterFusion(imgA, imgB, params);


% show fusion result
imshow(imgRes, []);

