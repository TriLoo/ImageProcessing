clc;
% clear;

close all;

% imgA = imread('../datas/source20_1.tif');
% imgB = imread('../datas/source20_2.tif');

% imgA = imread('../datas/1827i.bmp');
% imgB = imread('../datas/1827v.bmp');

% imgA = imread('../datas/source22_1.tif');
% imgB = imread('../datas/source22_2.tif');

imgA = imread('../datas/Kaptein_1123_IR.bmp');
imgB = imread('../datas/Kaptein_1123_Vis.bmp');
                     
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

% parameters of propagation filter & gaussian filter
pfilter_sigma_d = [0.5, 1.0, 2.0, 4.0];
pfilter_sigma_r = [1.5, 1.5, 1.5, 1.5];
% below window sizes changed at 2018.11.20 night. calculate the window size using: 2 * ceil(2 * sigma) + 1
% window_size = 2 * ceil(2 * pfilter_sigma_d) + 1;
window_size = [3, 5, 7, 9];

% params.pfilter_w = 3;    
params.window_size = window_size;
params.pfilter_sigma_d = pfilter_sigma_d;
params.pfilter_sigma_r = pfilter_sigma_r;
% params.pfilter_sigma = [pfilter_sigma_d, pfilter_sigma_r];


% parameters of fusion rules


% do fusion
imgRes = PfilterFusion(imgA, imgB, params);


% Calculate the ssim metric
ssimA = ssim(imgRes, imgA);
ssimB = ssim(imgRes, imgB);
ssimTotal = ssimA + ssimB;

% show fusion result
imshow(imgRes, []);

