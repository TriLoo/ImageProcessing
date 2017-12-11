clc;
% clear;
close all;

% imgA = imread('./datas/source20_1.tif');
% imgB = imread('./datas/source20_2.tif');

imgA = imread('./datas/VIS_18dhvR.bmp');
imgB = imread('./datas/IR_18rad.bmp');

imgF = imageFusion(imgA, imgB);

imgF = im2uint8(imgF);
imwrite(imgF, '18Road.jpg');

subplot(1, 3, 1);
imshow(imgA, []);
title('Input A');
subplot(1, 3, 2);
imshow(imgB, []);
title('Input B');
subplot(1, 3, 3);
imshow(imgF, []);
title('Result');

