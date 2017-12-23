clc;
clear;
close all;

addpath('./RGF_sal');

imgA = imread('./datas/source20_1.tif');
imgB = imread('./datas/source20_2.tif');

res = rgfsaliencyFusion(imgA, imgB);

res = im2uint8(res);

subplot(1, 3, 1);
imshow( imgA, []);
title('Input A');
subplot(1, 3, 2);
imshow(imgB, []);
title('Input B');
subplot(1, 3, 3);
imshow(res, []);
title('Fusion Result');

