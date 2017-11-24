clc;
close all;

% V = imread('barbara.gif');
V = imread('lena_gray.bmp');
V = double(V);
tic;
y = rlnsw(V, 1);

res = irlnsw(y, 1);
toc;
imshow(res, []);

