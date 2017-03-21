clc;
close all;

% V = imread('barbara.gif');
V = imread('lena_gray.bmp');
V = double(V);
tic;
y = rlnsw(V, 3);

res = irlnsw(y, 3);
toc;
imshow(res, []);

