clc;

img = imread('lena.jpg');
[m, n, c] = size(img);

sigma = [1, 1];
w = 3;

resultImg = pfilter(img, img, w, sigma);

imshow(resultImg, []);
