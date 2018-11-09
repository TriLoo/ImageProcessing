clc;

img = imread('lena.jpg');
[m, n, c] = size(img);

sigma = [1, 1];

resultImg = pfilter(img, img, sigma);

imshow(resultImg, []);
