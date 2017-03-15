clc;
close all;

V = imread('barbara.gif');
V = double(V);

y = rlnsw(V, 3);

res = irlnsw(y, 3);

imshow(res, []);

