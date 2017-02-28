clc;
close all;

V = imread('barbara.gif');

shear_parameters.dcomp = [3, 3, 4];
shear_parameters.dsize = [32, 32, 16];

% test the INSST decompose ...
[dst, shear_f] = nsst_dec(V, shear_parameters);

res = nsst_rec(dst);

imshow(res, []);