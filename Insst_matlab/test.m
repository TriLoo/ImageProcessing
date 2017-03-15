clc;
close all;

% change the image name to your own image name
V = imread('barbara.gif');

V = double(V);

shear_parameters.dcomp = [3, 3, 4];
shear_parameters.dsize = [32, 32, 16];

level = length(shear_parameters.dcomp);

% ------------ test rlnsw part ---------------%
% y = rlnsw(V, level);
% 
% res = irlnsw(y, level);
% 
% imshow(res, []);

% ------------ test complete ----------------%
% 

shear_parameters.dcomp = [3, 3, 4];
shear_parameters.dsize = [32, 32, 16];
tic;
% test the INSST decompose ...
[dst, shear_f] = nsst_dec(V, shear_parameters);
% disp('123');
res = nsst_rec(dst);
toc;
imshow(res, []);

