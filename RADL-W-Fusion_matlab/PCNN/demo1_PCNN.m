clc;
clear all;
close all;
%%
im=imread('A.tif');
im1=imread('B.tif');
im=double(im);
im1=double(im1);
%%
link_arrange=5;
iteration_times=20;
%%
firing_times=PCNN_large_arrange3(im,im1,link_arrange,iteration_times);
%%
figure,imshow(im);
figure,imshow(im1,[]);
figure,imshow(firing_times,[]);
y=entropy(firing_times)