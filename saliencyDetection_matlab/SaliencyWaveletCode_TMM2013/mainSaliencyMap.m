%% This code is used to compute bottom-up saliency map using low-level 
%% features based on Wavelet transform. This is a revised version of
%% the following publication for public distribution on web. 
%% Please cite the following paper when you use this code.
%%
%% Reference:
%% Nevrez Imamoglu, Weisi Lin, and Yuming Fang: 
%% A Saliency Detection Model Using Low-Level Features Based on Wavelet Transform. 
%% IEEE Transactions on Multimedia 15(1): 96-105 (January 2013)

%% DISCLAIMER: The Matlab code provided is only for evaluation of the algorithm. 
%% Neither the authors of the code, nor affiliations of the authors can be held 
%% responsible for any damages arising out of using this code in any manner. 
%% Please use the try out code at your own risk.

clear all;
close all;
clc;

[FileName,PathName] = uigetfile({'*.tif;*.png;*.bmp;*.jpg;*.jpeg;*.gif','Image Files'});
myFile = [PathName FileName];

tic
%Read the RGB image file and convert it to CIE Lab color space
Irgb = imread(myFile);
C = makecform('srgb2lab');
lab = applycform(Irgb,C);

%calculate saliency
waveName = 'db5';
[Smix, SL, SG] = funWaveletSaliency(lab,waveName);
%Enhance the calculated saliency to obtain final saliency map
SaliencyMap = funSaliencyEnhance(Smix);
toc

figure; 
subplot(2,2,1); imshow(Irgb,[]); title('Input Image')
subplot(2,2,2); imshow(SaliencyMap,[]); title('Final Saliency Map');
subplot(2,2,3); imshow(SL,[]); title('Local Saliency Map')
subplot(2,2,4); imshow(SG,[]); title('Global Saliency Map')


