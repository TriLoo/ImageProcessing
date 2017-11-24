clc,clear
%%%% gray image fusion
% I = load_images( '.\sourceimages\grayset',1); 
% F = GFF(I);
% imshow(F);
%%%% color image fusion
I = load_images( './sourceimages/myselfs',1); 
%I = load_images( './sourceimages/colourset',1); 
tic;
F = GFF(I);
toc;
figure,imshow(F);