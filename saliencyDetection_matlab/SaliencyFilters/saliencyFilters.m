function [ mask ] = saliencyFilters(IRGB, sp )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if max(IRGB(:)) > 1
    img = im2double(IRGB) ;
else
    img = IRGB ;
end

if ~exist('sp','var')
    sp = mexGenerateSuperPixel(img, 100);
end

if ~strcmp(class(sp), 'double')
    sp = double(sp);
end

if min(sp(:)) == 1
    sp = sp -1 ;
end

k=6;
%% step 1, oversegmentation

% segToImg(sp+1);
maxsp=max(sp(:));
trya=sp+1;

% the mean color of the sp
meanLabColor=zeros(maxsp+1,1,3) ;
meanImg = zeros(size(img)) ;
LabImg=RGB2Lab(img);
% imgLab=img;
for channel = 1: size(img,3)
    tempImg = LabImg(:,:,channel);
    for i=1:maxsp+1
        meanLabColor(i,1,channel)=mean( tempImg(trya==i));
    end
end

rgbImg = img*255;
meanrgbColor=zeros(maxsp+1,1,3) ;
for channel = 1: size(img,3)
    tempImg = rgbImg(:,:,channel);
    for i=1:maxsp+1
        meanrgbColor(i,1,channel)=mean( tempImg(trya==i));
        tempImg( trya == i) =  meanrgbColor(i,1,channel) ;
    end
    meanImg(:, :, channel) = tempImg;
end
meanImg = meanImg / 255 ;
%% step 2, element uniqueness
tic
[X, Y] = size(sp) ;
cntr=get_centers(sp+1);
cntr = cntr / max( X , Y);

% U=uniqueness( cntr, labImg, 15 );
% cntr
% meanColor
U=uniqueness( cntr, meanLabColor, 0.25 );
U=(U-min(U(:)))/(max(U(:))-min(U(:)));
tryb = sp+1;
for i=1:maxsp+1
    tryb(tryb==i)=U(i);
end
% figure;
% imshow(tryb);

%% step 3, element distribution
% D = distribution( cntr, labImg , 20);
D = distribution( cntr, meanLabColor , 20);
D =(D-min(D(:)))/ (max(D(:))-min(D(:)));
% D(D<0.3)=0;
% D(D>=0.3)=1;
tryc=sp+1;
for i=1:maxsp+1
    tryc(tryc==i)=1-D(i);
end
% figure;
% imshow(tryc);
%% step 4, saliency assignment

% S = assignment( U, D, cntr, img ) ;
mask = assignment( U,D, cntr, rgbImg, meanrgbColor, sp ,0.03,0.03,k) ;
% S = ( S-min(S(:))) / ( max(S(:)) - min(S(:))) ;

% figure;
% imshow(S);
toc
