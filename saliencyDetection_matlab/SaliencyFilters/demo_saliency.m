% function demo_saliency(numberofsp)
clc;
close all;
addpath './MEX/'
addpath './SLIC/'
% cd('D:\documents\saliency\2012_saliency_filters');
k=6;
% pic='../datas/Jeep_IR.bmp';
pic = './leaf.bmp'
% pic='../datas/lena.jpg';

numberofsp=50;
img  = im2double(imread(pic));

mask_path=strcat('./temp/',pic(length(pic)-9:length(pic)-4));
mask_path=strcat(mask_path,'_');    
mask_path=strcat(mask_path,num2str(numberofsp));
mask_path=strcat(mask_path,'.mat');

%% step 1, oversegmentation
if exist(mask_path,'file')
    load(mask_path);
else
    sp  = mexGenerateSuperPixel(img, numberofsp);
%     save (mask_path, 'sp') ;
end
sp = double(sp);
% segToImg(sp+1);
maxsp=max(sp(:));
trya=sp+1;

% the mean color of the sp
meanLabColor=zeros(maxsp+1,1,3) ;
meanImg = zeros(size(img)) ;
LabImg=RGB2Lab(img);
% imgLab=img;
for channel = 1: 3
    tempImg = LabImg(:,:,channel);
    for i=1:maxsp+1
        meanLabColor(i,1,channel)=mean( tempImg(trya==i));
    end
end

rgbImg = img*255;
meanrgbColor=zeros(maxsp+1,1,3) ;
for channel = 1: 3
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
S = assignment( U,D, cntr, rgbImg, meanrgbColor, sp ,0.03,0.03,k) ;
% S = ( S-min(S(:))) / ( max(S(:)) - min(S(:))) ;

% figure;
% imshow(S);
toc
%% adaptive threshold
[height,width] = size(S) ;
thres = 2/(height*width)*sum(S(:));
thres_S = im2bw(S,thres) ;


%% write the result
% imwrite(meanImg,'leaf_my_abstract.png');
% imwrite(tryb,'leaf_my_uniqueness.png');
% imwrite(1-tryc,'leaf_my_distribution.png');
% imwrite(S,'lock_my_result.png');

%% show the result

figure;
subplot(2,3,1);
imshow(img);
title('Source Image');
subplot(2,3,2);
imshow(meanImg);
title('Super Segmentation');
subplot(2,3,3) ;
imshow(tryb);
title('Element Uniqueness');
subplot(2,3,4);
imshow(tryc);
title('Element Distribution');
subplot(2,3,5);
imshow(S);
title('Final Result');
subplot(2,3,6);
imshow(thres_S);
title('Threshold Result');
