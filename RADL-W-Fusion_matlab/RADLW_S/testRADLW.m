clc;
clear;
close all;

% img = imread('./barb.bmp');
img = imread('../datas/source20_2.tif');
% img = imread('../datas/LWIR-MarnehNew_15RGB_603.tif');
% img = imread('../datas/NIR-MarnehNew_15RGB_603.tif');
% img = imread('../datas/Jeep_Vis.bmp');
% img = imread('../datas/IR_meting003_g.bmp');


if size(img, 3) == 3
    img = double(rgb2gray(img));
else
    img = double(img);
end

[MyHeight, MyWidth] = size(img);

% Horizontal Predict
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(img);
[imgH_Predict, Predict_Block_H] = my_Horizontal_Direction_Prediction(img, ori_QuartPixel_interp);

% Horizontal Update
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(imgH_Predict);
originNew = my_Horizontal_Update(imgH_Predict, img, ori_QuartPixel_interp, Predict_Block_H);


% --------------------- Vertical Predict & Update -----------------------%

originNew = originNew';
originHorizontal_H = originNew(:, MyHeight + 1 : end);
originHorizontal_L = originNew(:, 1 : MyHeight);  

% Begin Vertical Predict     PART I, Get the LL, LH
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(originHorizontal_L);
[imgV_Predict, Predict_Block_V_I] = my_Horizontal_Direction_Prediction(originHorizontal_L, ori_QuartPixel_interp);
% Begin Vertical Update
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(imgV_Predict);
imgVertical_L = my_Horizontal_Update(imgV_Predict, originHorizontal_L, ori_QuartPixel_interp, Predict_Block_V_I);

originLH = imgVertical_L';
originLH = abs(originLH);
originLH = uint8(originLH);
figure;
imshow(originLH, []);

% Begin Vertical Predict     PART II, Get the HL, HH
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(originHorizontal_H);
[imgV_Predict, Predict_Block_V_II] = my_Horizontal_Direction_Prediction(originHorizontal_H, ori_QuartPixel_interp);
% Begin Vertical Update
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(imgV_Predict);
imgVertical_H = my_Horizontal_Update(imgV_Predict, originHorizontal_H, ori_QuartPixel_interp, Predict_Block_V_II);

originHL = imgVertical_H';
originHL = abs(originHL);
originHL = uint8(originHL);
figure;
imshow(originHL, []);

% transpose the matrix, get the four child components
imgVertical_HH = imgVertical_H(MyWidth+1 : end, :)';
imgVertical_HL = imgVertical_H(1:MyWidth, :)';
imgVertical_LH = imgVertical_L(MyWidth+1:end, :)';
imgVertical_LL = imgVertical_L(1:MyWidth, :)';
% --------------------- Above: Wavelet -----------------------%


% **************************************** % 
%         Add Fusion Algorithm Here        %
% **************************************** % 



% --------------------- Below: Inverse Wavelet -----------------------%
% Inverse Vertical Predict Update:      PART II
imgVertical_HH = imgVertical_HH';
imgVertical_HL = imgVertical_HL';
% Begin Inverse Update
imgV_QuartPixel_interp = Horizontal_Sinc_interpolation(imgVertical_HH);
imgVerticalH = my_Inverse_Horizontal_Update(imgVertical_HL, imgV_QuartPixel_interp, Predict_Block_V_II);
% Begin Inverse Predict
% !!!!
% For redundant ADL-Wavelet, only need inverse Update, No need for Inverse
% Prediction.
% !!!!


% Inverse Vertical Predict Update:      PART I
imgVertical_LH = imgVertical_LH';
imgVertical_LL = imgVertical_LL';
% Begin Inverse Update
imgV_QuartPixel_interp = Horizontal_Sinc_interpolation(imgVertical_LH);
imgVerticalL = my_Inverse_Horizontal_Update(imgVertical_LL, imgV_QuartPixel_interp, Predict_Block_V_I);
% Begin Inverse Predict
% !!!!
% For redundant ADL-Wavelet, only need inverse Update, No need for Inverse
% Prediction.
% !!!!

imgHorizontalH = imgVerticalH';
imgHorizontalL = imgVerticalL';
imgH_QuartPixel_interp = Horizontal_Sinc_interpolation(imgHorizontalH);
imgRes = my_Inverse_Horizontal_Update(imgHorizontalL, imgH_QuartPixel_interp, Predict_Block_H);


imgRes_show = abs(imgRes);
imgRes_show = uint8(imgRes_show);
figure;
imshow(imgRes_show, []);

