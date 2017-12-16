function [ cA, cV, cH, cD, PredictBlocks ] = RADLWavelet( img )
% -------------------------
% Author : smh
% Date   : 2017.12.05
% Description:
%   This file including the top function of Redundant ADL wavelet based on
%   'my_Horizontal_Direction_Prediction', 'my_Horizontal_Direction_Update',
%   'Horizontal_Sinc_interpolation'.
%   Inputs:
%       img: the input image under decomposing.
%   Outputs:
%       cA : the 'LL' component of Wavelet
%       cV : the 'LH' component of Wavelet, i.e. low-pass of horizontal
%       direction & high-pass of vertical direction.
%       cH : the 'HL' component of Waveletl, i.e. high-pass of horizontal
%       direction & low-pass of vertical direction.
%       cD : the 'HH' component of Wavelet
%       PredictBlocks: a 1 * 3 cell containing direction information.
%   Copy Right @ * * GPLv2 * *
% -------------------------

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

% PART I, Get the LL, LH
% Begin Vertical Predict     
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(originHorizontal_L);
[imgV_Predict, Predict_Block_V_I] = my_Horizontal_Direction_Prediction(originHorizontal_L, ori_QuartPixel_interp);
% Begin Vertical Update
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(imgV_Predict);
imgVertical_L = my_Horizontal_Update(imgV_Predict, originHorizontal_L, ori_QuartPixel_interp, Predict_Block_V_I);

% PART II, Get the HL, HH
% Begin Vertical Predict     
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(originHorizontal_H);
[imgV_Predict, Predict_Block_V_II] = my_Horizontal_Direction_Prediction(originHorizontal_H, ori_QuartPixel_interp);
% Begin Vertical Update
ori_QuartPixel_interp = Horizontal_Sinc_interpolation(imgV_Predict);
imgVertical_H = my_Horizontal_Update(imgV_Predict, originHorizontal_H, ori_QuartPixel_interp, Predict_Block_V_II);

% generate Predict_Block.
PredictBlocks = {Predict_Block_H, Predict_Block_V_I, Predict_Block_V_II};



% transpose the matrix, get the four child components
cD = imgVertical_H(MyWidth+1 : end, :)';
cH = imgVertical_H(1:MyWidth, :)';
cV = imgVertical_L(MyWidth+1:end, :)';
cA = imgVertical_L(1:MyWidth, :)';

% TEST, show the decomposing results.
% figure;
% imshow(uint8(img));
% title('Input Image');
% figure;
% subplot(2, 2, 1);
% imshow(uint8(abs(cA)));
% title('LL');
% subplot(2, 2, 2);
% imshow(uint8(abs(cV)));
% title('LH');
% subplot(2, 2, 3);
% imshow(uint8(abs(cH)));
% title('HL');
% subplot(2, 2, 4);
% imshow(uint8(abs(cD)));
% title('HH');


end

