function [ imgRes ] = inverse_RADLWavelet( cA, cV, cH, cD, PredictBlock )
% ------------------------------
% Author : smh
% Date   : 2017.12.05
% Description:
%   This file including the implementation of inverse RADL Wavelet
%   transform based on 'my_Inverse_Horizontal_Prediction',
%   'my_Inverse_Horizontal_Update', 'Horizontal_Sinc_Wavelet'.
%   Inputs:
%       c* : the outputs of 'RADLWavelet'
%       PredictBlock: a 1 * 3 cell containing the direction information.
%   Outputs:
%       The restored image after inverse RADLWavelet.
% ------------------------------

% --------------------- Inverse Wavelet -----------------------%
% prepare direction block
Predict_Block_V_II = PredictBlock{3};
Predict_Block_V_I = PredictBlock{2};
Predict_Block_H = PredictBlock{1};


% Inverse Vertical Predict Update:      PART II
imgVertical_HH = cD';
imgVertical_HL = cH';
% Begin Inverse Update
imgV_QuartPixel_interp = Horizontal_Sinc_interpolation(imgVertical_HH);
imgVerticalH = my_Inverse_Horizontal_Update(imgVertical_HL, imgV_QuartPixel_interp, Predict_Block_V_II);
% Begin Inverse Predict
% !!!!
% For redundant ADL-Wavelet, only need inverse Update, No need for Inverse
% Prediction.
% !!!!


% Inverse Vertical Predict Update:      PART I
imgVertical_LH = cV';
imgVertical_LL = cA';
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


% imgRes_show = abs(imgRes);
% imgRes_show = uint8(imgRes_show);
% figure;
% imshow(imgRes_show, []);

end

