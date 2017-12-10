function [ originNew ] = my_Horizontal_Direction_Prediction( origin, ori_QuartPixel_interp )
% ---------------------------
% Author : smher
% Data   : 2017.12.02
% Description : 
%   This file includes my own implementation of redundant directional
%   lifting wavelet based on ADL Wavelet idea. Step: Predict.
%   Inputs : 
%       origin : the input image
%       ori_QuartPixel_interp : is the input image after Sinc
%                               interpolation.
%   Outputs : 
%       originNew : the output after predict
%       Predict_direc : the optimal direction obtained by minimum local residual energy.
% ---------------------------

[MyHeight,MyWidth] = size(origin);
Dir = 4; % total 9 direction

originT = padarray(ori_QuartPixel_interp, [Dir, Dir], 'circular', 'both');
originNew = zeros(MyHeight, MyWidth);

tempSum = 0;

Divd = 2 * (2 * Dir + 1);

for i = Dir+1 : Dir + MyHeight
    for j = 1 : MyWidth
        for k = -Dir : Dir
            tempSum = tempSum + originT(i-1, 4 * j - 3 + k + Dir) + originT(i + 1, 4 * j -3 + Dir + k);
        end
        originNew(i - Dir, j) = origin(i - Dir, j) - (tempSum / Divd);      
        tempSum = 0;
    end
end
   
end

