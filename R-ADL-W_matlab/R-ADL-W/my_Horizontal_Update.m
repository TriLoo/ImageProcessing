function [ originNew ] = my_Horizontal_Update(origin, originPre, ori_QuartPixel_interp )
% ------------------------
% Author : smher
% Data   : 2017.12.02
% Description:
%   This file includes my own implementation of redundant directional
%   lifting wavelet based on ADL wavelet. Step: Update
%   Inputs : 
%       originPre : the predicted image obtained in 'Predict Step', can also be substitued by
%                   ori_QuartPixel_interp.
%       origin    : the input image
%       ori_QuartPixel_interp : the input image after Sinc interpolation.
%       Predict_direc : the optimal predict & update direction
%   Outputs : 
%       originNew : the outputs of the 'Predict step', i.e. low-pass part.
%   Function : 
%       Update: means to update the even position pixels based on
%               Prediction values located at odd index.
% ------------------------

[MyHeight, MyWidth] = size(origin);

Dir = 4;
originT = padarray(ori_QuartPixel_interp, [Dir, Dir], 'circular', 'both');

tempSum = 0;
originNew = zeros(2 * MyHeight, MyWidth);
originNew(1:MyHeight, :) = origin;
originNew(MyHeight + 1 : end, :) = originPre;

Divd = 2 * (2 * Dir + 1);

for i = Dir + 1 : Dir + MyHeight
    for j = 1 : MyWidth
        for k = -Dir : Dir
            tempSum = tempSum + originT(i - 1, 4 * j - 3 + Dir + k) + originT(i + 1, 4 * j - 3 + Dir + k);
        end
        originNew(i - Dir, j) = origin(i - Dir, j) + (tempSum / ( 2 * Divd));
        tempSum = 0;
    end
end

end
