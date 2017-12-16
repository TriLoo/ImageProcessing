function [ sd ] = squareDifference( img )
% --------------------------
% Author : smh
% Date   : 2017.12.05
% Description:
%   This file including the impelementation of calculation of square
%   difference of input image.
% --------------------------

% get the mean.
img = double(img);

meanVal = mean(mean(img));

tempM = img - meanVal;

sd = sum(sum(tempM .* tempM));


end

