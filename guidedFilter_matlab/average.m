function [ imgAvg ] = average( imgIn )
% --------------------
% Author : smher
% Data   : 2017. 07. 31
% Description :
%       This file implements the mean filter of input image
% --------------------

len = numel(imgIn);

sumX = 0.0;

for i = 1 : len
    sumX = sumX + imgIn(i);
end

imgAvg = sumX / len;

end

