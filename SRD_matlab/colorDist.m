function [ imgDist ] = colorDist( imgHist )
% --------------------
% Author : smher
% Data   : 2017.7.17
% Description :
%       This file implement the color distance calculations.
% --------------------

imgDist = zeros(1, 256);

for i = 1 : 256
    temp = 0;
    for j = 1:256
        temp = temp + abs(i - j) * imgHist(j);
    end
    imgDist(i) = temp;
end

end

