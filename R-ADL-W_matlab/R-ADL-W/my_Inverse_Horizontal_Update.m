function [ img_low_updated ] = my_Inverse_Horizontal_Update( img_low, img_QuartPixel_interp )
% ----------------------
% Author: smher
% Date  : 2017.12.03
% Description:
%   This file includes the implementation of reduandant ADL wavelet:
%   inverse horizontal update: update the low-pass part based on high-pass
%   part.
% ----------------------

[MyHeight, MyWidth] = size(img_low);
img_low_updated = img_low;

Dir = 4;
originT = padarray(img_QuartPixel_interp, [Dir, Dir], 'circular', 'both');

img_low_updated = img_low;

Dvid = 2 * (2 * Dir + 1);
tempSum = 0;
for i = Dir + 1 : Dir + MyHeight
    for j = 1 : MyWidth
        for k = -Dir : Dir
            tempSum = tempSum + originT(i - 1, j * 4 - 3 + Dir + k) + originT(i + 1, j * 4 - 3 + Dir + k); 
        end
        img_low_updated(i - Dir, j) = img_low(i - Dir, j) - (tempSum / ( 2 * Dvid));
        tempSum = 0;
    end
end

end

