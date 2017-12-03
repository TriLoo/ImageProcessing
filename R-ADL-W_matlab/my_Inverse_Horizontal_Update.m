function [ img_low_updated ] = my_Inverse_Horizontal_Update( img_low, img_QuartPixel_interp, Prediction_Block )
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

M = 8;
N = 8;

% first line
for j = 1 : MyWidth
    k = Prediction_Block(ceil(1 / M), ceil(j / N));
    temp = img_low(1, j);
    
    x = j * 4 - 3 + k;
    y = j * 4 - 3 - k;
    
    if x < 1
        x = -x + 2;
    end
    if y < 1
        y = -y + 2;
    end
    if x > 4 * MyWidth
        x = 4 * MyWidth - 7;
    end
    if y > 4 * MyWidth
        y = 4 * MyWidth - 7;
    end
    
    temp1 = img_QuartPixel_interp(2, x);
    temp2 = img_QuartPixel_interp(2, y);
    
    img_low_updated(1, j) = temp - (temp1 + temp2) / 4;
end

% last MyHeight - 2 lines
for j = 1 : MyWidth
    for i = 2 : MyHeight - 1
        k = Prediction_Block(ceil(i / M), ceil(j / N));
        temp = img_low(i, j);
        
        x = j * 4 - 3 + k;
        y = j * 4 - 3 - k;
        
        if x < 1
            x = -x + 2;
        end
        if y < 1
            y = -y + 2;
        end
        if x > 4 * MyWidth
            x = 4 * MyWidth - 7;
        end
        if y > 4 * MyWidth
            y = 4 * MyWidth - 7;
        end

        temp1 = img_QuartPixel_interp(i - 1, x);
        temp2 = img_QuartPixel_interp(i + 1, y);

        img_low_updated(i, j) = temp - (temp1 + temp2) / 4;
    end
end

% last one line
for j = 1 : MyWidth
    k = Prediction_Block(ceil(MyHeight / M), ceil(MyWidth / N));
    temp = img_low(MyHeight, j);
    
    x = j * 4 - 3 + k;
    y = j * 4 - 3 - k;

    if x < 1
        x = -x + 2;
    end
    if y < 1
        y = -y + 2;
    end
    if x > 4 * MyWidth
        x = 4 * MyWidth - 7;
    end
    if y > 4 * MyWidth
        y = 4 * MyWidth - 7;
    end
    
    temp1 = img_QuartPixel_interp(MyHeight, x);
    temp2 = img_QuartPixel_interp(MyHeight, y);
    img_low_updated(MyHeight, j) = temp - (temp1 + temp2) / 4;
end

end

