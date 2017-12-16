function [ originNew ] = my_Horizontal_Update(originPre, origin, ori_QuartPixel_interp, Predict_direc )
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
M = 8;
N = 8;

originNew = ones(2 * MyHeight, MyWidth);
originNew(1:MyHeight, 1:MyWidth) = origin;
originNew(MyHeight + 1 : end, : ) = originPre;

% The first line
for j = 1 : MyWidth
    k = Predict_direc(1, ceil(j / N));
%     temp = ori_QuarPixel_interp(1, 4 * j - 3);
    temp = origin(1, j);
    
    x = 4 * j + k - 3;
    y = 4 * j - k - 3;
    
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
    
    temp1 = ori_QuartPixel_interp(2, x);
    temp2 = ori_QuartPixel_interp(2, y);
    
    originNew(1, j) = temp + (temp1 + temp2) / 4;
end

% the last MyHeight - 2  lines
for j = 1 : MyWidth
    for i = 2 : 1 : MyHeight - 1
        k = Predict_direc(ceil(i / M), ceil(j / N));
        temp = origin(i, j);
        
        x = 4 * j - 3 + k;
        y = 4 * j - 3 - k;
        
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
        
        temp1 = ori_QuartPixel_interp(i - 1, x);
        temp2 = ori_QuartPixel_interp(i + 1, y);
        
        originNew(i, j) = temp + (temp1 + temp2) / 4;     
    end
end

% The last line
for j = 1 : MyWidth
    k = Predict_direc(ceil(MyHeight / M), ceil( j / N));
    temp = origin(MyHeight, j);
    
    x = 4 * j - 3 + k;
    y = 4 * j - 3 - k;
    
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
    
    temp1 = ori_QuartPixel_interp(MyHeight, x);
    temp2 = ori_QuartPixel_interp(MyHeight, y);
    
    originNew(MyHeight, j) = temp + (temp1 + temp2) / 4;
end

end
