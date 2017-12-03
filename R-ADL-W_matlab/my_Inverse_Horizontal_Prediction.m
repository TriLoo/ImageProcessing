function [ originNew ] = my_Inverse_Horizontal_Prediction( origin, ori_QuartPixel_interp, Predict_direc )
% ------------------------
% Author : smher
% Data   : 2017.12.02
% Description:
%   This file implementation the invert transform of
%   my_Horizontal_Prediction, used by redundant directional lifting wavelet
%   transform.
% Inputs: 
%       origin: low-pass part
%       ori_QuartPixel_interp: high-pass part after Sinc interpolation.
%       Predict_direc: the prediction block
% Outpus:
%       The inverse updated low-passed h
% ------------------------

[MyHeight, MyWidth] = size(origin);

M = 8; N = 8;

%  front MyHeight - 1 lines
for j = 1 : MyWidth
    for i = 2 : 1 : MyHeight
        k = Predict_direc(ceil(i / M), ceil(j / N));
%         temp = ori_QuartPixel_interp(i, 4 * j - 3);
        
        x = 4 * j - 3 + k;
        y = 4 * j - 3 - k;
        
        if x < 1
            x = -x + 2;
        end
        if y < 1
            y =  -y + 2;
        end
        if x > 4 * MyWidth
            x = 4 * MyWidth - 7;
        end
        if y > 4 * MyWidth
            y = 4 * MyWidth - 7;
        end
        
%         temp1 = 
        
        
    end
end

% the MyHeight-th line





end

