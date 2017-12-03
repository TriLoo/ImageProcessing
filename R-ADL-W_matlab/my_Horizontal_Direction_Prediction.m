function [ originNew, Predict_direc ] = my_Horizontal_Direction_Prediction( origin, ori_QuartPixel_interp )
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
Dir = 4;
M=8;
N=8;
% Predict_direc=zeros(MyHeight/M,MyWidth/N);     % one direction per 8 * 8 block
Predict_direc = zeros(ceil(MyHeight / M), ceil(MyWidth / N));
% Block_energy=100000*ones(MyHeight/M,MyWidth/N);
Block_energy=100000*ones(ceil(MyHeight / M), ceil(MyWidth / N));

originNew = origin;

% To get the optimal direction per 8 * 8 (M * N) pixel block.
% Stored in 'Predict_direc'.
for k=-Dir:Dir             % Total 9 directions
    %     前MyHeight-1行
    for j=1:MyWidth
        for i=2:2:MyHeight-1
%           for i=2:1:MyHeight-1
            temp=ori_QuartPixel_interp(i,4*j-3);
            x=4*j-3+k;
            y=4*j-3-k;
            if x<1
                x=-x+2;
            end
            if y<1
                y=-y+2;
            end
            if x>4*MyWidth
                x=4*MyWidth-7;
            end
            if y>4*MyWidth
                y=4*MyWidth-7;
            end
            temp1=ori_QuartPixel_interp(i-1,x);
            temp2=ori_QuartPixel_interp(i+1,y);
            ori_QuartPixel_interp(i,4*j-3)=temp-(temp1+temp2)/2;
        end
    end
     %     MyHeight行
    for j=1:MyWidth
        temp=ori_QuartPixel_interp(MyHeight,4*j);
        x=4*j-3+k;
        y=4*j-3-k;
        if x<1
            x=-x+2;
        end
        if y<1
            y=-y+2;
        end
        if x>4*MyWidth
            x=4*MyWidth-7;
        end
        if y>4*MyWidth
            y=4*MyWidth-7;
        end
        temp1=ori_QuartPixel_interp(MyHeight-1,x);
        temp2=ori_QuartPixel_interp(MyHeight-1,y);
        ori_QuartPixel_interp(MyHeight,4*j-3)=temp-(temp1+temp2)/2;
    end
    %     计算某个方向下块内预测残差最小能量    
    for i=1:MyHeight/M
        for j=1:MyWidth/N   % update all prediction/energy blocks, i.e. 64 here
            minEnergy_dir=0;
            for t1=2:2:M         % calculate 32 pair points to get the final energy per block, sampled by 2, because, every two lines to get one energy of one direction
                for t2=1:N
                    minEnergy_dir = minEnergy_dir + abs(ori_QuartPixel_interp((i-1)*M+t1,(j-1)*N*4+t2*4-3));    % i : per block
                end
            end 
            if minEnergy_dir < Block_energy(i,j)
                Block_energy(i,j)=minEnergy_dir;
                Predict_direc(i,j)=k;
           end
       end
    end
    %     把原始像素值写回插值后的数组，继续预测
    for i=1:MyHeight 
        for j=1:MyWidth 
        ori_QuartPixel_interp(i,j*4-3) = origin(i,j);
        end
    end
end

% Second prediction based on 'Predict_direc' block, calculated above.
% front MyHeight - 1 lines
for j = 1 : MyWidth
    for i = 2 : 1 : MyHeight - 1
        k = Predict_direc(ceil(i / M), ceil(j / N));
        
        temp = ori_QuartPixel_interp(i, 4 * j - 3); % or 
%         temp = origin(i, j);
        
        x = 4 * j - 3 + k; % the column number of upper line
        y = 4 * j - 3 - k; % the column number of lower line
        
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
        
        originNew(i, j) = temp - (temp1 + temp2) / 2; 
    end
end

% the MyHeight line prediction
for j = 1 : MyWidth
    k = Predict_direc(ceil(MyHeight / M), ceil(j / N));
    temp = origin(MyHeight, j);
%         temp = ori_QuartPixel_interp(MyHeight, 4 * j - 3);

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

    temp1 = ori_QuartPixel_interp(MyHeight - 1, x);
    temp2 = ori_QuartPixel_interp(MyHeight - 1, y);

    originNew(MyHeight, j) = temp - (temp1 + temp2) / 2;
end

originNew(1, :) = originNew(2, :);
    
end

