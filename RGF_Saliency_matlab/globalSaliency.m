function [ salGlobal ] = globalSaliency( imgIn )
% ---------------------------
% Author : smh
% Date   : 2017.12.04
% Description:
%   This file including the implementation of global saliency detection
%   based on 'Global Contrast based Saliency Detection' MM. Cheng.
% ---------------------------

imgIn = im2uint8(imgIn);

[m, n, z] = size(imgIn);
if z == 3
    imgIn = rgb2gray(imgIn);
end


imgHist = calHist(imgIn);
imgHist = imgHist/(m * n);
imgDist = colorDist(imgHist);

salGlobal = zeros(m, n);

for i = 1:m
    for j = 1:n
        imgVal = imgIn(i, j) + 1;
        temp = imgDist(imgVal);
        salGlobal(i, j) = temp;        
    end
end

maxVal = max(max(salGlobal));
salGlobal = salGlobal / maxVal;


subplot(1, 2, 1);
imshow(imgIn, []);
title('Input Image');
subplot(1, 2, 2);
imshow(salGlobal, []);
title('Global Saliency');


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

function [ histA ] = calHist( img )
% ----------------------------
% Author : smher
% Date   : 2017.04.26
% Description :
%       This function implement the function of calculating the histogram
%       of input image img.  return 0~1 distrution
% ----------------------------

img = im2uint8(img);

histA = zeros(1, 256);

[m,n] = size(img);

for i=1:m
    for j = 1:n
        temp = img(i,j);
        histA(temp+1) = histA(temp+1) + 1;
    end
end

% For Test
% sum = 0;

% for i = 1:256
%     sum = sum + histA(1,i);
% end

end

end

