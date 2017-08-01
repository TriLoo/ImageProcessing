function [ imDst ] = boxFilter( imgIn, r )
% -------------------
% Author : smher
% Data   : 2017. 07. 31
% Description :
%       This file implement the box filter 
% -------------------

[m, n] = size(imgIn);

imDst = zeros(m, n);

% Calculate sum over Y Axis
imCum = cumsum(imgIn, 1);

% Difference over Y axis
imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
imDst(r+2:m-r, :) = imCum(2*r+2 : m, :) - imCum(1 : m-2*r-1, :);
imDst(m-r+1 : m, :) = repmat(imCum(m, :), [r, 1]) - imCum(m-2*r : m-r-1, :);

% Calculate sum over X Axis
imCum = cumsum(imDst, 2);
% Difference over X axis
imDst(:, 1:r+1) = imCum(:, 1+r : r*2+1);
imDst(:, r+2:n-r) = imCum(:, 2*r+2 : n) - imCum(:, 1 : n-2*r-1);
imDst(:, n-r+1 : n) = repmat(imCum(:, n), [1, r]) - imCum(:, n-2*r : n-r-1);



end

