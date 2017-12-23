function [ res ] = rollingguidancefilter( img, rad, sig, eps, level)
% -------------------------
% Author : smh
% Date   : 2017.12.23
% Description:
%   This file incluldes the implementation of rolling guidance filter based
%   on 'guided image filter'.
% -------------------------

if nargin < 1
    error('Too few inputs: no image input.');
end

if ~exist('rad', 'var')
    rad = [5, 5, 5];
    sig = [1, 1, 1];
    eps = [0.04, 0.02, 0.01];
    level = 3;
end

if length(rad) ~= level
    level = length(rad);
end

if size(img, 3) == 3
    img = rgb2gray(img);
end

if isa(img, 'uint8')
    img = im2double(img);
end

C = ones(size(img));
res = cell(level + 1, 1);

% tempImg = zeros(size(img));
se = fspecial('gaussian', 11, 5);

for i = 1 : level
    if i == 1
        res{i} = imfilter(img, se);                                  % Get the base layers
%         res{i} = guidedfilter(C, img, rad(i), sig(i), eps(i));                         
    else
        res{i} = guidedfilter(res{i - 1}, img, rad(i), sig(i), eps(i));    % Get the detail layers
    end
end

% len = length(res);

for i = level : -1 : 1
    if i == 1
        res{i+1} = img - res{i};
    else
        res{i + 1} = res{i-1} - res{i};
    end
end

end
