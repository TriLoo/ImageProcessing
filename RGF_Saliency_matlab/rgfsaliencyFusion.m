function [ res ] = rgfsaliencyFusion( imgA, imgB, rad, sig, eps )
% --------------------------
% Author : smh
% Date   : 2017.12.23
% Description:
%   This file includes the implementation of 'image fusion based on rolling
%   guidance filter' and 'saliency detection'.
% --------------------------

if nargin < 2
    error('Not enough images input ...');
end

imgA = im2double(imgA);
imgB = im2double(imgB);

eleA = rollingguidancefilter(imgA);
eleB = rollingguidancefilter(imgB);

len = length(eleA);

[W_D, W_B] = WeightedMap(imgA, imgB, 'GOL');
tempLayer = cell(len, 1);

% for i = 1 : len
%     if i == 1
%         tempLayer{i} = W_B(:, :, 1) .* eleA{i} + W_B(:, :, 2) .* eleB{i};
%     else
%         tempLayer{i} = W_D(:, :, 1) .* eleA{i} + W_D(:, :, 2) .* eleB{i};
%     end
% end

for i = 1 : len
    tempLayer{i} = (eleA{i} + eleB{i}) / 2;
end

for i = 1 : len
    if i == 1
        res = tempLayer{i};
    else
        res = res + tempLayer{i};
    end
end

% Alpha = [1.5, 1.1, 0.6, 0.3];
% for i = 1 : len
%     if i == 1
%         res = tempLayer{i} * Alpha(i);
%     end
%     res = res + tempLayer{i} * Alpha(i);
% end

end

