function [ res ] = RGF( img, rad, deltaS, deltaR)
% -------------------------
% Author : smh
% Date   : 2017.12.18
% Description:
%   This file including the implementation of 'rolling guided filter'
%   based on 'Rolling Guidance Filter' ---- changed
%   Inputs:
%       img: input image, * 1
%       res: the output of rolling guidance filter
%       deltaS : the JBF input 
%       deltaR : the JBF input
%       level  : the level of RGF (iteration time)\
%   Outputs:
%       res : the result of RGF
% -------------------------

% Small structure removal
% se = fspecial('gaussian', 5, 1);
% L = imfilter(img, se);

[M, N] = size(img);
% Edge recovery
% lve ...

if nargin == 1
    rad = 7;
    deltaS = 12;
    deltaR = [0.02, 0.08, 0.15];
end

if ~exist('deltaS', 'var')
    deltaS = 12;
    deltaR = [0.02, 0.08, 0.15];
end

level = length(deltaR);
temp = zeros(M, N, level+1);
res = zeros(M, N, level+1);
% C = ones(size(img));
for i = 1 : level
    if i == 1      % Small structure removal
        temp(:, :, i) = JBF(img, img, rad, deltaS, deltaR(i));
    else           % Edge Recovery
        temp(:, :, i) = JBF(img, img, rad, deltaS, deltaR(i));  % Use a blurry image to guide filtering
    end
end

for i = 1 : level
    if i == 1
        res(:, :, i) = img - temp(:, :, i);
    else
        res(:, :, i) = temp(:, :, i-1) - temp(:, :, i);
    end 
end

res(:, :, level+1) = temp(:, :, level);

end

