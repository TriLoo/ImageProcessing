function [ res ] = RGF( img, rad, deltaS, deltaR)
% -------------------------
% Author : smh
% Date   : 2017.12.18
% Description:
%   This file including the implementation of 'rolling guided filter'
%   based on 'Rolling Guidance Filter'
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

if ~exist('deltaS', 'var')
    deltaS = [4, 4, 4];
    deltaR = 0.1;
end

level = length(deltaS);
res = zeros(M, N, level+1);
C = ones(size(img));
for i = 1 : level + 1
    if i == 1      % Small structure removal
        res(:, :, i) = JBF(C, img, rad, deltaS(i), deltaR);
    else           % Edge Recovery
        res(:, :, i) = JBF(res(:, :, i-1), img, rad, deltaS(i-1), deltaR);  % Use a blurry image to guide filtering
    end
end

end

