function [ res ] = JBFcolor( I, p, rad, deltaS, deltaR )
% --------------------------
% Author : smh
% Date   : 2017.12.18
% Description:
%   This file includes the implementation of Bilateral filter.
%   Inputs:
%       I : the input color guidance image
%       p : the input color fitering image
%       rad : the radius of neighbour, i.e. [2 * rad + 1, 2 * rad + 1]
%       neighbour.
%       deltaS :  spartial similarity
%       deltaR :  range similarity
% --------------------------

if size(I, 3) ~= 3
    error('Input image is not a color img.');
end

if size(p, 3) ~= 3
    error('Input filtering image is not a color img.');
end

if nargin < 2
    error('Too few input parameters');
end

if isa(I, 'uint8')
    I = im2double(I);
end

if isa(p, 'uint8')
    p = im2double(p);
end

if ~exist('rad', 'var')
    rad = 2;     % i.e. [5, 5] region
end

if ~exist('deltaS', 'var')
    deltaS = 1.2;
end

if ~exist('deltaR', 'var')
    deltaR = 0.25;
end

len = 2 * rad + 1;
divS = 2 * deltaS * deltaS;
divR = 2 * deltaR * deltaR;
M = size(I, 1);
N = size(I, 2);
res = zeros(size(I));

imgI = padarray(I, [rad, rad], 'replicate', 'both');
imgP = padarray(p, [rad, rad], 'replicate', 'both');

[X, Y] = meshgrid(-rad:rad, -rad:rad);

gs = exp(- (X .* X + Y .* Y) / divS);

for i = 1 + rad : M + rad
    for j = 1 + rad : N + rad
       patch1 = imgP(i - rad : i + rad, j - rad : j + rad, :);
       patch2 = imgI(i - rad : i + rad, j - rad : j + rad, :);
       
       d = (repmat(imgI(i, j, :), [len, len]) - patch2).^2;
       gr = exp(-d / divR);
       
       for k = 1 : size(I, 3)
           g(:, :, k) = gr(:, :, k) .* gs;
           Kfactor = sum(sum(g(:, :, k)));
           res(i - rad, j - rad, k) = sum(sum(g(:, :, k) .* patch1(:, :, k))) / Kfactor;
       end
    end
end

end

