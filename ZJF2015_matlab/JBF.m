function [ res ] = JBF( I, p, rad, deltaS, deltaR)
% ----------------------
% Author : smh
% Date   : 2017.12.18
% Description:
%   This file includes the implementation of Joint Bilateral Filtering
%   (JBF) based on 'Guided image filter' and 'Rolling Guidance Filter'.
%   Inputs: 
%       I : The iuput guided image, single channel
%       p : The input filtering image
%       rad : the radius of neighbour size
%       deltaS : control the spartial similarity
%       deltaR : control the color similarity
%   Outputs:
%       res : the output of rolling guidance filter
% ----------------------

[M, N] = size(p);
res = zeros(M, N);

if nargin < 2
    error('Too few inputs parameters');
end

if ~exist('rad', 'var')
    rad = 2;
end

if ~exist('deltaS', 'var')
    deltaS = 1.2;
end

if ~exist('deltaR', 'var')
    deltaR = 0.25;
end

if isa(I, 'uint8')
    I = im2double(I);
end

if isa(p, 'uint8')
    p = im2double(p);
end

len = 2 * rad + 1;
divS = 2 * deltaS * deltaS;     % The division of spartial similarity.
divR = 2 * deltaR * deltaR;     % The division of color similarity.

[X,Y] = meshgrid(-rad:rad,-rad:rad);
gs = exp(- (X .* X + Y .* Y) / divS);

% pad the input guidance image
imgI = padarray(I, [rad, rad], 'replicate', 'both');
imgP = padarray(p, [rad, rad], 'replicate', 'both');

for i = 1 + rad : M + rad
    for j = 1 + rad : N + rad
       patch1 = imgP(i - rad : i + rad, j - rad : j + rad);
       patch2 = imgI(i - rad : i + rad, j - rad : j + rad);
       
       d = (repmat(imgI(i, j), [len, len]) - patch2).^2;
       gr = exp(-d / divR);
       
       g = gr .* gs;
       Kfactor = sum(sum(g));            % Mij
       res(i - rad, j - rad) = sum(sum(g .* patch1)) / Kfactor;
    end
end

end


