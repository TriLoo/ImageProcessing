function [ res, ent ] = RGFFusion( imgA, imgB)
% ----------------------------
% Author : smh
% Date   : 2018.01.05
% Description: 
%   This file implements the fusion of all layers based on ZJF2015.
% ----------------------------

[M, N] = size(imgA);
% res = zeros(M, N);

layerA = RGF(imgA);
layerB = RGF(imgB);

wmA = localSal(imgA);
wmB = localSal(imgB);

len = size(layerA, 3);
temp = zeros(M, N, len);

for i = 1 : len
    temp(:, :, i) = (layerA(:, :, i) .* wmA + layerB(:, :, i) .* (1 - wmA) + layerA(:, :, i) .* (1 - wmB) + layerB(:, :, i) .* wmB) / 2; 
end

weights = [1.1, 0.6, 0.3, 1.5];
res = inverseRGF(temp, weights);

resMin = min(min(res));
resMax = max(max(res));

res = (res - resMin) / (resMax - resMin);

ent = entropy(im2uint8(res));

end

