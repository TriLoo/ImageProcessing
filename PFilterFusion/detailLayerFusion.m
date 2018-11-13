function [ fuseImg ] = detailLayerFusion( detailA, detailB )
% Author: smh
% Date  : 2018.11.13
% Inputs: detailA - detail layer of infrared image
%         detailB - detail layer of visible image

% get the preset map
detailA_abs = abs(detailA);
detailB_abs = abs(detailB);

maskA = detailA_abs > detailB_abs;
pmA = zeros(size(detailA);
pmA(maskA) = 1;

% apply gaussian filtering to pms for obtaining the smooth maps as:
spm = imgaussfilt(pmA, 2);   % sigma = 2
% spmB = imgaussfilt(pmB, 2);

% obtain the initial combined detail layers IFu as following:
ifu = spm .* detailA + (1 - spm) .* detailB;

% do the weighted least square optimization to calculate the final combined
% detail layer
% first step: calculate matrix a
mu = 0.0001;
windowSum = ones(7);   % window size is set to 7 * 7
matrix_a = conv2(detailA, windowSum);
matrix_a = abs(matrix_a) + mu;

% second step: calculate the diagonal matrix A
matrixA = 0;   % TODO


% third step: calculate the final 
gamma = 0.01;
matrixU = diag(1);   % TODO
fuseImg = (ifu + gamma * matrixA * detailB) / (matrixU + gamma * matrixA);   % here, use b/A instead inv(A) to speedup the calculation, prompt by MATLAB

end

