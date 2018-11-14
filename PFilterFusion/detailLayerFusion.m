function [ fuseImg ] = detailLayerFusion( detailA, detailB )
% Author: smh
% Date  : 2018.11.13
% Inputs: detailA - detail layer of infrared image
%         detailB - detail layer of visible image

% get the preset map
detailA_abs = abs(detailA);
detailB_abs = abs(detailB);

maskA = detailA_abs > detailB_abs;
pmA = zeros(size(detailA));
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
matrix_a = 1 ./ abs(matrix_a) + mu;

% second step: calculate the diagonal matrix A
matrix_a = matrix_a(:);
[m, n] = size(detailA);
matrix_a = sparse(1:1:m*n, 1:1:m*n, matrix_a(1:1:m*n)', m*n, m*n);

temp = sparse(1:1:m*n, 1:1:m*n, ones(1, m*n), m*n, m*n);
gamma = 0.01;

denominator = 2 * temp + gamma * (matrix_a + matrix_a');
numerator = 2 * ifu(:) + gamma * (matrix_a + matrix_a') * detailB(:);

% third step: calculate the final 
fuseImg = denominator \ numerator;   % here, use b/A instead inv(A) to speedup the calculation, prompt by MATLAB
fuseImg = reshape(fuseImg, m, n);

end

