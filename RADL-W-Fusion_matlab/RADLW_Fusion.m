function [ imgRes ] = RADLW_Fusion( imgA, imgB )
% ------------------------------
% Author: smh
% Date  : 2017.12.06
% Description:
%   This file is the top function of image fusion basing on R-ADL-Wavelet
%   transform & saliency detection.
% ------------------------------

% convert to double
imgA = im2double(imgA);
imgB = im2double(imgB);

% decompose the images
[cA_A, cV_A, cH_A, cD_A, PB_A] = RADLWavelet(imgA);
[cA_B, cV_B, cH_B, cD_B, PB_B] = RADLWavelet(imgB);

% Generate Weighted Map
I(:, :, 1) = imgA;
I(:, :, 2) = imgB;
[W_D, W_B] = WeightedMap(I, 'GOL');

% Fusion basing on Weighted Maps & decomposed Results.
% cA = (cA_A + cA_B) / 2;
% cV = (cV_A + cV_B) / 2;
% cH = (cH_A + cH_B) / 2;
% cD = (cD_A + cD_B) / 2;

% cA = W_B(:, :, 1) .* cA_A + W_B(:, :, 2) .* cA_B;
% cV = W_D(:, :, 1) .* cV_A + W_D(:, :, 2) .* cV_B;
% cH = W_D(:, :, 1) .* cH_A + W_D(:, :, 2) .* cH_B;
% cD = W_D(:, :, 1) .* cD_A + W_D(:, :, 2) .* cD_B;

lambda = 0.001;
cA = W_B(:, :, 1) .* cA_A + W_B(:, :, 2) .* cA_B;
cV_t = W_D(:, :, 1) .* cV_A + W_D(:, :, 2) .* cV_B;
cV = Solve_Optimal(cV_t, cV_A, cV_B, lambda);
cH_t = W_D(:, :, 1) .* cH_A + W_D(:, :, 2) .* cH_B;
cH = Solve_Optimal(cH_t, cH_A, cH_B, lambda);
cD_t = W_D(:, :, 1) .* cD_A + W_D(:, :, 2) .* cD_B;
cD = Solve_Optimal(cD_t, cD_A, cD_B, lambda);



% Inverse RADLWavelet transform to get the fusion result.
% Generate the final Prediction Block
PB = generatePB(PB_A, PB_B);
% Apply inverse RADLWavelet based on generated Prediction block
imgRes = inverse_RADLWavelet(cA, cV, cH, cD, PB);

end

