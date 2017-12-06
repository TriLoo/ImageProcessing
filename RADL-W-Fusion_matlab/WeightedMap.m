function [ W_D, W_B ] = WeightedMap( I, name )
% -----------------------
% Author : smh
% Date   : 2017.12.06
% Description:
%       This file including the implementation of 'generation of weighted
%       map' used in R-ADL-Wavelet.
%   Inputs:
%       I: A M * N * 2 matrix, stroing the input images
%       name:
%           'GOL': local saliency use Gaussian of Laplacian.
%           'CA' : local saliency use 'context-aware' saliency map.
%   Outputs:
%       W_D: A M * N * 2 matrix including the result detailed saliency map.
%       W_B: A M * N * 2 matrix including the result base saliency map.
% -----------------------

S(:, :, 1) = LocalGlobalSaliency(I(:, :, 1), name);
S(:, :, 2) = LocalGlobalSaliency(I(:, :, 2), name);


% Initial Weight Construction
P = IWconstruct(S);

% Weight Optimization with Guided Filtering
r1 = 45;
eps1 = 0.3;
r2 = 7;
eps2 = 10^-6;
W_B = GuidOptimize(I, P, r1, eps1);
W_D = GuidOptimize(I, P, r2, eps2);



function [P] = IWconstruct( S )
% construct the initial weight maps
[r c N] = size(S);
[X Labels] = max(S,[],3); % find the labels of the maximum
clear X
for i = 1:N
    mono = zeros(r,c);
    mono(Labels==i) = 1;
    P(:,:,i) = mono;
end
end

function [W] = GuidOptimize( I, P, r, eps)
N = size(I,3);
I = double(I)/255;
for i=1:N
P(:,:,i) = double(P(:,:,i));
W(:,:,i) = guidedfilter(I(:,:,i), P(:,:,i), r, eps);
end
W = uint8(W.*255); % Remove values which are not in the [0-1] range
W = double(W)/255;
W = W + 1e-12; %Normalization
W = W./repmat(sum(W,3),[1 1 N]);
end

end

