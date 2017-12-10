function [ res ] = myGFF( I )
% ------------------
% Author : smh
% Data   : 2017-11-24
% Description:
%   This file implement the RLNSW + Saliency Map based Image fusion.
% ------------------

% r1 = 45; 
% eps1 = 0.3; 
% r2 = 7; 
% eps2 = 10^-6;
% r3 = 10; 
% eps3 = 0.001;

% G = I;

% Saliency map based on 'image fusion with guided filter'
% H = LapFilter(I);
% S = GauSaliency(H);

% Saliency map based on 'global contrast based saliency region detection'
% S(:, :, 1) = hc(I(:, :, 1));
% S(:, :, 2) = hc(I(:, :, 2));

% Saliency map based on 'my local + global saliency detection, versioin 1'
% S(:, :, 1) = localglobal(I(:, :, 1));
% S(:, :, 2) = localglobal(I(:, :, 2));

% Saliency map based on 'frequency-tuned saliency region detection'
% S(:, :, 1) = ftSaliency(I(:, :, 1));
% S(:, :, 2) = ftSaliency(I(:, :, 2));

% Saliency map based on 'my local + global saliency detection, versioin 2':
% HC + LoG


[W_D, W_B] = WeightedMap(I, 'GOL');


% S(:, :, 1) = LocalGlobalSaliency(I(:, :, 1), 'GOL');
% S(:, :, 2) = LocalGlobalSaliency(I(:, :, 2), 'GOL');
% 
% % Initial Weight Construction
% P = IWconstruct(S);
% 
% 
% % Weight Optimization with Guided Filtering
% W_B = GuidOptimize(G,P,r1,eps1);
% W_D = GuidOptimize(G,P,r2,eps2);
% % W_M = GuidOptimize(G, P, r3, eps3);

% Two Scale Decomposition and Fusion
% F = GuidFuse(I,W_B,W_D, W_M);
F = GuidFuse(I, W_B, W_D);

% minF = min(min(F));
% maxF = max(max(F));
% F = 1.0 * (F - minF) / maxF;

% Image Format Transformation
res = im2uint8(F);

% imwrite(res, './results/MarnehNew.jpg');


function [H] = LapFilter(G)

L = [0, 1, 0; 1, -4, 1; 0, 1, 0];

N = size(G,3);
G = double(G)/255;
H = zeros(size(G,1),size(G,2),N); % Assign memory
for i = 1:N
    H(:,:,i) = abs(imfilter(G(:,:,i),L,'replicate'));
end

function [ S ] = GauSaliency( H )
% Using the local average of the absolute value of H to construct the 
% saliency maps
N = size(H,3);
S = zeros(size(H,1),size(H,2),N);
for i=1:N
se = fspecial('gaussian',11,5);
S(:,:,i) = imfilter(H(:,:,i),se,'replicate');
end
S = S + 1e-12; %avoids division by zero
S = S./repmat(sum(S,3),[1 1 N]);%Normalize the saliences in to [0-1]


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

% W_B, W_D, W_M : all are (r * c * 2) 3-dim matrix
% function [ F ] = GuidFuse(I, W_B, W_D, W_M)
function [ F ] = GuidFuse(I, W_B, W_D)

%     I = double(I) / 255;
    
%     rlnswA = rlnsw(I(:, :, 1), 2);   % return 1*3 cell
%     rlnswB = rlnsw(I(:, :, 2), 2);   % return 1 * 3 cell
    
    [r, c, N] = size(I);  % N = 2 for two images input
    
%     F_B = zeros(r, c);
%     F_D = zeros(r, c);
%     F_M = zeros(r, c);
    
%     F_B = rlnswA{1} .* W_B(:, :, 1) + rlnswB{1} .* W_B(:, :, 2);
%     F_D = rlnswA{3} .* W_D(:, :, 1) + rlnswB{3} .* W_D(:, :, 2);
%     F_M = rlnswA{2} .* W_M(:, :, 1) + rlnswB{2} .* W_M(:, :, 2);
    
%     X = {F_B, F_M, F_D};
    
%     F = irlnsw(X, 2);
    se = fspecial('gaussian', 11, 5);
    gaussA_A = imfilter(I(:, :, 1), se);  % below
    gaussA_B = I(:, :, 1) - gaussA_A;     % high
    gaussB_A = imfilter(I(:, :, 2), se);  % below
    gaussB_B = I(:, :, 2) - gaussB_A;     % high
    
    F_B = W_B(:, :, 1) .* gaussA_A + W_B(:, :, 2) .* gaussB_A;
    F_D = W_D(:, :, 1) .* gaussA_B + W_B(:, :, 2) .* gaussB_B;
%     F_D = (gaussA_B + gaussB_B) / 2;
%     F_B = (gaussA_A + gaussB_A) / 2;
    
    F = F_B + F_D;
    


% end
