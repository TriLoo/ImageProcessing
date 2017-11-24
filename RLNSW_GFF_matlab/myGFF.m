function [ res ] = myGFF( I )
% ------------------
% Author : smh
% Data   : 2017-11-24
% Description:
%   This file implement the RLNSW + Saliency Map based Image fusion.
% ------------------

r1 = 45; 
eps1 = 0.3; 
r2 = 7; 
eps2 = 10^-6;
r3 = 20; 
eps3 = 0.01;

G = I;

tic;
H = LapFilter(I);

S = GauSaliency(H);

% Initial Weight Construction
P = IWconstruct(S);


% Weight Optimization with Guided Filtering
W_B = GuidOptimize(G,P,r1,eps1);
W_D = GuidOptimize(G,P,r2,eps2);
W_M = GuidOptimize(G, P, r3, eps3);

% Two Scale Decomposition and Fusion
F = GuidFuse(I,W_B,W_D, W_M);

minF = min(min(F));
maxF = max(max(F));
F = 1.0 * (F - minF) / maxF;

toc;

% Image Format Transformation
res = uint8(F*255);

imwrite(res, './results/MarnehNew.jpg');


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
function [ F ] = GuidFuse(I, W_B, W_D, W_M)
    I = double(I) / 255;
    
    rlnswA = rlnsw(I(:, :, 1), 2);   % return 1*3 cell
    rlnswB = rlnsw(I(:, :, 2), 2);   % return 1 * 3 cell
    
    [r, c, N] = size(I);  % = 2 for two images input
    
    F_B = zeros(r, c);
    F_D = zeros(r, c);
    F_M = zeros(r, c);
    
    F_B = rlnswA{1} .* W_B(:, :, 1) + rlnswB{1} .* W_B(:, :, 2);
    F_D = rlnswA{3} .* W_D(:, :, 1) + rlnswB{3} .* W_D(:, :, 2);
    F_M = rlnswA{2} .* W_M(:, :, 1) + rlnswB{2} .* W_M(:, :, 2);
    
    X = {F_B, F_M, F_D};
    
    F = irlnsw(X, 2);
    


% end