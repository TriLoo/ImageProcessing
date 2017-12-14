function [ saliency ] = gff_log( img )
% --------------
% Author: smher
% Data  : 2017.11.27
% Description:
%       This file include the saliency detection in 'Image fusion with
%       Guided Filter': 
%           Step 1 : absolute value of laplacian filtering input image;
%           Step 2 : Gaussian filtering the output of step 1.
% --------------

H = LapFilter(img);
saliency = GauSaliency(H);


function [ H ] = LapFilter( G )
%Conduct Laplacian filtering on each source image
L = [0 1 0; 1 -4 1; 0 1 0]; % The 3*3 laplacian filter
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
% S = S + 1e-12; %avoids division by zero
% S = S./repmat(sum(S,3),[1 1 N]);%Normalize the saliences in to [0-1]

% end

