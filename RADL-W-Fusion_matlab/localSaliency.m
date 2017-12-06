function [ salLocal ] = localSaliency( img )
% ----------------------
% Author: smh
% Date  : 2017.12.04
% Description:
%   This file including the implementation of local saliency detection
%   based on 'context-aware Saliency Detection' 
% ----------------------

img = im2uint8(img);

H = LapFilter(img);
salLocal = GauSaliency(H);

maxVal = max(max(salLocal));

salLocal =  salLocal / maxVal;


function [ H ] = LapFilter( G )
% Conduct Laplacian filtering on each source image
L = [0 1 0; 1 -4 1; 0 1 0]; % The 3*3 laplacian filter
N = size(G,3);
G = double(G)/255;
H = zeros(size(G,1),size(G,2),N); % Assign memory
for i = 1:N
    H(:,:,i) = abs(imfilter(G(:,:,i),L,'replicate'));
end
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
end

end


