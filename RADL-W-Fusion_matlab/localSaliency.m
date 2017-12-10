function [ salLocal ] = localSaliency( img )
% ----------------------
% Author: smh
% Date  : 2017.12.04
% Description:
%   This file including the implementation of local saliency detection
%   based on 'context-aware Saliency Detection' 
% ----------------------


H = LapFilter(img);
salLocal = GauSaliency(H, 3, 0.2);


% img = im2uint8(img);

% img = GauSaliency(img, 5, 0.5);
% img = GauSaliency(img, 5, 2);
% salLocal = LapFilter(img);
% salLocal = GauSaliency(H);
% H = GauSaliency(img, 5, 0.5);
% H = abs(imgL - imgH);
% H = LapFilter(H);
% salLocal = GauSaliency(H, 5, 2);

se = strel('disk', 1);
salLocal = imclose(salLocal, se);


maxVal = max(max(salLocal));

salLocal =  double(salLocal) / maxVal;


function [ H ] = LapFilter( G )
% Conduct Laplacian filtering on each source image
L = [1 1 1; 1 -8.5 1; 1 1 1]; % The 3*3 laplacian filter
% L = [1 2 1; 2 -12 2; 1 2 1]; % The 3*3 laplacian filter
% L = [0, 1, 0; 1, -4, 1; 0, 1, 0];
N = size(G,3);
% G = double(G)/255;
H = zeros(size(G,1),size(G,2),N); % Assign memory
for i = 1:N
    H(:,:,i) = abs(imfilter(G(:,:,i),L,'replicate'));
end
end

function [ S ] = GauSaliency( H, rad,  sigma)
% Using the local average of the absolute value of H to construct the 
% saliency maps
N = size(H,3);
S = zeros(size(H,1),size(H,2),N);
for i=1:N
se = fspecial('gaussian', rad, sigma);
S(:,:,i) = imfilter(H(:,:,i),se,'replicate');
end
end

end


