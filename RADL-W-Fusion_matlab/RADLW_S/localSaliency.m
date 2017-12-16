function [ salLocal ] = localSaliency( img )
% ----------------------
% Author: smh
% Date  : 2017.12.04
% Description:
%   This file including the implementation of local saliency detection
%   based on 'context-aware Saliency Detection' 
% ----------------------

% ---------------------------- %
%  Not Good !
% can be replaced by 'GOL' or 'DO A & G'
% se = fspecial('average', [35, 35]);
% salL = imfilter(img, se);
% 
% salG = medfilt2(img, [3, 3]);
% salLocal = abs(salL - salG);
% 
% se = strel('ball', 3, 0);
% salLocal = imclose(salLocal, se);
% ---------------------------- %

% ---------------------------- %
se = fspecial('gaussian', 5, 1);
img = imfilter(img, se);

H = LapFilter(img);
salLocal = GauSaliency(H, 3, 0.5);

% se = strel('ball', 3, 0);
% salLocal = imclose(salLocal, se);

% salLocal = guidedfilter(img, salLocal, 45, 10^-6); 
% ---------------------------- %

% ---------------------------- %
% se = fspecial('average', [35, 35]);
% salL = imfilter(img, se);
% 
% se = fspecial('Gaussian', 5, 1);
% salG = imfilter(img, se);
% 
% salLocal = abs(salG - salL);
% 
% se = strel('ball', 3, 0);
% salLocal = imclose(salLocal, se);

% salLocal = guidedfilter(img, salLocal, 60, 10^-6); 
% ---------------------------- %

% ---------------------------- %
%  Not Good !
% imgT = padarray(img, [5, 5], 'circular', 'both');
% se = fspecial('gaussian', 5, 2);
% [M, N] = size(img);
% distT = zeros(5, 5);
% salT = zeros(M, N);
% 
% for i = 6 : 5 + M
%     for j = 6 : 5 + N
%         valT = imgT(i, j);
%         for m = -2 : 2
%             for n = -2 : 2
%                distT(m+3, n+3) = abs(valT - imgT(i+m, j+n)); 
%             end
%         end
%         salT(i - 5, j - 5) = sum(sum(se .* distT));
%     end
% end

% se = strel('ball', 3, 0);
% salLocal = imclose(salT, se);

% salLocal = salT;
% ---------------------------- %

minVal = min(min(salLocal));
maxVal = max(max(salLocal));
tempVal = maxVal - minVal;
salLocal =  double(salLocal - minVal) / tempVal;


function [ H ] = LapFilter( G )
% Conduct Laplacian filtering on each source image
L = [1 1 1; 1 -8 1; 1 1 1]; % The 3*3 laplacian filter
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


