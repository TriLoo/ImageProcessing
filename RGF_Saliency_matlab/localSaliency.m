function [ salLocal ] = localSaliency( img )
% ----------------------
% Author: smh
% Date  : 2017.12.04
% Description:
%   This file including the implementation of local saliency detection
%   based on 'context-aware Saliency Detection' 
% ----------------------
% ---------------------------- %
% G = GauSaliency(img);
% H = LapFilter(H);
img = GauSaliency(img);
H = LapFilter(img);
salLocal = GauSaliency(H);

subplot(2, 3, 2);
imshow(salLocal, []);
title('Local Saliency A-No');

se = strel('ball', 3, 0);
salLocal = imclose(salLocal, se);
% ---------------------------- %

% ---------------------------- %
% se = fspecial('average', [35, 35]);
% salL = imfilter(img, se);
% 
% salG = medfilt2(img, [3, 3]);
% salLocal = abs(salL - salG);
% 
% se = fspecial('Gaussian', 5, 1);
% salLocal = imfilter(salLocal, se);
% ---------------------------- %

% ---------------------------- %
% img = double(img);
% se = fspecial('average', [35, 35]);
% salL = imfilter(img, se);
% 
% se = fspecial('Gaussian', 5, 0.8);
% salG = imfilter(img, se);
% 
% salLocal = abs(salG - salL);
% salLocal = guidedfilter(img, salLocal, 15, 10^-6); 
% ---------------------------- %

% ---------------------------- %
imgT = padarray(img, [5, 5], 'circular', 'both');
se = fspecial('gaussian', 5, 2);
[M, N] = size(img);
distT = zeros(5, 5);
salT = zeros(M, N);

for i = 6 : 5 + M
    for j = 6 : 5 + N
        valT = imgT(i, j);
        for m = -2 : 2
            for n = -2 : 2
               distT(m+3, n+3) = abs(valT - imgT(i+m, j+n)); 
            end
        end
        salT(i - 5, j - 5) = sum(sum(se .* distT));
    end
end

subplot(2, 3, 3);
imshow(salT, []);
title('Local Saliency B-No');

se = strel('ball', 3, 0);
salLocalB = imclose(salT, se);

% salLocalB = salT;

maxVal = max(max(salLocalB));
salLocalB =  salLocalB / maxVal;
% ---------------------------- %

maxVal = max(max(salLocal));
salLocal =  salLocal / maxVal;

% subplot(1, 2, 1);
% imshow(img, []);
% title('Input Image');
% subplot(1, 2, 2);
% imshow(salLocal, []);
% title('Saliency Map');

subplot(2, 3, 1);
imshow(img, []);
title('Input image A');

subplot(2, 3, 5);
imshow(salLocal, []);
title('Local saliency GOL');

subplot(2, 3, 6);
imshow(salLocalB, []);
title('Local Saliency B');



function [ H ] = LapFilter( G )
% Conduct Laplacian filtering on each source image
% L = [0 1 0; 1 -4 1; 0 1 0]; % The 3*3 laplacian filter
L = [1 1 1; 1 -8 1; 1 1 1]; % The 3*3 laplacian filter
% L = [1, 4, 1; 4, -20, 4; 1, 4, 1];
N = size(G,3);
% G = double(G)/255;
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
se = fspecial('gaussian',5, 0.8);
S(:,:,i) = imfilter(H(:,:,i),se,'replicate');
end
end

end


