function [ salCA ] = caSaliency( img, rad, c )
% ----------------------------
% Author : smh
% Date   : 2017.12.06
% Description:
%   This file including the implementation of Local Saliency Detection
%   basing on 'Context-Aware Saliency Detection': 'Local-Global
%   Single-Scale Saliency'.
% ----------------------------

[M, N] = size(img);

img = im2double(img);

salCA = zeros(M, N);

if exist('c', 'var') == 0
    c = 3;
end

% extend the image to zero boundary: M * N ==> (M + 2 * rad) * (N + 2 *
% rad) work!
% imgE = zeros(M + 2 * rad, N + 2 * rad);
% imgE(rad+1:rad + M, rad+1 : rad + N) = img(1:end, 1:end);

% Another way to pad the matrix. imgE = img extented.
imgE = padarray(img, [rad, rad]) ;

% imshow(imgE, []);   % for test

d = 1;

% tempSum = 0;
Divd = (2 * rad + 1)^2;

for i = rad + 1 : rad + M
    for j = rad + 1 : rad + N
        tempSum = 0;
        for m = -rad : rad
            for n = -rad : rad
                if (m ~= 0) & (n ~= 0)
                    Dcolor = abs(imgE(i + m, j + n) - imgE(i, j));
                    Dpos = sqrt(m * m + n * n);
                    tempSum = tempSum + (Dcolor / (1 + c * Dpos));
                end
            end
        end
        salCA(i - rad, j - rad) = 1 - exp(-tempSum / Divd);
    end
end

salCA = guidedfilter(img, salCA, 45, 10^-6);

maxVal = max(max(salCA));
salCA = salCA / maxVal;

subplot(1, 2, 1);
imshow(uint8(img * 225), []);
title('Input Image');
subplot(1, 2, 2);
imshow(salCA, []);
title('Saliency Map');

end

