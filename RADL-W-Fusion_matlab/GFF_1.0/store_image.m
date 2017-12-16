clc;

% A = imread('./datas/source20_1.tif');
% B = imread('./datas/source20_2.tif');

% A = imread('./datas/Jeep_IR.bmp');
% B = imread('./datas/Jeep_Vis.bmp');

% A = imread('./datas/Marne_24_IR.bmp');
% B = imread('./datas/Marne_24_Vis.bmp');

A = imread('./datas/LWIR-MarnehNew_15RGB_603.tif');
B = imread('./datas/NIR-MarnehNew_15RGB_603.tif');

% A = imread('./datas/7422i.bmp');
% B = imread('./datas/7422v.bmp');source20


[m, n, z] = size(A);
fprintf('Input image A size is: %d * %d * %d. \n', m, n, z);
[m, n, z] = size(B);
fprintf('Input image B size is: %d * %d * %d. \n', m, n, z);

I = zeros(m, n, 2);
I(:, :, 1) = A;
I(:, :, 2) = B;

tic;
F = GFF(A, B);
toc;

minF = min(min(F));
maxF = max(max(F));

minF = double(minF);
maxF = double(maxF);

F = 255.0 * (double(F) - minF) / maxF;
imwrite(uint8(F), './results/MarnehNew.jpg');

