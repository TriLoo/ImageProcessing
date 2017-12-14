clc;
close all;
clear;

% imgA = imread('../datas/Reek_IR.bmp');
% imgB = imread('../datas/Reek_Vis.bmp');

imgA = imread('../datas/source20_1.tif');
imgB = imread('../datas/source20_2.tif');
% 
% imgA = imread('../datas/LWIR-MarnehNew_15RGB_603.tif');
% imgB = imread('../datas/NIR-MarnehNew_15RGB_603.tif');

imgA = double(imgA);
imgB = double(imgB);

I(:, :, 1) = imgA;
I(:, :, 2) = imgB;

salA_local = localSaliency(imgA);
salA_global = globalSaliency(imgA);
salA_gol = LocalGlobalSaliency(imgA, 'GOL');
salA_ca = LocalGlobalSaliency(imgA, 'CA');

salB_local = localSaliency(imgB);
salB_global = globalSaliency(imgB);
salB_gol = LocalGlobalSaliency(imgB, 'GOL');
salB_ca = LocalGlobalSaliency(imgB, 'CA');

[WM_gol_D, WM_gol_B] = WeightedMap(I, 'GOL');
WM_ca = WeightedMap(I, 'CA');

subplot(2, 6, 1);
imshow(imgA, []);
title('Input A');

subplot(2, 6, 2);
imshow(salA_local, []);
title('salA local A');

subplot(2, 6, 3);
imshow(salA_global, []);
title('salA global A');

subplot(2, 6, 4);
imshow(salA_gol, []);
title('salA gol A');

subplot(2, 6, 5);
imshow(salA_ca, []);
title('salA ca A');

subplot(2, 6, 6);
imshow(WM_gol_B(:, :, 1), []);
title('WM gol A');

subplot(2, 6, 7);
imshow(imgB, []);
title('Input B');

subplot(2, 6, 8);
imshow(salB_local, []);
title('salB local B');

subplot(2, 6, 9);
imshow(salB_global, []);
title('salB global B');

subplot(2, 6, 10);
imshow(salB_gol, []);
title('salB gol B');

subplot(2, 6, 11);
imshow(salB_ca, []);
title('salB ca B');

subplot(2, 6, 12);
imshow(WM_gol_B(:, :, 2), []);
title('WM gol B');
