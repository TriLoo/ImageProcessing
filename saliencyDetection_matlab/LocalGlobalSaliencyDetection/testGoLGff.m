clc;
close all;
imgA = imread('../datas/source20_1.tif');

imgA = double(imgA);

salA_local = localSaliency(imgA);
salA_gff = guidedfilter(imgA, salA_local, 45, 0.3);

salA_global = localSaliency(imgA);
salA_gff = salA_gff + guidedfilter(imgA, salA_global, 45, 0.3);


subplot(1, 3, 1);
imshow(imgA, []);
title('Input Image');

subplot(1, 3, 2);
imshow(salA_local + salA_global, []);
title('Origin saliency');

subplot(1, 3, 3);
imshow(salA_gff, []);
title('Gff saliency');


imgB = imread('../datas/source20_2.tif');

imgB = double(imgB);

salB_local = localSaliency(imgB);
salB_gff = guidedfilter(imgB, salB_local, 45, 0.3);
salB_global = globalSaliency(imgB);
salB_gff = salB_gff + guidedfilter(imgB, salB_global, 45, 0.3);

figure;
subplot(1, 3, 1);
imshow(imgB, []);
title('Input Image');

subplot(1, 3, 2);
imshow(salB_local + salB_global, []);
title('Origin saliency');

subplot(1, 3, 3);
imshow(salB_gff, []);
title('Gff saliency');

I(:, :, 1) = salA_gff;
I(:, :, 2) = salB_gff;



