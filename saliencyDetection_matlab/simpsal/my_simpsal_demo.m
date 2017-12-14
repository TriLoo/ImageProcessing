
%%% demonstration of how to use simpsal,
%%% the simple matlab implemenation of visual saliency.

%% 1. simplest possible usage : compute standard Itti-Koch Algorithm:

map1 = simpsal('lena.jpg');

%% 2. more complciated usage:

img = imread('lena.jpg');
p = default_fast_param;
p.blurRadius = 0.02;     % e.g. we can change blur radius 
map2 = simpsal(img,p);

subplot(1,3,1);
imshow(img);
title('Original');

subplot(1,3,2);
imshow(map1)
title('Itti Koch');

subplot(1,3,3);
imshow(map2);
title('Itti Koch Simplified');
