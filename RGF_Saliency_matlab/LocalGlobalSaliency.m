function [ salLocalGlobal ] = LocalGlobalSaliency( img, name )
% ----------------------
% Author : smh
% Date   : 2017.12.04
% Description:
%   This file including the implementation of combining local & global
%   saliency detection based on 'globalSaliency.m' & 'localSaliency.m'.
%   Inputs:
%       img: the input image.
%       name: the name of local saliency detection algorithm.
%           'GOL': gaussian of laplacian.
%           'CA' : the context-aware saliency detection algorithm.
%           'FT' : frequency tuned saliency detection based.
%   Outputs:
%       salLocalGlobal: the output saliency map.
% ----------------------


SalGlobal = globalSaliency(img);
% SalGlobal = ftSaliency(img);
switch name
    case 'GOL'
        SalLocal = localSaliency(img);
    case 'CA'
        SalLocal = caSaliency(img, 8, 2);
    case 'FT'
        SalLocal = ftSaliency(img);
    otherwise
        error('Invalid arguments');
end

close all;
% the first processing method: simply add two saliency map.
c = 0.9;
salLocalGlobal = c * SalLocal + (1 - c) * SalGlobal;

% the processing method used in 'Saliency Filter: Contrast Based Filtering
% for Saliency Region Detection'.
% salLocalGlobal = SalGlobal + exp(6 * SalLocal);


% subplot(1, 2, 1);
% imshow(img, []);
% title('Input Image');
% subplot(1, 2, 2);
% imshow(salLocalGlobal, []);
% title('Saliency Map');


end

