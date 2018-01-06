function [ sal ] = localSal( img, rad, delta1, delta2 )
% ----------------------
% Author : smh
% Date   : 2018.01.05
% Description: 
%   This file implementing the local saliency detection in "Fusion for visible and infrared images using visual weight analysis and bilateral filter-based multi scale decomposition"
% ----------------------

if nargin == 1
    rad = 7;
    delta1 = 3;
    delta2 = 0.5;
end

se1= fspecial('gaussian', rad, delta1);
se2 = fspecial('gaussian', rad, delta2);

sal = abs(imfilter(img, se1) - imfilter(img, se2));

salMin = min(min(sal));
salMax = max(max(sal));

sal = (sal - salMin) / (salMax - salMin);

end

