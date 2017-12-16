function [ wmA, wmB ] = WeightedMap( imgA, imgB )
% -----------------------
% Author : smh
% Date   : 2017.12.11
% Description:
%   This file including the implementation of generation weighted map in
%   'two-scale image fusion of visible and infrared iamges using saliency
%   detection'.
% -----------------------

salA = SaliencyMap(imgA);
salB = SaliencyMap(imgB);

wmA = salA ./ (salA + salB);
wmB = salB ./ (salB + salA);

end

