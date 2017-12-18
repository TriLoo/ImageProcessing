function [ res ] = RGFcolor( img, rad, deltaS, deltaR, level )
% -------------------------
% Author : smh
% Date   : 2017.12.18
% Description:
%   This file including the implementation of 'rolling guidance filter'
%   working on color input images.
%   Reference: 'Rolling Guidance Filter', https://cn.mathworks.com/matlabc
%       entral/fileexchange/62455-joint-bilateral-filter?focused=7502124&tab=function&requestedDomain=www.mathworks.com
%   Inputs : same as single channel RGF
%   Outputs : a 'level + 1' cell.
% -------------------------

if nargin < 1
    error('Too few inputs ...');
end

if ndims(img) ~= 3
    error('Input image should be color image');
end

if isa(img, 'uint8')
    img = im2double(img);
end

if nargin < 2
    rad = 2;
    deltaS = 1.2;
    deltaR = 0.25;
    level = 3;
end

if length(deltaS) == 1
    deltaS = repmat(deltaS, [level + 1, 1]);
end

C = ones(size(img));

res = cell(level + 1, 1);

for i = 1 : level
    if i == 1
        temp = JBFcolor(C, img, rad, deltaS(i), deltaR);
        res{i} = temp;
    end
    res{i + 1} = JBFcolor(res{i}, img, rad, deltaS(i + 1), deltaR);
end

end

