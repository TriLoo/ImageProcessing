function [ globaldist ] = globalDist( img, val )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

globaldist = zeros(size(img));

[m, n] = size(img);

for i = 1:m
    for j = 1:n
        globaldist(i, j) = abs(img(i, j) - val);
    end
end

end

