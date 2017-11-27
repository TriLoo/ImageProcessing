function [ SaliencyMap ] = localglobal( img )
% ----------------------
% Author : smher
% Data   : 2017.11.27
% Description:
%       This file implementing the combining local & global saliency
%       detection. Derived from 'context-aware saliency detection'.
% ----------------------

% local contrast based on gaussian filter, rad: 5 * 5

SaliencyMap = zeros(size(img));

c = 3;   % first try

localmean = GauSaliency(img);
localdist = localDist(img, localmean);

% global contrast based on mean filter and global dist
globalmean = GlobalMean(img);
globaldist = globalDist(img, globalmean);

% calculate the saliency map based on local & global contrast
SaliencyMap = localdist / (1 + c * globaldist);


function [ S ] = GauSaliency( H )
% Using the local average of the absolute value of H to construct the 
% saliency maps
N = size(H,3);
S = zeros(size(H,1),size(H,2),N);
for i=1:N
se = fspecial('gaussian',11,5);
S(:,:,i) = imfilter(H(:,:,i),se,'replicate');
end
end

    function [ G ] = GlobalMean(img)
        [m, n] = size(img);
        totalsum = 0;
        for i = 1 : m
            for j = 1 : n
                totalsum = total + img(i, j);
            end
        end
        G = totalsum / (m * n);
    end

end