function [ res ] = JBF( I, p, rad, deltaS, deltaR)
% ----------------------
% Author : smh
% Date   : 2017.12.18
% Description:
%   This file includes the implementation of Joint Bilateral Filtering
%   (JBF) based on 'Guided image filter' and 'Rolling Guidance Filter'.
%   Inputs: 
%       I : The iuput guided image
%       p : The input filtering image
%       rad : the radius of neighbour size
%       deltaS : control the spartial similarity
%       deltaR : control the color similarity
%   Outputs:
%       res : the output of rolling guidance filter
% ----------------------

[M, N] = size(p);
res = zeros(M, N);

% pad the input guidance image
img = padarray(I, [rad, rad], 'circular', 'both');

divS = 2 * deltaS * deltaS;     % The division of spartial similarity.
divR = 2 * deltaR * deltaR;     % The division of color similarity.
for i = rad + 1 : rad + M
    for j = rad + 1 : rad + N
        sumExp = 0.0;
        sumColor = 0.0; 
        for m = -rad : rad
            for n = -rad : rad
                expVal = exp(-(m * m + n * n) / divS - (img(i, j) - img(i + m, j + n))^2 / divR);
                sumExp = sumExp + expVal;
                sumColor = sumColor + expVal * p(i - rad, j - rad);
            end
        end
        res(i - rad, j - rad) = sumColor / sumExp;
    end
end

end

