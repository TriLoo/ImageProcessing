function [ histA ] = calHist( img )
% ----------------------------
% Author : smher
% Date   : 2017.04.26
% Description :
%       This function implement the function of calculating the histogram
%       of input image img.  return 0~1 distrution
% ----------------------------

histA = zeros(1, 256);

[m,n] = size(img);

for i=1:m
    for j = 1:n
        temp = img(i,j);
        histA(temp+1) = histA(temp+1) + 1;
    end
end

% sum = 0;

% for i = 1:256
%     sum = sum + histA(1,i);
% end

end

