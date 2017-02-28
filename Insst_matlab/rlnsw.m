function [ y ] = rlnsw( x, level )
% --------------------------------
% Author : smh
% Data   : 2017. 02. 28
% Description :
%       This function implement the various levels rlnsw decompose 
% --------------------------------

% Input : x : the matrix 
%         level : the level to decompose.
% Output : 
%         y : the cell vector of coefficient matrix.

if nargin <= 1
    level = 3;
end

y = cell(1, level+1);

% [m, n] = size(x);

[y{1, 2}, y{1, 1}] = my_rlnsw(x);


for i = 2:level
    [y{1, i+1}, y{1, i}] = my_rlnsw(y{1, i});
end

end

