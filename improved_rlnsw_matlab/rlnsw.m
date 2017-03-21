function [ y ] = rlnsw( V, level )
% ------------------------------------
% Author : smher
% Data   : 2017.03.15
% Description :
%       this file implement multi-levels rlnsw decompose based on
%       single-level rlnsw_0
% ------------------------------------

% Input : V :  inputted double image data.
% Output : y : a Cell data structure, including multi-levels coefficiences.

y = cell(1, 2^level + 1);

[y{1, 3}, y{1, 2}, y{1, 1}] = my_rlnsw(V);

for i=3:2^level-1
    [y{1, i+2}, y{1, i+1}, y{1, i}] = my_rlnsw(y{1, i});
end

end

