function [ y ] = irlnsw( x, level )
% ----------------------------------
% Author : smher
% Data   : 2017.03.15
% Descriptioin :
%       This file implement the multi-levels invert decomposition based on 
%       inv_my_rlnsw.
% ----------------------------------

% Input : x : the multi-levels coefficiences
% Output : y : the restored image obtained by multi-levels invert rlnsw.

temp = cell(1, level);
temp{1, 1} = inv_my_rlnsw(x{1, 2^level + 1}, x{1, 2^level}, x{1, 2^level - 1});

for i = 2:level
    temp{1, i} = inv_my_rlnsw(temp{1, i-1}, x{1,2^(level-i + 1)}, x{1, 2^(level-i + 1)-1});
end

y = temp{1, level};

end

