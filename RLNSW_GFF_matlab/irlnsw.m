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

temp = cell(1, level + 1);
temp{1, 1} = inv_my_rlnsw(x{1, level + 1}, x{1, level});

for i = 1:level-1
    temp{1, i+1} = inv_my_rlnsw(temp{1, i}, x{1,level - i});
end

y = temp{1, level};

end

