function [ y ] = inv_rlnsw( x, level )
% ---------------------------------
% Author : smh
% Data   : 2017. 02. 28
% Descriptioin :
%       This function implement various levels rlnsw invert decompose.
% ---------------------------------

% Input : x : the cell of coefficients obtained in rlnsw.
%         level : the level to invert decompose
% Output : y : the origin image matrix

% y = cell(1, level+1);
% 
% y{1,1} = x{1,1};

temp = inv_my_rlnsw(x{1, level+1}, x{1, level});

for i = level-1:-1:1
    temp = inv_my_rlnsw(temp, x{1, i});
end

y = temp;

imshow(y, []);

end

