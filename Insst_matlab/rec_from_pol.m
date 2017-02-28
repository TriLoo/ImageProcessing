function [ C ] = rec_from_pol( l, n, x1, y1, x2, y2, D)
% ----------------------------------
% Author : smh
% Data   : 2017. 02. 28
% Description :
%       This function re-assembles radical slice into block.
% ----------------------------------

% Output : C is the re-assembled block matrix

C = zeros(n, n);
option = 0;
if(option == 1)
    for i = 1:n
        for j = 1:n
            C(y1(i, j), x1(i, j)) = l(i, j);
            C(y2(i, j), x2(i, j)) = l(i+n, j);
        end
    end
else
    for i = 1:n
        for j=1:n
            C(y1(i,j), x1(i,j)) = C(y1(i,j), x1(i,j)) + l(i, j);
            C(y2(i,j), x2(i,j)) = C(y2(i,j), x2(i,j)) + l(i, j);
        end
    end
    C = C./D;
end


end

