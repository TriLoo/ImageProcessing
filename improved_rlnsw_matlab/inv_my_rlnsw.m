function [ V ] = inv_my_rlnsw( H, M, L)
% -----------------------------------
% Author : smh
% Data   : 2017.02.18
% Descripton :
%       This file implement the invert transform of rlnsw.
% Copyright(C): git@github.com:TriLoo/ImageProcessing.git
% -----------------------------------

% Input  : H : high pass coefficient
%          L : low pass coefficient
% Output : V : the outcome of invert transform based on input.
% -------------!!!!! update !!!!------------------%
% Update 2017. 03. 15 
% Description : delete the "V = abs(V)" !
% -------------!!!!! update !!!!------------------%

if (size(H) ~= size(L))
    error('the size is error!');
else
    [m, n] = size(H);
end

% predict matrix
% P = [1, 3/4; 3/4, 9/16] ./ 2;
P = [1/4, 1/4; 1/4, 1/4];

% update matrix 
U = (1/2) * P;

V = zeros(m, n);

% diaglog direction
for i=1:m
    for j=1:n
        if (i==1)||(i==m)
            V(i,j) = M(i,j);
        elseif (j==1)||(j==n)
            V(i,j) = M(i,j);
        else
            V(i,j) = M(i,j) - U(1,1)*H(i-1,j-1) - U(1,2)*H(i-1,j+1) - U(2, 1)*H(i+1,j-1) - U(2,2)*H(i+1,j+1);
        end
    end
end

T = V;

% horization direction
for i=1:m
    for j=1:n
        if (i==1)||(i==m)
            V(i,j) = L(i,j);
        elseif (j==1)||(j==n)
            V(i,j) = L(i,j);
        else
            V(i,j) = L(i,j) - U(1,1)*T(i-1,j) - U(1,2)*T(i,j-1) - U(2, 1)*T(i+1,j) - U(2,2)*T(i,j+1);
        end        
    end
end

% V = abs(V);
% V = V;

end

