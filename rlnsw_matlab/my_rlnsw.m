function [ T, B ] = my_rlnsw( V )
%--------------------------------
% Author : smh
% Data   : 2017.02.17
% Desctiption :
%       this file implement the rlnsw algorithm after learning kong
%       weiwei's
%       algorithm structure : image copy - update - predict
%--------------------------------

% Input : V : image matrix
% Output : T : top matrix 
%          B : bottom matrix

clc;

[m, n] = size(V);
% matrix V process, add the two zero edges

% predict matrix
P = [1, 3/4; 3/4, 9/16];

% update matrix 
U = (1/2) * P;

% image copy
% I_1 = V;
% I_2 = V;

E = zeros(m,n);
L = zeros(m,n);

% Predict
for i = 1:m
    for j = 1:n
        if (i==1) || (i == m)        % ignore the first or last row
            E(i,j) = V(i,j);
        elseif (j==1) || (j == n)    % ignore the first or last column
            E(i,j) = V(i,j); 
        else
%             tmp = V();
            E(i,j) = V(i,j) - P(1,1) * V(i-1,j-1) + P(1,2)*V(i-1,j+1) + P(2,1)*V(i+1, j-1) + P(2,2)*V(i+1, j+1);
        end
    end
end

% Update
for i = 1:m
    for j = 1:n
        if(i==1)||(i==m)
            L(i,j) = V(i,j);
        elseif (j==1)||(j==n)
            L(i,j) = V(i,j);
        else
            L(i,j) = V(i,j) + U(1,1)*E(i-1,j-1) + U(1,2)*E(i-1,j+1) + U(2,1)*E(i+1,j-1) + U(2,2)*E(i+1,j+1);
        end
    end
end
                
T = E;
B = L;
end

