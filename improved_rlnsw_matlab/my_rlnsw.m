function [ T, M, B ] = my_rlnsw( V )
%--------------------------------
% Author : smh
% Data   : 2017.03.21
% Desctiption :
%       this file implement the rlnsw algorithm after learning kong
%       weiwei's
%       algorithm structure : image copy - update - predict
%--------------------------------

% Input  : V : image matrix
% Output : T : top matrix 
%          B : bottom matrix

clc;

[m, n] = size(V);
% matrix V process, add the two zero edges
% TO DO ...

V = double(V);

% predict matrix
% P = [1, 3/4; 3/4, 9/16];
P = [1/4, 1/4; 1/4, 1/4];

% update matrix 
U = (1/2) * P;

% image copy
% I_1 = V;
% I_2 = V;

Feven = V;
Fodd = V;

T = zeros(m, n);
B = zeros(m, n);
M = zeros(m, n);

% horizontal && vertical
% Predict
for i = 1:m
    for j = 1:n
        if(i==1) || (i == m)        % ignore the first or last row
            T(i,j) = V(i,j);
        elseif(j==1) || (j == n)    % ignore the first or last column
            T(i,j) = V(i,j); 
        else
            T(i,j) = Fodd(i,j) - P(1,1) * Feven(i-1,j) - P(1,2)*Feven(i,j-1) - P(2,1)*Feven(i+1, j) - P(2,2)*Feven(i, j+1);
        end
    end
end

% Update
for i = 1:m
    for j = 1:n
        if(i==1)||(i==m)
            B(i,j) = Feven(i,j);
        elseif (j==1)||(j==n)
            B(i,j) = Feven(i,j);
        else
            B(i,j) = Feven(i,j) + U(1,1)*T(i-1,j) + U(1,2)*T(i,j-1) + U(2,1)*T(i+1,j) + U(2,2)*T(i,j+1);
        end
    end
end

Feven = T;
Fodd = T;

% diagonal directional
% Predict
for i = 1:m
    for j = 1:n
        if(i==1) || (i == m)        % ignore the first or last row
            T(i,j) = Fodd(i,j);
        elseif(j==1) || (j == n)    % ignore the first or last column
            T(i,j) = Fodd(i,j); 
        else
            T(i,j) = Fodd(i,j) - P(1,1) * Feven(i-1,j-1) - P(1,2)*Feven(i-1,j+1) - P(2,1)*Feven(i+1, j-1) - P(2,2)*Feven(i+1, j+1);
        end
    end
end

% Update
for i = 1:m
    for j = 1:n
        if(i==1)||(i==m)
            M(i,j) = Feven(i,j);
        elseif (j==1)||(j==n)
            M(i,j) = Feven(i,j);
        else
            M(i,j) = Feven(i,j) + U(1,1)*T(i-1,j-1) + U(1,2)*T(i-1,j+1) + U(2,1)*T(i+1,j-1) + U(2,2)*T(i+1,j+1);
        end
    end
end
           
end

