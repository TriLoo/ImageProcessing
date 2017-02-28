function [ x1n, y1n, x2n, y2n, D ] = gen_x_y_cordinates( n )
% ----------------------------------
% Author : smh
% Data   : 2017.02.28
% Description :
%       This function generate the x & y vectors that contains the i, j
%       coordinates to extract radical slices.
% ----------------------------------

% Input : n is the order of the block to be used.
% Output : x1, y1, 

n = n + 1;

x1 = zeros(n, n);
y1 = zeros(n, n);
x2 = zeros(n, n);
y2 = zeros(n, n);

xt = zeros(1, n);
m = zeros(1, n);

for i=1:n
    y0 = 1;
    x0 = i;
    x_n = n - i + 1;
    y_n = n;
    if(x_n == x0)
        flag = 1;
    else
        m(i) = (y_n - y0)/(x_n - x0);
        flag = 0;
    end
    xt(i, :) = linspace(x0, x_n, n);
    for j= 1:n
        if flag == 0
            y1(i, j) = m(i)*(xt(i, j) - x0) + y0;
            y1(i, j) = round(y1(i, j));
            x1(i, j) = round(xt(i, j));
            x2(i, j) = y1(i, j);
            y2(i, j) = x1(i, j);
        else
            x1(i, j) = (n - 1)/2 + 1;
            y1(i, j) = j;
            x2(i, j) = j;
            y2(i, j) = (n - 1)/2 + 1;
        end
    end
end

n = n - 1;
x1n = zeros(n, n);
y1n = zeros(n, n);
x2n = zeros(n, n);
y2n = zeros(n, n);

for i = 1:n 
    for j = 1:n 
        x1n(i, j) = x1(i, j);
        y1n(i, j) = y1(i, j);
        x2n(i, j) = x2(i+1, j);
        y2n(i, j) = y2(i+1, j);
    end
end

x1n = flipud(x1n);
y2n(n, 1) = n;

D = avg_pol(n, x1n, y1n, x2n, y2n);

end

function D = avg_pol(L, x1, y1, x2, y2)

D = zeros(L);
for i = 1:L
    for j =  1:L
        D(y1(i, j), x1(i, j)) = D(y1(i, j), x1(i, j)) + 1;
    end
end

for i = 1:L
    for j=1:L
        D(y2(i,j), x2(i, j)) = D(y2(i,j), x2(i,j)) + 1;
    end
end

end
