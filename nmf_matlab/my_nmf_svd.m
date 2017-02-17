function [ W, H, err ] = my_nmf_svd(V, maxiter)
%-------------------------
% Author : smh
% Data   : 2017.02.16
% Description :
%        This file implement the nmf algorithm.
%        First : svd algorithm to initialization W & H
%        Then  : multiplicative to update W & H
%-------------------------

% Input : V : image matrix
%         maxiter : maxmum iteration
%         r : rank to factorization

eps = 1e-4;

[u, d, v] = svds(double(V),1);

u_size = size(u, 1);
v_size = size(v, 1);

u_pos = zeros(u_size, 1);
u_min = zeros(u_size, 1);
v_pos = zeros(v_size, 1);
v_min = zeros(v_size, 1);

% generate u+\u-; v+\v-
for i = 1 : u_size
    if(u(i) >= 0)
        u_pos(i) = u(i);
        u_min(i) = 0;
    else
        u_pos(i) = 0;
        u_min(i) = -u(i);
    end
end

for i = 1 : v_size
    if(u(i) >= 0)
        v_pos(i) = v(i);
        v_min(i) = 0;
    else
        v_pos(i) = 0;
        v_min(i) = -v(i);
    end
end

u_pos_norm = norm(u_pos, 1);
u_min_norm = norm(u_min, 1);
v_pos_norm = norm(v_pos, 1);
v_min_norm = norm(v_min, 1);

if(u_pos_norm == 0)
    u_pos_norm = eps;
end

if(u_min_norm == 0)
    u_min_norm = eps;
end

if(v_pos_norm == 0)
    v_pos_norm = eps;
end

if(v_min_norm == 0)
    v_min_norm = eps;
end


% calculate var 
var_pos = sqrt(u_pos_norm * v_pos_norm);
var_min = sqrt(u_min_norm * v_min_norm);

if(var_pos >= var_min)
    w_1 = sqrt(d) * u_pos;
    h_1 = sqrt(d) * v_pos;
%     w_1 = sqrt(d) * var_pos * u_pos;
%     h_1 = sqrt(d) * var_pos * v_pos;
else
    w_1 = sqrt(d) * u_min;
    h_1 = sqrt(d) * v_min;
%     w_1 = sqrt(d) * var_min * u_min;
%     h_1 = sqrt(d) * var_pos * v_min;
end

h_1_norm = norm(h_1, 1);

W = w_1 * h_1_norm;
H = h_1' / h_1_norm;

[W, H] = multiplicative(V, H, W, maxiter);

err = error_cal(V, W, H);

% show the figure
subplot(1, 2, 1);
imshow(V,[]);
title('origin image');
subplot(1,2,2);
imshow(W*H, []);
title('NMFed image');

end

