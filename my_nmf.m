function [ W, H ] = my_nmf( V, r, maxiter )
%------------------------------
% Author : smh
% Data   : 2017.02.12
% Description :
%       The implementation of nmf matrix factorization
%       Random Initialization, low speed to convergence.
%------------------------------

% input : V : image matrix 
%         r : rank for the factorization
%         maxiter : maximum number of iterations

close all;
clear ;

inShowOn = 1;

if nargin ~= 3
    disp 'Usage : [w, h] = new_nmf(V, r, maxiter)'
    disp 'Copyright(C): http://TrueMark.cn'
end
v = imread('barbara.gif');
r = input('Choose your own rank for the factorization [49]:');
if isempty(r)
    r = 49;
end
maxiter = input('Choose the maxmum number of iterations[100]:');
if isempty(maxiter)
    maxiter = 100;
end

v = double(v);
[n, m] = size(v);

W = rand(n, r);
W = W./(ones(n,1)*sum(W));

H = rand(r, m);
eps = 1e-5;
h = waitbar(0, 'NMF');

for iter=1:maxiter
    H = H.*(W'*((v+eps)./(W*H + eps)));
    W = W.*((v*H'+eps)./(W*H*H' + eps));
    W = W./(ones(n,1)*sum(W));
    waitbar(iter/maxiter, h);
end

close(h);

if (inShowOn)
    subplot(1,2,1);
    imshow(v,[]);
    title('origin image');
    subplot(1,2,2);
    imshow(W*H, []);
    title('NMFed image');
end

end

