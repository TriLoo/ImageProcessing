function [ W, H ] = multiplicative( V, A, B, maxiter)

%-------------------------
% A - H
% B - W
%-------------------------

eps = 1e-5;

V = double(V);

n = size(B, 1);

h = waitbar(0, 'NMF');

A_curr = A;
B_curr = B;

for iter=1:maxiter
    A_next = A_curr.*(((B_curr'*V+eps)./(B_curr'*B_curr*A_curr + eps)));
    B_next = B_curr.*((V*A_curr'+eps)./(B_curr*A_curr*A_curr' + eps));
    B_curr = B_next./(ones(n,1)*sum(B_next));
    A_curr = A_next;
    waitbar(iter/maxiter, h);
end
close(h);

W = B_curr;
H = A_curr;

end

