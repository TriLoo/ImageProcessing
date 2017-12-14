function [S] = pdfSmap(DATA, ROW, COL)

muDATA = mean(DATA);
cmDATA = cov(DATA);
icmDATA = pinv(cmDATA);
[L, D] = size(DATA);
pvl = zeros(L,1);
for i = 1:L
    v = DATA(i,:);
    pvl(i) = exp( -(1/2) * (v-muDATA)*icmDATA*(v-muDATA)' ) / ( (2*pi)^(D/2) * (det(cmDATA))^(1/2) );
end

new_pvl = reshape(pvl,ROW,COL);
ipvl = new_pvl.^(-1);
z = isinf(ipvl); ipvl(z) = -1000; z = ipvl == -1000; ipvl(z) = max(ipvl(:)); %adjuest Inf values to maximum valuein ipvl
S = log10( ipvl ).^(0.5);
z = isnan(S); S(z) = 0; %adjust NaN values to 0
S = imfilter(S, fspecial('gaussian', 5, 5), 'symmetric', 'conv');
S = real(S - min(real(S(:)))); %range starting from 0 if any minus value
end