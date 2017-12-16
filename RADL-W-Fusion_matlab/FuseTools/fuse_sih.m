function Y = fuse_sih(M1, M2, zt, ap, mp)
%Y = fuse_sih(M1, M2, zt, ap, mp) image fusion with SIDWT, Wavelet is Haar
%
%    M1 - input image A
%    M2 - input image B
%    zt - maximum decomposition level
%    ap - coefficient selection highpass (see selc.m) 
%    mp - coefficient selection base image (see selb.m) 
%
%    Y  - fused image   

%    (Oliver Rockinger 16.08.99)

% check inputs 
[z1 s1] = size(M1);
[z2 s2] = size(M2);
if (z1 ~= z2) | (s1 ~= s2)
  error('Input images are not of same size');
end;

% cells for selected images
E = cell(3,zt);

% loop over decomposition depth -> analysis
for i1 = 1:zt 
  % calculate and store actual image size 
  [z s]  = size(M1); 
  zl(i1) = z; sl(i1)  = s;
  
  % define actual filters (inserting zeros between coefficients)
  h1 = [zeros(1,floor(2^(i1-2))), 0.5, zeros(1,floor(2^(i1-1)-1)), 0.5, zeros(1,max([floor(2^(i1-2)),1]))];
  g1 = [zeros(1,floor(2^(i1-2))), 0.5, zeros(1,floor(2^(i1-1)-1)), -0.5, zeros(1,max([floor(2^(i1-2)),1]))];
  fh = floor(length(h1)/2);
  
  % image A
  Z1 = conv2(es(M1, fh, 1), g1, 'valid');
  A1 = conv2(es(Z1, fh, 2), g1','valid');
  A2 = conv2(es(Z1, fh, 2), h1','valid');
  Z1 = conv2(es(M1, fh, 1), h1, 'valid');
  A3 = conv2(es(Z1, fh, 2), g1','valid');
  A4 = conv2(es(Z1, fh, 2), h1','valid');
  % image B
  Z1 = conv2(es(M2, fh, 1), g1, 'valid');
  B1 = conv2(es(Z1, fh, 2), g1','valid');
  B2 = conv2(es(Z1, fh, 2), h1','valid');
  Z1 = conv2(es(M2, fh, 1), h1, 'valid');
  B3 = conv2(es(Z1, fh, 2), g1','valid');
  B4 = conv2(es(Z1, fh, 2), h1','valid');
 
  % select coefficients and store them
  E(1,i1) = {selc(A1, B1, ap)};
 	E(2,i1) = {selc(A2, B2, ap)};
 	E(3,i1) = {selc(A3, B3, ap)};
 
 	% copy input image for next decomposition stage
  M1 = A4;  
  M2 = B4;   
end;

% select base coefficients of last decompostion stage
A4 = selb(A4,B4,mp);

% loop over decomposition depth -> synthesis
for i1 = zt:-1:1
	% define actual filters (inserting zeros between coefficients)
  h2 = fliplr([zeros(1,floor(2^(i1-2))), 0.5, zeros(1,floor(2^(i1-1)-1)), 0.5, zeros(1,max([floor(2^(i1-2)),1]))]);
  g2 = fliplr([zeros(1,floor(2^(i1-2))), 0.5, zeros(1,floor(2^(i1-1)-1)), -0.5, zeros(1,max([floor(2^(i1-2)),1]))]);
  fh = floor(length(h2)/2);
  
  % filter (rows)
  A4 = conv2(es(A4, fh, 2), h2', 'valid');   
  A3 = conv2(es(E{3,i1}, fh, 2), g2', 'valid'); 
  A2 = conv2(es(E{2,i1}, fh, 2), h2', 'valid'); 
  A1 = conv2(es(E{1,i1}, fh, 2), g2', 'valid'); 

  % filter (columns)  
  A4 = conv2(es(A4+A3, fh, 1), h2, 'valid');  
  A2 = conv2(es(A2+A1, fh, 1), g2, 'valid');  

  % add images 
  A4 = A4 + A2;
end;

% copy image
Y = A4;
