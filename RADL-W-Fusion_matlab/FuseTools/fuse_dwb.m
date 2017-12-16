function Y = fuse_dwb(M1, M2, zt, ap, mp)
%Y = fuse_dwb(M1, M2, zt, ap, mp) image fusion with DWT, Wavelet is DBSS(2,2)
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

% define filters, padd with zeros due to phase distortions
h1 = [-1 2 6 2 -1 0 0]   / (4*sqrt(2));
g1 = [0 0 -2 4 -2 0 0]   / (4*sqrt(2));
h2 = [0 0 0 2 4 2 0]     / (4*sqrt(2));
g2 = [0 -1 -2 6 -2 -1 0] / (4*sqrt(2));

% cells for selected images
E = cell(3,zt);

% loop over decomposition depth -> analysis
for i1 = 1:zt 
  % calculate and store actual image size 
  [z s]  = size(M1); 
  zl(i1) = z; sl(i1)  = s;
  
  % image A
  Z1 = dec(conv2(es(M1, 7, 1), g1, 'valid'),2);
  A1 = dec(conv2(es(Z1, 7, 2), g1','valid'),1);
  A2 = dec(conv2(es(Z1, 7, 2), h1','valid'),1);
  Z1 = dec(conv2(es(M1, 7, 1), h1, 'valid'),2);
  A3 = dec(conv2(es(Z1, 7, 2), g1','valid'),1);
  A4 = dec(conv2(es(Z1, 7, 2), h1','valid'),1);
  % image B
  Z1 = dec(conv2(es(M2, 7, 1), g1, 'valid'),2);
  B1 = dec(conv2(es(Z1, 7, 2), g1','valid'),1);
  B2 = dec(conv2(es(Z1, 7, 2), h1','valid'),1);
  Z1 = dec(conv2(es(M2, 7, 1), h1, 'valid'),2);
  B3 = dec(conv2(es(Z1, 7, 2), g1','valid'),1);
  B4 = dec(conv2(es(Z1, 7, 2), h1','valid'),1);
 
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
  % undecimate and interpolate (rows)
  A4 = conv2(es(undec(A4,1), 3, 2), h2', 'valid');   
  A3 = conv2(es(undec(E{3,i1},1), 3, 2), g2', 'valid'); 
  A2 = conv2(es(undec(E{2,i1},1), 3, 2), h2', 'valid'); 
  A1 = conv2(es(undec(E{1,i1},1), 3, 2), g2', 'valid'); 

  % undecimate and interpolate (columns)  
  A4 = conv2(es(undec(A4+A3,2), 3, 1), h2, 'valid');
  A2 = conv2(es(undec(A2+A1,2), 3, 1), g2, 'valid');  

  % add images and select valid part 
  A4 = A4 + A2;
  A4 = A4(5:5+zl(i1)-1,5:5+sl(i1)-1);
end;

% copy image
Y = A4;


