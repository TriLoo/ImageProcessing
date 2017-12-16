function Y = fuse_gra(M1, M2, zt, ap, mp)
%Y = fuse_gra(M1, M2, zt, ap, mp) image fusion with gradient pyramid
%
%    M1 - input image #1
%    M2 - input image #2
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

% define filters 
w  = [1 4 6 4 1] / 16;
v  = [1 2 1] / 4;
d1 = [1 -1];
d2 = [0 -1; 1 0] / sqrt(2);
d3 = [-1 1];
d4 = [-1 0; 0 1] / sqrt(2);
% compute derivatives
d1e = conv2(d1,d1); 
d1e = [zeros(1,3); d1e; zeros(1,3)];
d2e = conv2(d2,d2);
d3e = d1e';
d4e = conv2(d4,d4);

% cells for selected images
E = cell(1,zt);

% loop over decomposition depth -> analysis
for i1 = 1:zt 
  % calculate and store actual image size 
  [z s]  = size(M1); 
  zl(i1) = z; sl(i1)  = s;
  
  % check if image expansion necessary 
  if (floor(z/2) ~= z/2), ew(1) = 1; else, ew(1) = 0; end;
  if (floor(s/2) ~= s/2), ew(2) = 1; else, ew(2) = 0; end;

  % perform expansion if necessary
  if (any(ew))
  	M1 = adb(M1,ew);
  	M2 = adb(M2,ew);
  end;	
 
 	% perform filtering 
  G1 = conv2(conv2(es2(M1,2), w, 'valid'),w', 'valid');
  G2 = conv2(conv2(es2(M2,2), w, 'valid'),w', 'valid');
  Z1 = es2(M1+conv2(conv2(es2(M1, 1), v, 'valid'), v', 'valid'), 1);
  Z2 = es2(M2+conv2(conv2(es2(M2, 1), v, 'valid'), v', 'valid'), 1);
  
  % compute directional derivatives
  B  = zeros(size(M1));
  D1 = conv2(Z1, d1e, 'valid');
  D2 = conv2(Z2, d1e, 'valid');
  B  = B + selc(D1, D2, ap);
  
  D1 = conv2(Z1, d2e, 'valid');
  D2 = conv2(Z2, d2e, 'valid');
  B  = B + selc(D1, D2, ap);
  
  D1 = conv2(Z1, d3e, 'valid');
  D2 = conv2(Z2, d3e, 'valid');
  B  = B + selc(D1, D2, ap);
  
  D1 = conv2(Z1, d4e, 'valid');
  D2 = conv2(Z2, d4e, 'valid');
  B  = B + selc(D1, D2, ap);
  
  % store coefficients
  E(i1) = {-B/8};

  % decimate
  M1 = dec2(G1);
  M2 = dec2(G2);
end;

% select base coefficients of last decompostion stage
M1 = selb(M1,M2,mp);

% loop over decomposition depth -> synthesis
for i1 = zt:-1:1
  % undecimate and interpolate 
  M1T = conv2(conv2(es2(undec2(M1), 2), 2*w, 'valid'), 2*w', 'valid');
  % add coefficients
  M1  = M1T + E{i1};
  % select valid image region 
  M1 	= M1(1:zl(i1),1:sl(i1));
end;

% copy image
Y = M1;
