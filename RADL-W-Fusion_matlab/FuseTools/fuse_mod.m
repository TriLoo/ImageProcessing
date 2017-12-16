function Y = fuse_mod(M1, M2, zt, ap, mp)
%Y = fuse_lap(M1, M2, zt, ap, mp) image fusion with morphological difference pyramid
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
  % check and store new size
  [z1 s1] = size(M1);
  
  % gray scale opening
	O1 = ordfilt2(ordfilt2(es2(M1, 3), 1, ones(5)), 25, ones(5));
	O2 = ordfilt2(ordfilt2(es2(M2, 3), 1, ones(5)), 25, ones(5));
	% gray scale closing 
	O1 = ordfilt2(ordfilt2(O1, 25, ones(5)), 1, ones(5));
	O2 = ordfilt2(ordfilt2(O2, 25, ones(5)), 1, ones(5));
  % select valid image region 
  O1 = O1(4:z1+3,4:s1+3);
  O2 = O2(4:z1+3,4:s1+3);
  
  % decimate
  Z1 = dec2(O1);
  Z2 = dec2(O2);
  
  % decimate, undecimate and dilate
  O1 = ordfilt2(es2(undec2(dec2(O1)), 3), 49, ones(7));
  O2 = ordfilt2(es2(undec2(dec2(O2)), 3), 49, ones(7));
  % select valid image region 
  O1 = O1(4:z1+3,4:s1+3);
  O2 = O2(4:z1+3,4:s1+3);

 	% select coefficients and store them
  E(i1) = {selc(M1-O1, M2-O2, ap)};
  
  % copy tmp images
  M1 = Z1;
  M2 = Z2;
end;  

% select base coefficients of last decompostion stage
M1 = selb(M1,M2,mp);

% loop over decomposition depth -> synthesis
for i1 = zt:-1:1
  % dilate 
  M1 = ordfilt2(es2(undec2(M1), 3), 49, ones(7));
 	% select valid image region 
	M1 = M1(4:ceil(zl(i1)/2)*2+3,4:ceil(sl(i1)/2)*2+3);
  % add coefficients
  M1  = M1 + E{i1};
  % select valid image region 
  M1 	= M1(1:zl(i1),1:sl(i1));
end;

Y = M1;
