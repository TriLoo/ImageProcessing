function [ imgRes ] = HybridMSDFusion( imgA, imgB )
% ---------------------
% Author : smh
% Date   : 2017.12.11
% Description:
%   This file is the top-level function of Hybrid_MSD_Fusion.
% ---------------------
imgA = double(im2uint8(imgA));
imgB = double(im2uint8(imgB));

nLevel = 4;
%% ***********************************************************************
% For this parameter, you can also try "lambda = 3000", depending on the
% source images (also your preference).
lambda = 30;  % lambda = 3000

%% ---------- Hybrid Multi-scale Decomposition --------------
sigma = 2.0;
sigma_r = 0.05;
k = 2;

M1 = cell(1, nLevel+1);
M1L = cell(1, nLevel+1);
M1{1} = imgA;
M1L{1} = imgA;
M1D = cell(1, nLevel+1);
M1E = cell(1, nLevel+1);
sigma0 = sigma;
for j = 2:nLevel+1,
    w = floor(3*sigma0);
    h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);   
    M1{j} = imfilter(M1{j-1}, h, 'symmetric');
    %M1L{j} = 255*bfilter2(M1L{j-1}/255,w,[sigma0, sigma_r/(k^(j-2))]);
    M1L{j} = 255*fast_bfilter2(M1L{j-1}/255,[sigma0, sigma_r/(k^(j-2))]);
 
    M1D{j} = M1{j-1} - M1L{j};
    M1E{j} = M1L{j} - M1{j};
    
    sigma0 = k*sigma0;
end

M2 = cell(1, nLevel+1);
M2L = cell(1, nLevel+1);
M2{1} = imgB;
M2L{1} = imgB;
M2D = cell(1, nLevel+1);
M2E = cell(1, nLevel+1);
sigma0 = sigma;
for j = 2:nLevel+1,
    w = floor(3*sigma0);
    h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);   
    M2{j} = imfilter(M2{j-1}, h, 'symmetric');
    %M2L{j} = 255*bfilter2(M2L{j-1}/255,w,[sigma0, sigma_r/(k^(j-2))]);
    M2L{j} = 255*fast_bfilter2(M2L{j-1}/255,[sigma0, sigma_r/(k^(j-2))]);
 
    M2D{j} = M2{j-1} - M2L{j};
    M2E{j} = M2L{j} - M2{j};

    sigma0 = k*sigma0;
end

%% ---------- Multi-scale Combination --------------
for j = nLevel+1:-1:3
b2 = abs(M2E{j});
b1 = abs(M1E{j});
R_j = max(b2-b1, 0);
Emax = max(R_j(:));
P_j = R_j/Emax;

C_j = atan(lambda*P_j)/atan(lambda);

% Base level combination
sigma0 = 2*sigma0;
if j == nLevel+1
    w = floor(3*sigma0);
    h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);
    lambda_Base = lambda;
    %lambda_Base = 30;
    C_N = atan(lambda_Base*P_j)/atan(lambda_Base);
    C_N = imfilter(C_N, h, 'symmetric');
    MF = C_N.*M2{j} + (1-C_N).*M1{j};
end

% Large-scale combination
sigma0 = 1.0;
w = floor(3*sigma0);
h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);   
C_j = imfilter(C_j, h, 'symmetric');

D_F = C_j.*M2E{j}+ (1-C_j).*M1E{j};
MF = MF + D_F;
D_F = C_j.*M2D{j}+ (1-C_j).*M1D{j};
MF = MF + D_F;
end 

% Small-scale combination
sigma0 = 0.2;
w = floor(3*sigma0);
h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);   
C_0 = double(abs(M1E{2}) < abs(M2E{2}));
C_0 = imfilter(C_0, h, 'symmetric');
D_F = C_0.*M2E{2} + (1-C_0).*M1E{2};
MF = MF + D_F;  
C_0 = abs(M1D{2}) < abs(M2D{2});
C_0 = imfilter(C_0, h, 'symmetric');
D_F = C_0.*M2D{2} + (1-C_0).*M1D{2};
MF = MF + D_F;

%% ---------- Fusion Result --------------
% FI = ImRegular(MF);   % The intensities are regulated into [0, 255]
imgRes = max(min(MF*1.08,255), 0);

imgRes = im2double(uint8(imgRes));

end

