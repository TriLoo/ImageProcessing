function [S, SL, SG] = funWaveletSaliency(LAB,waveName)

%adjust to a value in which decompostion can reach the coarsest level
%on x or y axis before this value and can stop
maxDecompostionLevel = 16; 
fprintf('Intensity channel wavelet decompostion: ');
[caI, chI, cvI, cdI, RowDataI, ColDataI]  = createWaveletMap(LAB(:,:,1), maxDecompostionLevel, waveName);
fprintf('Red/Green channel wavelet decompostion: ');
[caRG,chRG,cvRG,cdRG,RowDataRG,ColDataRG] = createWaveletMap(LAB(:,:,2), maxDecompostionLevel, waveName);
fprintf('Blue/Yellow channel wavelet decompostion: ');
[caBY,chBY,cvBY,cdBY,RowDataBY,ColDataBY] = createWaveletMap(LAB(:,:,3), maxDecompostionLevel, waveName);

[ROW,COL,DIM] = size(LAB);
SL = zeros(ROW,COL);

DATA = [];

for i = 1:length(caI)
    waveletLevel = i;
    % close all;    

    %% Intensity Conspicuity Map
    C1 = createFeatureMap(chI, cvI, cdI, RowDataI, ColDataI, ROW, COL, waveletLevel, waveName);

    %% Red/Green Conspicuity Map
    C2 = createFeatureMap(chRG,cvRG,cdRG,RowDataRG,ColDataRG, ROW, COL, waveletLevel, waveName);

    %% Blue/Yellow ConspicuityMap
    C3 = createFeatureMap(chBY,cvBY,cdBY,RowDataBY,ColDataBY, ROW, COL, waveletLevel, waveName);
    
    % features in several decomposition level for global saliency detection in multi-channel
    DATA = [DATA C1(:) C2(:) C3(:)];
    
    %Combining conspicuity maps of each channel using max function 
    CS = max(max(C1,C2),C3);
    %Local saliency across-scale addition
    SL = SL + CS;

end

SL = imfilter(SL, fspecial('gaussian', 5, 5), 'symmetric', 'conv');

%Global Sliency Computation using probability density function
SG = pdfSmap(DATA, ROW, COL);


% Combining local (SL)  and gloabl (SG) saliency
Smix = mat2gray(SL).*exp( mat2gray(SG) );
% S = Smix.^( log(2)/log(exp(1))); %nonlinear normalization to to diminish the effect of amplification on the map not original paper
S = Smix.^log(sqrt(2))/sqrt(2); %nonlinear normalization to to diminish the effect of amplification on the map paper orijinal
S = imfilter(S, fspecial('gaussian', 5, 5), 'symmetric', 'conv');

end

