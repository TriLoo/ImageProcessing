% 
%  Written by Jufeng Zhao and his parterns
% please refer to our paper:
% *Jufeng Zhao, Huajun Feng, Zhihai Xu, Qi Li, Infrared image enhancement throughsaliency feature analysis based on multi-scale decomposition,Infrared Physics & Technology, 2014, 62:86�C93
% 
%%  Image in
filename = '../datas/lena.bmp';

hdr0 = imread(filename);
 Si = size(size(hdr0));
if Si(2)==3
hdr = rgb2gray(hdr0);
else
   hdr = hdr0 ; 
end

%  figure,imshow(hdr ,[]);
%%
%% MultiScale Decomposition

% three Levels
% L1 = 5.0e-5; L2 = 3.0e-2;  % lambda, regularized factor in Eq.(4)
% LL = hdr;
% double_Ori = im2double(LL);
% u1 = L0Smoothing(LL, L1 );  
% u2 = L0Smoothing(LL, L2 );
% Med{1} = double_Ori-u1;
% Med{2} = u1-u2;
% Med{3} =  u2;

% four Levels
L1 = 5.0e-5; L2 = 5.0e-4;  L3 = 5.0e-2;  % lambda, regularized factor in Eq.(4)
LL = hdr;
double_Ori = im2double(LL);
u1 = L0Smoothing(LL, L1 );  
u2 = L0Smoothing(LL, L2 );
u3 = L0Smoothing(LL, L3 );
Med{1} = double_Ori-u1;
Med{2} = u1-u2;
Med{3} =  u2 - u3;
Med{4} =  u3;
%%
%%  Enhancement in SubSclae
% LocalWin_width, LocalWin_height -- the largest size of local window along with and height direction

LocalWin_width = 13;
LocalWin_height = 13; 

[ final_out ] = MultiScale_Windows(Med, LocalWin_width, LocalWin_height);

%%  Synthesis

% weight = [0.1 0.6 0.3];  % synthetic weight for three levels
weight = [0.1 0.4  0.4 0.1];  % synthetic weight for four levels

FinalOut = Synthesis(final_out, weight );


figure,imshow(FinalOut);
