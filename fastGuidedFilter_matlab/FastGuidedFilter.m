function [ Res ] = FastGuidedFilter( imgI, imgP, samp, rad, eps)
% ---------------------
% Author : smher
% Data   : 2017. 07. 31
% Description:
%       This file implement the fast guided image filter based on " Fast
%       Guided Filter " by Kaiming He and Jian Sun
%       Input :     imgI : The guided image 
%                   imgP : the filtering image
%                   samp : the downsample and upsample value
%                   rad : the readius
%                   eps  : the regulation value 
% ---------------------

imgI = double(imgI);
imgP = double(imgP);

[m, n, z] = size(imgP);
if z == 3
    imgP = rgb2gray(imgP);
end

tic;

% Step 1 : 
downI = subsample(imgI, samp);
downP = subsample(imgP, samp);

downRad = rad / samp;

[m, n] = size(downP);

N = boxFilter(ones(m, n), downRad);

% Step 2 :
% Get the mean filtered result of guidance image I 
meanI = boxFilter(downI, downRad) ./ N;

% Get the mean filtered result of input image P
meanP = boxFilter(downP, downRad) ./ N;

% Get the corr of guidance image I
corrI = boxFilter(downI .* downI, downRad) ./ N;

% Get the corr of input image P
corrIP = boxFilter(downI .* downP, downRad) ./ N;

% Get the Variable of Guidance image I
varI = corrI - meanI .* meanI;

% Get the Cov of I and P
covIP = corrIP - meanI .* meanP;

a = covIP ./ (varI + eps);
b = meanP - a .* meanI;

meanA = boxFilter(a, downRad) ./ N;
meanB = boxFilter(b, downRad) ./ N;

upA = usample(meanA, samp);
upB = usample(meanB, samp);

Res = upA .* imgI + upB;

toc;

imshow(uint8(imgP), []);
figure;
imshow(uint8(Res), []);

end

