function [ imgRes ] = PfilterFusion( imgA, imgB, params)
% Author: smh
% Date  : 2018.11.13

if(~isfloat(imgA))    %  OR: isa(imgA, 'double')
    imgA = im2double(imgA);
end
if(~isfloat(imgB))
    imgB = im2double(imgB);
end

if(nargin ~= 3)
    fprintf('The number of input should be (imgA, imgB, params). \n');
end

gaussian_sigmas = params.gaussian_sigma;
scales = length(gaussian_sigma);   % levels of multi-scale decomposition

% Step 1: remove the fine-scale details with gaussian filter
imgA_gaussian = imgaussfilt(imgA, params.gaussian_sigma);
imgB_gaussian = imgaussfilt(imgB, params.gaussian_sigma);

% Step 2: Extract the edge features with propagation filter with multiple
% scales
pfilter_ws = params.pfilter_w;
pfilter_sigmas = params.pfilter_sigma;

imgA_pfilter = pfilter(imgA_gaussian, params.pfilter_w, params.pfilter_sigma); 
imgB_pfilter = pfilter(imgB_gaussian, params.pfilter_w, params.pfilter_sigma);

% Step 3: Obtain details at multiple scales


% Step 4: 




end

