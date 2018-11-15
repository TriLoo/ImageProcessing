function [ imgRes ] = PfilterFusion( imgA, imgB, params)
% Author: smh
% Date  : 2018.10.28

% params contains all parameters needed by gfilter and pfilter and guided
% filter

if(~isfloat(imgA))    %  OR: isa(imgA, 'double')
    imgA = im2double(imgA);
end
if(~isfloat(imgB))
    imgB = im2double(imgB);
end

if(nargin ~= 3)
    fprintf('The number of input should be (imgA, imgB, params). \n');
end

% PART I: MSD of input images using gfilter + pfilter
pfilter_w = params.pfilter_w;
pfilter_sigma_ds = params.pfilter_sigma_ds;
pfilter_sigma_rs = params.pfilter_sigma_rs;

coeffAs = PfilterGfilterMSD(imgA, pfilter_w, pfilter_sigma_ds, pfilter_sigma_rs);
coeffBs = PfilterGfilterMSD(imgB, pfilter_w, pfilter_sigma_ds, pfilter_sigma_rs);


% PART II: Fuse the coefficient matrices
scales = length(coeffAs);
fuseLevels = cell(1, scales);

for i = 1:1:scales-1      % Detail levels fusion
    fuseLevels{i} = detailLayerFusion(coeffAs{i}, coeffBs{i});
end

% Base level fusion
fuseLevels{scales} = baseLayerFusion(imgA, imgB, coeffAs{scales}, coeffBs{scales});


% PART III: Restore the fused image
imgRes = PfilterGfilterMSDinv(fuseLevels);

end

