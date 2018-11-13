function [ fuseImg ] = baseLayerFusion( imgA, imgB, baseA, baseB )
% Author: smh
% Date  : 2018.11.13

% do a laplacian to input layers
lap_kernel = fspecial('laplacian');
imgA_lap = imfilter(imgA, lap_kernel);
imgB_lap = imfilter(imgB, lap_kernel);

% smooth with gaussian low-pass filter, k = 5, sigma = 5
imgA_lap_smooth = imgaussfilt(abs(imgA_lap), 5, 'FilterSize', 11); 
imgB_lap_smooth = imgaussfilt(abs(imgB_lap), 5, 'FilterSize', 11);

% calculate IWP_type
iwpA = zeros(size(imgA));
iwpB = zeros(size(imgB));

% get the initial weight maps (iwp)(
maskA = iwpA >= iwpB;
iwpA(maskA) = 1;
iwpB(1-maskA) = 1;

% use guided filter to get spatial consistency weight masks (fwm: final weight map)
r = 45;
t = 0.3;
fwmA = imguidedfilter(iwpA, imgA, 'NeighborhoodSize', r, 'DegreeOfSmoothing', t);   % imgA is the guided image
fwmB = imguidedfilter(iwpB, imgB, 'NeighborhoodSize', r, 'DegreeOfSmoothing', t);   % imgB is the guided image

% Get the fused base layer
fuseImg = fwmA .* baseA + fwmB .* baseB;

end

