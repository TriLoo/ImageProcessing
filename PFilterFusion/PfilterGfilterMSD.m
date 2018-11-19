function [ coeffImgs ] = PfilterGfilterMSD( img, w, sigma_ds, sigma_rs )
% Author: smh
% Date  : 2018.10.28

% Inputs:
%       w: the window radius of pfilter, default = 3.
%       sigma_ds, sigma_rs: the sigma values needed by gaussian filter and
%                           propagation filter.

% preprocess: convert color image to gray-scale image
% img = rgb2gray(img);   % input image default to gray-scale image

% [m, n] = size(img);

scales = length(sigma_ds);   % i.e. the number of levels: N

% coeffMats = zeros(m, n);
coeffImgs = cell(scales+1, 1);

% Detail Level 1
% SI_curr = imgaussfilt(img, sigma_ds(1), 'FilterSize', 11);
SI_curr = imgaussfilt(img, sigma_ds(1));
Ipf_last = pfilter(SI_curr, SI_curr, w, [sigma_ds(1), sigma_rs(1)]); 
coeffImgs{1} = img - Ipf_last;

for i  = 2:1:scales-1   % Levels from 2 to N - 1
%     SI_curr = imgaussfilt(Ipf_last, sigma_ds(i), 'FilterSize', 11);
    SI_curr = imgaussfilt(Ipf_last, sigma_ds(i));
    Ipf_curr = pfilter(SI_curr, SI_curr, w, [sigma_ds(i), sigma_rs(i)]);
    
    coeffImgs{i} = Ipf_last - Ipf_curr;   % Detail level i
    
%     SI_last = SI_curr;
    Ipf_last = Ipf_curr;
end

% Calculate the base level
B = imgaussfilt(Ipf_curr, sigma_ds(scales));

% Calculate detail level scales:
coeffImgs{scales} = Ipf_curr - B;

coeffImgs{scales+1} = B;

end
