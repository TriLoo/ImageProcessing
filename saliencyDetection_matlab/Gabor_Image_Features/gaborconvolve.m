% GABORCONVOLVE - function for convolving image with log-Gabor filters
%
% Usage: EO = gaborconvolve(im,  nscale, norient, minWaveLength, mult, ...
%			    sigmaOnf, dThetaOnSigma, feedback)
%
% Arguments:
% The convolutions are done via the FFT.  Many of the parameters relate 
% to the specification of the filters in the frequency plane.  
%
%   Variable       Suggested   Description
%   name           value
%  ----------------------------------------------------------
%    im                        Image to be convolved.
%    nscale          = 4;      Number of wavelet scales.
%    norient         = 6;      Number of filter orientations.
%    minWaveLength   = 3;      Wavelength of smallest scale filter.
%    mult            = 2;      Scaling factor between successive filters.
%    sigmaOnf        = 0.65;   Ratio of the standard deviation of the
%                              Gaussian describing the log Gabor filter's
%                              transfer function in the frequency domain
%                              to the filter center frequency. 
%    dThetaOnSigma   = 1.5;    Ratio of angular interval between filter
%                              orientations and the standard deviation of
%                              the angular Gaussian function used to
%                              construct filters in the freq. plane.
%    feedback         0/1      Optional parameter.  If set to 1 a message
%                              indicating which orientation is being
%                              processed is printed on the screen.
%
% Returns:
%
%   EO a 2D cell array of complex valued convolution results
%
%        EO{s,o} = convolution result for scale s and orientation o.
%        The real part is the result of convolving with the even
%        symmetric filter, the imaginary part is the result from
%        convolution with the odd symmetric filter.
%
%        Hence:
%        abs(EO{s,o}) returns the magnitude of the convolution over the
%                     image at scale s and orientation o.
%        angle(EO{s,o}) returns the phase angles.
%   
%
% Notes on filter settings to obtain even coverage of the spectrum
% dthetaOnSigma 1.5
% sigmaOnf  .85   mult 1.3
% sigmaOnf  .75   mult 1.6     (bandwidth ~1 octave)
% sigmaOnf  .65   mult 2.1
% sigmaOnf  .55   mult 3       (bandwidth ~2 octaves)
%                                                       
% For maximum speed the input image should be square and have a 
% size that is a power of 2, but the code will operate on images
% of arbitrary size.  
%
%
% The determination of mult given sigmaOnf is entirely empirical
% What I do is plot out the sum of the filters in the frequency domain
% and see how even the coverage of the spectrum is.
% If there are concentric 'gaps' in the spectrum one needs to
% reduce mult and/or reduce sigmaOnf (which increases filter bandwidth)
%
% If there are 'gaps' radiating outwards then one needs to reduce
% dthetaOnSigma (increasing angular bandwidth of the filters)
%

% For details of log-Gabor filters see: 
% D. J. Field, "Relations Between the Statistics of Natural Images and the
% Response Properties of Cortical Cells", Journal of The Optical Society of
% America A, Vol 4, No. 12, December 1987. pp 2379-2394

% Copyright (c) 2001-2005 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% http://www.csse.uwa.edu.au/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% May 2001

function EO = gaborconvolve(im, nscale, norient, minWaveLength, mult, ...
			    sigmaOnf, dThetaOnSigma, feedback)
    
    if nargin == 7
	feedback = 0;
    end

    if ~isa(im,'double')
	im = double(im);
    end
    
    
[rows cols] = size(im);					
imagefft = fft2(im);                 % Fourier transform of image
EO = cell(nscale, norient);          % Pre-allocate cell array

% Pre-compute some stuff to speed up filter construction

[x,y] = meshgrid( [-cols/2:(cols/2-1)]/cols,...
		  [-rows/2:(rows/2-1)]/rows);
radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
radius(round(rows/2+1),round(cols/2+1)) = 1; % Get rid of the 0 radius value in the middle 
                                             % so that taking the log of the radius will 
                                             % not cause trouble.

% Precompute sine and cosine of the polar angle of all pixels about the
% centre point					     

theta = atan2(-y,x);              % Matrix values contain polar angle.
                                  % (note -ve y is used to give +ve
                                  % anti-clockwise angles)
sintheta = sin(theta);
costheta = cos(theta);
clear x; clear y; clear theta;      % save a little memory

thetaSigma = pi/norient/dThetaOnSigma;  % Calculate the standard deviation of the
                                        % angular Gaussian function used to
                                        % construct filters in the freq. plane.
% The main loop...

for o = 1:norient,                   % For each orientation.
  if feedback
     fprintf('Processing orientation %d \r', o);
  end
    
  angl = (o-1)*pi/norient;           % Calculate filter angle.
  wavelength = minWaveLength;        % Initialize filter wavelength.

  % Pre-compute filter data specific to this orientation
  % For each point in the filter matrix calculate the angular distance from the
  % specified filter orientation.  To overcome the angular wrap-around problem
  % sine difference and cosine difference values are first computed and then
  % the atan2 function is used to determine angular distance.

  ds = sintheta * cos(angl) - costheta * sin(angl);     % Difference in sine.
  dc = costheta * cos(angl) + sintheta * sin(angl);     % Difference in cosine.
  dtheta = abs(atan2(ds,dc));                           % Absolute angular distance.
  spread = exp((-dtheta.^2) / (2 * thetaSigma^2));      % Calculate the angular filter component.

  for s = 1:nscale,                  % For each scale.

    % Construct the filter - first calculate the radial filter component.
    fo = 1.0/wavelength;                  % Centre frequency of filter.

    logGabor = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
    logGabor(round(rows/2+1),round(cols/2+1)) = 0; % Set the value at the center of the filter
                                                   % back to zero (undo the radius fudge).

    filter = fftshift(logGabor .* spread); % Multiply by the angular spread to get the filter
                                           % and swap quadrants to move zero frequency 
                                           % to the corners.

    % Do the convolution, back transform, and save the result in EO
    EO{s,o} = ifft2(imagefft .* filter);    

    wavelength = wavelength * mult;       % Finally calculate Wavelength of next filter
  end                                     % ... and process the next scale

end  % For each orientation

if feedback, fprintf('                                        \r'); end












