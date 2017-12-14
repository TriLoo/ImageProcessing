
function param = default_fast_param()

param = {};

% Note "classic Itti" refers to  "A model of saliency-based visual attention for rapid scene analysis" in PAMI


%%%%%%%%% scale parameters %%%%%%%%%%%%%%%

param.mapWidth = 64;               % this controls the size of the 'Center' scale

param.useMultipleCenterScales = 0; % classic Itti Algorithm uses multiple scales ( "c \in {2,3,4}" ). but here we default to just 1.

param.surroundSig = 5;             % this is the standard deviation of the Gaussian blur applied to obtain the 'surround' scale(s).
                                   % default : one surround scale works fine in my experience. with sigma = 5.
                                   % this can also be an array of surround sigmas, e.g. [ 4 6 ]
                                   % Note: in classic  Itti algorithm, we have ( "delta \in {3,4}\" ).
                                   %  .. **I think** this should correspond to roughly surroundSig = [sqrt(2^3) sqrt(2^4)]

%%%%%%%% normalize maps according to peakiness ? %%%%%

param.useNormWeights = 0;        % 0 = do not weight maps differently , 1 = weight according to peakiness
                                 % in classic Itti algorithm, this is used with local maxima normalization.

param.subtractMin = 1;           % 1 => (subtract min, divide by max) ; 0 => (just divide by max)

%%%%%%%%% channel parameters %%%%%%%%%%%%%

param.channels = 'DI';           % can include following characters: C or D (color), O (orientation), I (intensity), F or X (flicker), M (motion)
                                 % (D is in DKL color space, C is RG , BY color space)
                                 % e.g. use 'IO' to include only intensity and orientation channels.
                                 % note (F or X) and M require temporal input.

param.nGaborAngles = 4;          % number of oriented gabors if there is an 'O' channel
                                 % as an example, 4 implies => 0 , 45 , 90 , 135 degrees
% for video
param.flickmotionT = [1 2 4];    % this is the (number of frames difference) .. e.g., 1 means subtract previous frame from current one.
                                 % see getchan.m to understand how this is used.

param.nMotionAngles = 4;         % number of motion directions if there is an 'M' channel

%%%%%%%%% final operations on saliency map %%%%

param.centerbias = 0;            % apply global center bias (0 =no, 1=yes)
                                 % (using center bias tends to improve predictive performance)

param.blurRadius = 0.04;         % blur final saliency map (sigma=0.04 works well in my experience).
                                 % NOTE: ROC and NSS Scores are VERY sensitive to saliency map blurring. This is 
                                 % highly suggested.