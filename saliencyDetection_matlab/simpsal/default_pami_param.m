
function param = default_pami_param()

param = default_fast_param;

param.mapWidth = 64;

% DIO is better, but change to CIO for more faithful implementation
param.channels = 'CIO'; 

param.useMultipleCenterScales = 1;
param.surroundSig = [ 2 8 ];
param.useNormWeights = 1;
