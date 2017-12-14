function [map,chanmaps,maps,chans] = simpsal( img , param );

% outputs:
%
% map : final saliency map
% chanmaps : final saliency map for each channel
% maps : saliency map for each sub-channel, for each channel
% chans : feature maps for each sub-channel, for each channel
%--------------------------------------------------------------

% read in file if img is filename
if ( strcmp(class(img),'char') == 1 ) img = imread(img); end

% convert to double if image is uint8
if ( strcmp(class(img),'uint8') == 1 ) img = double(img)/255; end

% use default parameters if no parameters input
if ( nargin == 1 )
  param = default_pami_param;
end

% compute Gabor filters 
if ( isfield(param,'nGaborAngles') && ismember( 'O' , param.channels ) )
  nAngles = param.nGaborAngles;    
  maxAngle = 180 - (180/nAngles);
  angs = linspace(0,maxAngle,nAngles);
  param.gabor = {};
  for ang = angs,
    param.gabor{ end + 1 } = simpleGabor( ang );
  end
end

% determine size and number of Center scales
mapSize = [ round(size(img,1) / size(img,2 ) * param.mapWidth) param.mapWidth ];
imgs = {};
if ( param.useMultipleCenterScales == 0 )
  imgs{1} = my_imresize( img , mapSize );
else
  imgs{1} = my_imresize( img , mapSize * 2 );
  imgs{2} = my_imresize( img , mapSize );
  imgs{3} = my_imresize( img , round( mapSize / 2 ) );
end


if (size(img,3)~=3) 
  param.channels = setdiff( param.channels , 'CD' );
end

chans = {}; maps = {};
for ci = 1 : numel( param.channels )
  % get feature maps for this channel

  chans{ci} = getchan( imgs , param.channels(ci) , param );

  % compute saliency of each map
  for cj = 1 : length(chans{ci})    
    maps{ci}{cj} = zeros( size(chans{ci}{cj}) );    
    for fr = 1 : size( chans{ci}{cj} , 3 )
      maps{ci}{cj}(:,:,fr) = pixsal( chans{ci}{cj}(:,:,fr) , param );
    end
  end
  
  % sum together maps within this channel according to weight
  chanmap = 0;
  sumW = 0;
  for cj = 1 : length(chans{ci})    
    if ( param.useNormWeights ) wj = mypeakiness( maps{ci}{cj}(:,:,end) );
    else wj = 1; end

    chanmap = chanmap + wj * imresize(maps{ci}{cj}, mapSize );
    sumW = sumW + wj;
  end
  chanmaps{ci} = chanmap / sumW;
end

% sum together maps across channels
map = 0;
for i = 1 : length( chanmaps )
  if (param.useNormWeights) wt = mypeakiness( chanmaps{i}(:,:,end) );
  else wt = 1; end
  map = map + wt * chanmaps{i};
end

% apply final blur
if ( param.blurRadius > 0 )
  ker = mygausskernel( param.blurRadius * size(map,1) , 1.5 );
  for fr = 1 : size(map,3)
    map(:,:,fr) = myconv2(myconv2(map(:,:,fr),ker),ker');
  end
end

% apply global center bias
if ( param.centerbias )
  h = size(map,1);
  w = size(map,2);
  map = map .* (gausswin(h,1) * gausswin(w,1)');
end

map = mynorm(map,param);