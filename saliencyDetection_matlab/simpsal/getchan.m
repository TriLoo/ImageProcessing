function chan = getchan(imgs,channel,param)

chan = {};

% iterate over center scales
for scaleI = 1 : length(imgs)  

  img = imgs{scaleI};

  if ( channel == 'I' )
    
    chan{end+1} = squeeze( mean(img,3) );
    
  elseif ( channel == 'C' )
    
    if ( size(img,3) == 3 )
      lum = squeeze( mean(img,3) ) + 0.01;    
      chan{end+1} = squeeze(abs(img(:,:,3,:) - min( img(:,:,1,:) , img(:,:,2,:)) )) ./ lum; % b - y (r,g)
      chan{end+1} = squeeze(abs(img(:,:,1,:) - img(:,:,2,:)))  ./ lum; % r - g 
    end

  elseif ( channel == 'D' )
    
    if ( size(img,3) == 3 )
      map = img;      
      for fr = 1 : size(img,4)
        map(:,:,:,fr) = rgb2dkl( map(:,:,:,fr) );
      end      
      for i = 1 : 3
        chan{end+1} = squeeze( map(:,:,i,:) );
      end
    end

  elseif ( channel == 'O' )

    img = squeeze( mean(img,3) );
    
    for angi = 1 : param.nGaborAngles
      map = img;
      for fr = 1 : size(img,3)
        f0 = myconv2(img(:,:,fr),param.gabor{angi}.g0);
        f90 = myconv2(img(:,:,fr),param.gabor{angi}.g90);
        map(:,:,fr) = attenuateBordersGBVS( abs(f0) + abs(f90) , 13 );
      end
      chan{end+1} = map;
    end
        
  elseif ( channel == 'F' ) % flicker channel with multiple image differences
    
    Ts = param.flickmotionT;
    for T = Ts,
      d = abs( img(:,:,:,T+1:end) - img(:,:,:,1:end-T) );
      d = squeeze(max( d , [] , 3 ));    
      z = repmat( 0 , [ size(img,1) size(img,2) size(img,4) ] );
      z(:,:,T+1:end) = d;
      z(:,:,1:T) = repmat( z(:,:,T+1) , [ 1 1 T ] );
      chan{end+1} = z;
    end
    
  elseif ( channel == 'X' ) % flicker channel with only one frame difference (faster than (and equivalent to) F with param.flickerMotionT = [1] )
    
    lum = squeeze( mean(img,3) );
    f = abs( diff( lum , 1 , 3 ) );
    z = repmat( 0 , [ size(img,1) size(img,2) size(img,4) ] );
    z(:,:,2:end) = f;
    z(:,:,1) = z(:,:,2);
    chan{end+1} = z;
    
  elseif ( channel == 'M' )
    lum = squeeze( mean(img,3) );
    
    nAngles = param.nMotionAngles;    
    maxAngle = 180 - (180/nAngles);
    ang = linspace(0,maxAngle,nAngles);
        
    lumshift = repmat( 0 , [ size(lum) numel(ang) ] );
    for fri = 1 : size( lum , 3 )
      for angi = 1 : numel( ang )    
        lumshift( : , : , fri , angi ) = shiftImage( lum(:,:,fri) , ang(angi) );
      end
    end
    
    for T = param.flickmotionT,
      for angi = 1 : numel(ang)
        m = abs( lum(:,:,T+1:end) .* lumshift(:,:,1:end-T,angi) - lum(:,:,1:end-T) .* lumshift(:,:,T+1:end,angi) );
        mm = repmat( 0 , size(lum) );
        mm(:,:,T+1:end) = m;
        chan{end+1} = mm;
      end
    end

  end  % end if chan == M

end % end scale loop
