function map = pixsal( img , param )

summap = 0;
for ssig = param.surroundSig
  ker = mygausskernel( ssig , 2 );
  map_ = mynorm( (img - myconv2(myconv2(img,ker),ker')).^2 , param );  
  if ( param.useNormWeights )
    wt_ = mypeakiness(map_);
  else
    wt_ = 1;
  end
  summap = summap + map_ * wt_;
end
map = mynorm(summap,param);