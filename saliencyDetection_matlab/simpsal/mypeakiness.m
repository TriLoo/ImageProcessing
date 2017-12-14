function p = mypeakiness(m)

[locmax_avg,locmax_num] = mexLocalMaximaGBVS( mat2gray(m)  , 0.1 );

if (locmax_num > 1)
  p = (1 - locmax_avg)^2;
else
  p = 1;
end