function imOut = my_imresize( imIn , resizeFrac )

if ( numel(size(imIn)) < 4 )
  imOut = imresize( imIn, resizeFrac );
else
  
  firstOut = imresize( imIn(:,:,:,1) , resizeFrac );
  imOut = repmat( firstOut , [ 1 1 1 size(imIn,4) ] );

  for ind = 2 : size(imIn,4)
    imOut(:,:,:,ind) = imresize( imIn(:,:,:,ind) , resizeFrac );
  end

end