list=dir( 'images_beta_normalised\*.jpeg' );

for i=1:length(list)
    im=imread( sprintf('%s', list(i).name ) );
    [h dab] = stainSeparator(im);

    A=sscanf( list(i).name, 'good_IMG_%d.jpg' )

    
    imwrite( h, sprintf('images_hStain\\hStain_norm_IMG_%04d.tif', A ) );

end