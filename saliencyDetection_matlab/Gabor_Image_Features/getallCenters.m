list=dir( 'images_beta_normalised\*.jpeg' );

for i=1:length(list)
    close all;
    centers = NEW_lipsym(  sprintf('images_beta_normalised\\%s', list(i).name )  );
    A=sscanf( list(i).name, 'good_IMG_%d.jpg' )

    
    imwrite( centers, sprintf('result_LIPSyM_beta_normalised\\LIPSyM_norm_IMG_%04d.tif', A ) );
    clear centers;
end

