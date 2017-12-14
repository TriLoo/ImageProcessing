function [Features] = ColorFeatures(Sp, N_Sp, Locpixels, im, imlab, distances) 

Features = zeros(40, N_Sp);
im = im2double(im);
iml = mat2gray(imlab(:,:,1)); ima = mat2gray(imlab(:,:,2)); imb = mat2gray(imlab(:,:,3));
imR = im(:,:,1);  imG = im(:,:,2);   imB = im(:,:,3);  
imhsv = rgb2hsv(im);
imh = mat2gray(imhsv(:,:,1)); ims = mat2gray(imhsv(:,:,2)); imv = mat2gray(imhsv(:,:,3));
    
expdist = exp(-1/0.16 * distances);
    
histrgb = zeros(N_Sp, 48);
histlab = zeros(N_Sp, 40);
histh = zeros(N_Sp, 8);
hists = zeros(N_Sp, 8);
for i = 1 : N_Sp
	histrgb(i,:) = [imhist(imR(Locpixels{i}),16); imhist(imG(Locpixels{i}),16); imhist(imB(Locpixels{i}),16)] ./ length(Locpixels{i});
	histlab(i,:) = [imhist(iml(Locpixels{i}),8); imhist(ima(Locpixels{i}),16); imhist(imb(Locpixels{i}),16)] ./ length(Locpixels{i});
	histh(i,:) = imhist(imh(Locpixels{i}),8) ./ length(Locpixels{i});
	hists(i,:) = imhist(ims(Locpixels{i}),8) ./ length(Locpixels{i});
end
    
    
for i = 1 : N_Sp
    % The average color values
	Features(1,i) = mean(imR(Locpixels{i}));
	Features(2,i) = mean(imG(Locpixels{i}));
	Features(3,i) = mean(imB(Locpixels{i}));
	Features(4,i) = mean(iml(Locpixels{i}));
	Features(5,i) = mean(ima(Locpixels{i}));
	Features(6,i) = mean(imb(Locpixels{i}));
	Features(7,i) = mean(imh(Locpixels{i}));
	Features(8,i) = mean(ims(Locpixels{i}));
    Features(9,i) = mean(imv(Locpixels{i}));
        
    % Color Histogram Features
	Features(10,i) = sum(chi_square_statistics_fast(histrgb(i,:),histrgb));
	Features(11,i) = sum(chi_square_statistics_fast(histlab(i,:),histlab));
	Features(12,i) = sum(chi_square_statistics_fast(histh(i,:),histh));
	Features(13,i) = sum(chi_square_statistics_fast(hists(i,:),hists));
end
    
Rdist = zeros(N_Sp,N_Sp);  Gdist = zeros(N_Sp,N_Sp);  Bdist = zeros(N_Sp,N_Sp);
ldist = zeros(N_Sp,N_Sp);  adist = zeros(N_Sp,N_Sp);  bdist = zeros(N_Sp,N_Sp);
hdist = zeros(N_Sp,N_Sp);  sdist = zeros(N_Sp,N_Sp);  vdist = zeros(N_Sp,N_Sp);
for i = 1 : N_Sp
	Rdist(i,:) = pdist2(Features(1,:)', Features(1,i), 'euclidean');
	Gdist(i,:) = pdist2(Features(2,:)', Features(2,i), 'euclidean');
	Bdist(i,:) = pdist2(Features(3,:)', Features(3,i), 'euclidean');
	ldist(i,:) = pdist2(Features(4,:)', Features(4,i), 'euclidean');
	adist(i,:) = pdist2(Features(5,:)', Features(5,i), 'euclidean');
	bdist(i,:) = pdist2(Features(6,:)', Features(6,i), 'euclidean');
	hdist(i,:) = pdist2(Features(7,:)', Features(7,i), 'euclidean');
	sdist(i,:) = pdist2(Features(8,:)', Features(8,i), 'euclidean');
	vdist(i,:) = pdist2(Features(9,:)', Features(9,i), 'euclidean');
end

expR = exp(-1/0.16 * Rdist);
expG = exp(-1/0.16 * Gdist);
expB = exp(-1/0.16 * Bdist);
expl = exp(-1/0.16 * ldist);
expa = exp(-1/0.16 * adist);
expb = exp(-1/0.16 * bdist);
exph = exp(-1/0.16 * hdist);
exps = exp(-1/0.16 * sdist);
expv = exp(-1/0.16 * vdist);
    
for i = 1 : N_Sp 
	Features(14,i) = sum(Rdist(i,:));       %
	Features(15,i) = sum(Gdist(i,:));
	Features(16,i) = sum(Bdist(i,:));
	Features(17,i) = sum(ldist(i,:));
	Features(18,i) = sum(adist(i,:));
	Features(19,i) = sum(bdist(i,:));
	Features(20,i) = sum(hdist(i,:));
	Features(21,i) = sum(sdist(i,:));
	Features(22,i) = sum(vdist(i,:));
        
	Features(23,i) = sum(Rdist(i,:) .* expdist(i,:));   % local contrast
	Features(24,i) = sum(Gdist(i,:) .* expdist(i,:));
	Features(25,i) = sum(Bdist(i,:) .* expdist(i,:));
	Features(26,i) = sum(ldist(i,:) .* expdist(i,:));
	Features(27,i) = sum(adist(i,:) .* expdist(i,:));
	Features(28,i) = sum(bdist(i,:) .* expdist(i,:));
	Features(29,i) = sum(hdist(i,:) .* expdist(i,:));
	Features(30,i) = sum(sdist(i,:) .* expdist(i,:));
	Features(31,i) = sum(vdist(i,:) .* expdist(i,:));
        
	Features(32,i) = sum(distances(i,:) .* expR(i,:));  % element distribution
	Features(33,i) = sum(distances(i,:) .* expG(i,:));
	Features(34,i) = sum(distances(i,:) .* expB(i,:));
	Features(35,i) = sum(distances(i,:) .* expl(i,:));
	Features(36,i) = sum(distances(i,:) .* expa(i,:));
	Features(37,i) = sum(distances(i,:) .* expb(i,:));
	Features(38,i) = sum(distances(i,:) .* exph(i,:));
	Features(39,i) = sum(distances(i,:) .* exps(i,:));
	Features(40,i) = sum(distances(i,:) .* expv(i,:));
end