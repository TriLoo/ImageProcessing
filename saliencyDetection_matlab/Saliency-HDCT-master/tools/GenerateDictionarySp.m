function [Dic] = GenerateDictionarySp(N_Sp, Sp, Locpixels, im, imhsv, imlab)

imR = im(:,:,1);  imG = im(:,:,2);  imB = im(:,:,3);
imH = imhsv(:,:,1); imS = imhsv(:,:,2);
imL = imlab(:,:,1);  ima = imlab(:,:,2);  imb = imlab(:,:,3);

[GX, GY] = gradient(im2double(im(:,:,1)));
GR = abs(GX) + abs(GY);
[GX, GY] = gradient(im2double(im(:,:,2)));
GG = abs(GX) + abs(GY);
[GX, GY] = gradient(im2double(im(:,:,3)));
GB = abs(GX) + abs(GY);

imR_Sup = zeros(N_Sp, 1);
imG_Sup = zeros(N_Sp, 1);
imB_Sup = zeros(N_Sp, 1);
imR_Gra = zeros(N_Sp, 1);
imG_Gra = zeros(N_Sp, 1);
imB_Gra = zeros(N_Sp, 1);
imL_Sup = zeros(N_Sp, 1);
ima_Sup = zeros(N_Sp, 1);
imb_Sup = zeros(N_Sp, 1);
imH_Sup = zeros(N_Sp, 1);
imS_Sup = zeros(N_Sp, 1);

for j = 1 : N_Sp
	loc = Locpixels{j};
    imR_Sup(j) = mean(imR(loc));
	imG_Sup(j) = mean(imG(loc));
    imB_Sup(j) = mean(imB(loc));
    imR_Gra(j) = mean(GR(loc));
	imG_Gra(j) = mean(GG(loc));
    imB_Gra(j) = mean(GB(loc));
	imL_Sup(j) = mean(imL(loc));
	ima_Sup(j) = mean(ima(loc));
    imb_Sup(j) = mean(imb(loc));
	imH_Sup(j) = mean(imH(loc));
	imS_Sup(j) = mean(imS(loc));
end
Dic = [imR_Sup imG_Sup imB_Sup imR_Gra imG_Gra imB_Gra imL_Sup ima_Sup imb_Sup imH_Sup imS_Sup];