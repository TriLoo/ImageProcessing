function [feat] = SVF(im, Sp, N_Sp, Locpixels)

feat = zeros(1,N_Sp);
img_original = im;
img = rgb2gray(img_original);

k = 3; 
Z = 0;

part_num = 30;
[img_sizeY img_sizeX] = size(img);
cell_sizeY = floor(img_sizeY / part_num);
cell_sizeX = floor(img_sizeX / part_num);
for y = 1 : part_num+1
    k = 3;
    rey = (y-1) * cell_sizeY+1 : y * cell_sizeY;
    if (y == part_num+1) 
        rey = (y-1) * cell_sizeY+1 : img_sizeY;
        if (size(rey,2) == 0)
            continue;
        end
        if (size(rey,2) < k)
            k = (size(rey,2));
        end        
    end
    for x = 1 : part_num+1
        rex = (x-1) * cell_sizeX+1 : x * cell_sizeX;
        if (x == part_num+1) 
            rex = (x-1) * cell_sizeX+1 : img_sizeX;
            if (size(rex,2) == 0)
                continue;
            end
            if (size(rex,2) < k )
                k = (size(rex,2));
            end
        end
  
        part_img = img(rey, rex);
        [U S V] = svd(single(part_img),'econ');
        DS = diag(S);
        B1 = sum(DS(1:k)) / (sum(DS) + eps);
        sal_map(rey, rex) = B1;       
    end
end

renewed_img2 = (sal_map - mean(sal_map(:)))/ std(sal_map(:));


for i = 1 : N_Sp
    feat(i) = mean(renewed_img2(Locpixels{i}));
end
