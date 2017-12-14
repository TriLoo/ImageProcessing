function [HOG_Sp locations] = HOGFeatures(im, N_Sp, feat);

locations = round([feat(1,:).*size(im,1); feat(2,:).*size(im,2)]);
locations = [max(locations(1,:),9); max(locations(2,:),9)];
locations = [min(locations(1,:),size(im,1)-8); min(locations(2,:),size(im,2)-8)];
    
HOG_Sp = zeros(31,N_Sp);
for i = 1 : N_Sp
    HOG_Sp(:,i) = vl_hog(single(im(locations(1,i)-8:locations(1,i)+8, locations(2,i)-8:locations(2,i)+8)), 17);
end