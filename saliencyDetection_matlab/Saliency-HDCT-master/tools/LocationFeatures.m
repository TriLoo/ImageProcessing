function [Features Locpixels distances] = LocationFeatures(Sp, N_Sp, im) 

% Features = zeros(4,N_Sp);

for i = 1 : N_Sp
   [x y] = find(Sp==i);
   
   
   loc = (y-1)*size(im,1) + x;
   Locpixels{i} = loc;
   
   lengthofpix = length(loc);
   % mean
   Features(1,i) = mean(x) / size(im,1);
   Features(2,i) = mean(y) / size(im,2);
   
   % 10th, 90th
%    Features(3,i) = x(round(lengthofpix*0.1)) / size(im,1);
%    Features(4,i) = y(round(lengthofpix*0.1)) / size(im,2);
%    Features(5,i) = x(round(lengthofpix*0.9)) / size(im,1);
%    Features(6,i) = y(round(lengthofpix*0.9)) / size(im,2);
   
   Features(3,i) = lengthofpix;
end

for i = 1 : N_Sp
    distances(i,:) = pdist2(Features(1:2,:)', Features(1:2,i)', 'euclidean')';
end

