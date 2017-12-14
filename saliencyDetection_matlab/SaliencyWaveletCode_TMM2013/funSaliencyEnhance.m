function smap = funSaliencyEnhance(S)

% Enhance the saliency S = Smix which is the combination of local (SL) and global (SG) saliency
smap = mat2gray(S);
%% the following code is for enhancing the saliency map
[temp_row temp_col] = size(smap);
[salient_row salient_col salient_value] = find(smap > 0.8);
salient_distance = zeros(temp_row, temp_col);
for temp_i = 1:temp_row
    for temp_j = 1:temp_col
        salient_distance(temp_i,temp_j) = min(sqrt((temp_i - salient_row).^2 + (temp_j - salient_col).^2));
    end
end
salient_distance = mat2gray(salient_distance);
smap = smap .* (1 - salient_distance);
smap = mat2gray(smap);