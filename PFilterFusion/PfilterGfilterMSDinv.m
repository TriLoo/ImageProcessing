function [ resImg ] = PfilterGfilterMSDinv( fuseLevels )
% Author: smh
% Date  : 2018.11.15

scales = length(fuseLevels);

resImg = zeros(size(fuseLevels{1}));   % Fused image

for i = 1:1:scales
    resImg = resImg + fuseLevels{i};
end

end

